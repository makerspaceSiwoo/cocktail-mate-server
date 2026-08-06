# Vertex sensory teacher batch (local artifact boundary)

This pipeline prepares and validates the proposed sensory-48 teacher run. It is
approved only as a local experiment: it does not approve a database schema or
production rollout. The code imports no Google SDK, makes no network or Vertex
call, reads no database, and never reads a service-account JSON file.

## Frozen input

The source CSV must contain `cocktail_id` and `normalized_recipe_json`. Other
columns, including cocktail names, are ignored. `recipe_facts` is accepted as a
focused-fixture alias; if both recipe columns are populated, their parsed JSON
must agree. The frozen output contains
`cocktail_id,recipe_facts,recipe_source_column`, sorted by numeric ID, so the
production manifest can preserve whether the primary column or fixture alias was
used.

Recipe facts are reduced before freezing. Each ingredient keeps only
`canonical_name`, `category`, one normalized amount (ratio preferred, otherwise
millilitres), and `presence_only`. The cocktail-level facts are method, mixing
ice, serving ice, carbonation, garnish, and estimated pre-dilution ABV/status.
Nulls, display names, ingredient IDs/order, and normalization boilerplate are
excluded. Neither cocktail name nor public ID is included in a model prompt.

The checked-in
`app/sensory_embedding/data/sensory_axis_registry_48_ae_v2.csv` preserves the
reviewed Korean/English labels and definitions in exact axis order 0–47. It is
hash-pinned and marked `APPROVED_LOCAL_EXPERIMENT`; this is not DB approval.

```bash
python -m scripts.sensory_vertex_batch freeze \
  --input sensory-lab/normalized_cocktails.csv \
  --cohort-ids taste-data/cocktail-taste-descriptions.csv \
  --output sensory-batch/frozen-recipes.csv
```

All outputs are create-only and written atomically. Existing artifacts are never
replaced.

## Request and shard contract

The production corpus is exactly the pinned current 602-ID cohort × 48 axes =
28,896 records. The normalized source may contain additional rows, but they
cannot enter a full run. Full `freeze` and `build` commands require an explicit
602-ID cohort CSV and reject any other ID set. Eight shards are assigned by
`axis_order % 8`; each contains 3,612 records and
six complete axes for every cocktail. The manifest binds the input hash, row
count, pinned cohort source/ID-set hashes, registry and prompt hashes, request
configuration, model, project/location, SDK version, shard hashes/specifications,
cost estimate, reserve, timestamp, and run ID.

The model contract is:

- model `gemini-2.5-flash`
- default project `gen-lang-client-0477982146`
- Vertex location `global` (the live adapter uses a separate valid GCS bucket
  region)
- `responseMimeType: text/x.enum`
- response schema `STRING`, enum exactly `A,B,C,D,E`
- `responseLogprobs: true`, `logprobs: 20`
- `temperature: 1.0`, `topP: 1.0`, `maxOutputTokens: 32`
- `thinkingConfig.thinkingBudget: 0`

Every prompt contains minimal recipe facts, one reviewed axis label/definition,
the monotonic intensity rubric `A=none, B=weak, C=medium, D=strong, E=very
strong`, and a demand for exactly one code. The downstream preference projection
may use a nonmonotonic basic-taste utility; that does not alter teacher intensity
labels.

```bash
python -m scripts.sensory_vertex_batch build \
  --input sensory-batch/frozen-recipes.csv \
  --cohort-ids taste-data/cocktail-taste-descriptions.csv \
  --output-dir sensory-batch/run-20260806 \
  --run-id sensory-20260806-a \
  --sdk-version REVIEWED_SDK_VERSION
```

The conservative planning envelope is 845 input and 32 output tokens per
record. The manifest also records UTF-8 prompt-byte diagnostics. Before any paid
job is separately created, a reviewed pilot must supply measured token counts;
creation must remain blocked if any count exceeds 845. A token-count JSON object
can be attached with `--pilot-token-counts`.

## Cost and job safety

The estimate uses Vertex Batch prices of $0.15 per million input tokens and
$1.25 per million output tokens. At the conservative envelopes, the full run is
$4.818408 before the fixed $0.50 historical reserve, or $5.318408 with that
reserve. This remains below the $7.50 soft stop and $10 hard limit.

The pure ledger guard implements:

- soft stop at a projected cumulative $7.50 (manual override required);
- unconditional creation block at a projected cumulative $10.00;
- at most one non-terminal job;
- unknown, missing, or failed-to-refresh state is `UNKNOWN` and blocks another
  job;
- only an explicitly known terminal state permits considering another job.

The separate `scripts.sensory_vertex_live` adapter owns `create`, `status`,
`download`, and `cleanup`. It calls ADC through official Google libraries and
uses the offline guard first. `GEMINI_API_KEY` and `GOOGLE_API_KEY` are removed
from the live process environment. The code never opens credential JSON
manually; the official ADC loader may use `GOOGLE_APPLICATION_CREDENTIALS`.

GCS input/output objects must be scoped to the run prefix and carry run ID,
manifest hash, shard index, object hash, and schema version metadata. The bucket
lifecycle contract deletes run objects after one day. Any future cloud adapter
must verify both metadata and lifecycle before job creation.

## Recorded response parsing

Pass eight locally downloaded/recorded output shards in shard-index order:

```bash
python -m scripts.sensory_vertex_batch parse \
  --manifest sensory-batch/run-20260806/manifest.json \
  --response responses-00.jsonl \
  --response responses-01.jsonl \
  --response responses-02.jsonl \
  --response responses-03.jsonl \
  --response responses-04.jsonl \
  --response responses-05.jsonl \
  --response responses-06.jsonl \
  --response responses-07.jsonl \
  --output sensory-batch/accepted.jsonl \
  --quarantine sensory-batch/quarantine.jsonl \
  --summary sensory-batch/parse-summary.json
```

The parser requires one exact selected A–E code and finite top-candidate
logprobs for all five labels. Whitespace token variants such as `"A"` and
`" A"` are combined with log-sum-exp before normalizing the complete A–E
distribution. Missing labels, malformed responses, and remote error records are
quarantined rather than imputed. Accepted records preserve a canonical response
hash and the raw JSONL-line hash. A shard line-count mismatch is a structural
error, not a recoverable quarantine.

Only a complete registry-ordered set of 48 accepted axes can become a
projection-ready record:

```bash
python -m scripts.sensory_vertex_batch project \
  --input sensory-batch/accepted.jsonl \
  --output sensory-batch/projection-ready.jsonl
```

Each output preserves all 240 ordered probabilities, per-axis response/raw
hashes, the registry hash, and the teacher source-content hash used by the
deterministic sensory-48 projection. `--allow-partial` exists only for focused
offline tests and reviewed pilots; it does not relax the 48-axis requirement for
an individual cocktail.
