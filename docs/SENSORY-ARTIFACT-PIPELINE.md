# Sensory artifact pipeline

`scripts/build_sensory_artifacts.py` is the offline boundary from parsed Vertex
teacher distributions to reviewable local artifacts. It has no database or
network path.

## Input gate

The input must be projection-ready JSONL created by:

```bash
python -m scripts.sensory_vertex_batch project \
  --input parsed-distributions.jsonl \
  --output projection-ready.jsonl
```

Every cocktail record must contain all 48 registry-ordered axes, each with five
finite A-E probabilities that sum to one. The loader verifies the flattened
raw240 values, registry SHA-256, source SHA-256, and all recorded Vertex
response hashes before projecting anything.

Production mode accepts only the pinned current cohort:

- 602 unique cocktail IDs
- integer-ID cohort SHA-256
  `56e77646b60ad9b45cbdcd43f4807dde994ef40b1d5e4461dbfa41ca2d59c05f`

The integer-ID cohort hash and graph48's canonical string-ID hash are different
contracts and are stored separately. `--allow-partial` is an explicit
test-only escape hatch and still requires at least six cocktails for top-5.

## Build

```bash
python -m scripts.build_sensory_artifacts \
  --input projection-ready.jsonl \
  --output-dir sensory-graph-artifacts/run-20260806-01 \
  --run-id run-20260806-01 \
  --clusters 7 \
  --report-only
```

The default S² settings use 450 graph-force iterations and 16 deterministic
starts. `--enforce-quality` makes any failed layout acceptance gate abort the
build; `--report-only` retains the complete diagnostics for local inspection.

The output directory is create-only. All content is built in a same-filesystem
staging directory, each file is flushed and created without replacement, and
the completed directory is renamed into place. A second run cannot overwrite
an existing result. `manifest.json` is written last.

## Outputs

- `raw240.csv`: exact registry-order A-E teacher probabilities.
- `graph48.csv`: category-balanced, unit-L2 graph vectors.
- `preference48.csv`: unnormalized user-query MIPS vectors.
- `canonical-run.csv`: exact graph48 run/vector/ID hashes.
- `graph48-directed-top5.csv`: five exact-cosine recommendations per cocktail.
- `graph48-union-edges.csv`: either-direction union of the directed top-5.
- `spherical-graph-public.json`: graph-only S² cocktail coordinates, the same
  canonical directed rows and union edges, and layout quality diagnostics.
- `manifest.json`: source, cohort, registry, contracts, vector-set, layout, and
  per-file hashes.

The S² solver consumes the fixed node/edge topology; it does not reduce a
48-dimensional vector to coordinates. High-dimensional cosine is used only
before layout for exact topology, clustering, hidden hub forces, and the
post-layout audit. Private hub identifiers, nodes, and edges are rejected if
they appear in the public JSON. Public hub counts are therefore always zero.

Raw240, graph48, and preference48 rows all retain:

- projection-ready record SHA-256;
- teacher response-lineage SHA-256;
- raw source SHA-256;
- projection provenance SHA-256;
- contract and vector SHA-256 where applicable.

No ORM, migration, database import, or production write is part of this
pipeline.
