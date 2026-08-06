from __future__ import annotations

import csv
import json
import math
from decimal import Decimal
from pathlib import Path

import pytest

from app.sensory_embedding.registry import SENSORY_V2_REGISTRY
from app.sensory_embedding.vertex_batch import (
    ADC_ENV_ALLOWLIST,
    AXIS_REGISTRY_FILE_SHA256,
    AXIS_REGISTRY_PATH,
    BATCH_INPUT_USD_PER_MILLION,
    BATCH_OUTPUT_USD_PER_MILLION,
    CORPUS_ROWS,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT,
    GCS_LIFECYCLE_DAYS,
    HARD_CREATION_BLOCK_USD,
    HISTORICAL_RESERVE_USD,
    INPUT_TOKENS_PER_REQUEST,
    MODEL,
    OUTPUT_TOKENS_PER_REQUEST,
    PROMPT_AXES,
    REQUEST_CONFIG,
    REQUEST_COUNT,
    SHARD_COUNT,
    SHARD_SIZE,
    SOFT_STOP_USD,
    FrozenCocktail,
    GcsLifecycleContract,
    JobLedgerEntry,
    RunCostLedger,
    VertexSensoryBatchError,
    adc_environment,
    atomic_create,
    build_manifest,
    build_requests,
    estimate_cost,
    frozen_csv_bytes,
    gcs_run_metadata,
    guard_job_creation,
    guard_production_job_creation,
    load_prompt_axis_registry,
    load_source_csv,
    minimal_recipe_facts,
    parse_response_line,
    project_ready_records,
    prompt_envelope_diagnostics,
    sha256_bytes,
    update_job_state,
    validate_pilot_token_counts,
)


def _raw_recipe(name: str = "Gin") -> dict[str, object]:
    return {
        "ingredients": [
            {
                "ingredient_id": 99,
                "ingredient_order": 1,
                "canonical_name": name,
                "display_name_ko": "표시 이름",
                "category": "spirit",
                "normalized_amount_ratio": 0.75,
                "normalized_amount_ml": 45,
                "normalization_status": "volume_normalized",
                "ratio_status": "included_in_normalized_volume_denominator",
            },
            {
                "ingredient_id": 100,
                "canonical_name": "Lemon Peel",
                "display_name_ko": "레몬 필",
                "category": "garnish",
                "normalized_amount_ratio": None,
                "normalized_amount_ml": None,
                "ratio_status": "presence_only_excluded",
            },
        ],
        "method": "stir",
        "mixing_ice": "cubed",
        "serving_ice": "none",
        "carbonated": False,
        "garnish": "lemon peel",
        "estimated_pre_dilution_abv_on_normalized_volume": 31.5,
        "abv_estimate_status": "complete_pre_dilution",
        "dilution_status": "unknown_not_applied",
        "normalized_volume_ml": 60,
    }


def _row(cocktail_id: int = 1) -> FrozenCocktail:
    return FrozenCocktail(
        cocktail_id=cocktail_id,
        recipe_facts=minimal_recipe_facts(_raw_recipe()),
        source_column="normalized_recipe_json",
    )


def _recorded_response(
    *,
    selected: str = "C",
    include_e: bool = True,
) -> bytes:
    # Shape captured from Vertex generateContent batch output; token values are
    # reduced and deterministic so the fixture contains no remote/user data.
    candidates = [
        {"token": "A", "logProbability": -3.0},
        {"token": " A", "logProbability": -4.0},
        {"token": "B", "logProbability": -2.0},
        {"token": " C ", "logProbability": -0.2},
        {"token": "D", "logProbability": -1.5},
    ]
    if include_e:
        candidates.append({"token": "E", "logProbability": -3.5})
    value = {
        "status": "",
        "response": {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": f" {selected}\n"}],
                    },
                    "logprobsResult": {
                        "topCandidates": [{"candidates": candidates}],
                    },
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 412,
                "candidatesTokenCount": 1,
            },
        },
    }
    return json.dumps(value, separators=(",", ":")).encode()


def test_registry_csv_is_ordered_hash_pinned_and_local_experiment_only() -> None:
    rows = load_prompt_axis_registry()

    assert len(rows) == len(PROMPT_AXES) == 48
    assert [row.axis_order for row in rows] == list(range(48))
    assert [row.axis_id for row in rows] == [
        axis.axis_id for axis in SENSORY_V2_REGISTRY.axes
    ]
    assert sha256_bytes(AXIS_REGISTRY_PATH.read_bytes()) == AXIS_REGISTRY_FILE_SHA256
    with AXIS_REGISTRY_PATH.open(encoding="utf-8", newline="") as source:
        raw = list(csv.DictReader(source))
    assert {row["registry_version"] for row in raw} == {"sensory-48-ae-v2"}
    assert {row["status"] for row in raw} == {"APPROVED_LOCAL_EXPERIMENT"}


def test_source_loader_minimizes_recipe_and_never_serializes_names_or_ids(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    recipe = json.dumps(_raw_recipe(), ensure_ascii=False)
    source.write_text(
        "cocktail_id,cocktail_name_ko,normalized_recipe_json,ignored\n"
        f'7,비밀 이름,"{recipe.replace(chr(34), chr(34) * 2)}",x\n',
        encoding="utf-8",
    )

    rows = load_source_csv(source, expected_rows=1)

    assert rows[0].source_column == "normalized_recipe_json"
    encoded = frozen_csv_bytes(rows).decode()
    assert "비밀 이름" not in encoded
    assert "display_name_ko" not in encoded
    assert "ingredient_id" not in encoded
    assert "ingredient_order" not in encoded
    assert "ratio_status" not in encoded
    assert rows[0].recipe_facts["ingredients"] == [
        {
            "canonical_name": "Gin",
            "category": "spirit",
            "normalized_amount_ratio": 0.75,
            "presence_only": False,
        },
        {
            "canonical_name": "Lemon Peel",
            "category": "garnish",
            "presence_only": True,
        },
    ]


def test_source_alias_is_allowed_but_disagreement_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    first = json.dumps(_raw_recipe("Gin"), separators=(",", ":"))
    second = json.dumps(_raw_recipe("Rum"), separators=(",", ":"))
    with source.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "cocktail_id",
                "normalized_recipe_json",
                "recipe_facts",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "cocktail_id": 1,
                "normalized_recipe_json": first,
                "recipe_facts": second,
            }
        )

    with pytest.raises(VertexSensoryBatchError, match="disagree"):
        load_source_csv(source, expected_rows=1)


def test_exact_vertex_enum_request_config_and_name_free_prompt() -> None:
    requests = build_requests((_row(777),))
    sweetness = requests[0][0]
    pungency = requests[5][0]

    assert REQUEST_CONFIG == {
        "responseMimeType": "text/x.enum",
        "responseSchema": {"type": "STRING", "enum": ["A", "B", "C", "D", "E"]},
        "responseLogprobs": True,
        "logprobs": 20,
        "temperature": 1.0,
        "topP": 1.0,
        "maxOutputTokens": 32,
        "thinkingConfig": {"thinkingBudget": 0},
    }
    assert "777" not in sweetness.prompt
    assert "비밀 이름" not in sweetness.prompt
    assert "E=매우 강함" in sweetness.prompt
    assert "E=매우 강함" in pungency.prompt
    assert "과도" not in sweetness.prompt
    vertex = sweetness.vertex_record()
    assert set(vertex) == {"request"}
    assert vertex["request"]["generationConfig"] == REQUEST_CONFIG


def test_full_602x48_axis_modulo_sharding_and_manifest_contract() -> None:
    rows = tuple(_row(index) for index in range(1, CORPUS_ROWS + 1))
    shards = build_requests(rows)
    input_bytes = frozen_csv_bytes(rows)
    manifest = build_manifest(
        rows,
        shards,
        input_sha256=sha256_bytes(input_bytes),
        run_id="sensory-20260806-a",
        created_at="2026-08-06T00:00:00+00:00",
        sdk_version="recorded-test",
    )

    assert len(shards) == SHARD_COUNT == 8
    assert {len(shard) for shard in shards} == {SHARD_SIZE}
    assert sum(map(len, shards)) == REQUEST_COUNT == 28_896
    assert all(
        request.shard_index == request.axis_order % 8
        for shard in shards
        for request in shard
    )
    assert manifest["model"] == MODEL == "gemini-2.5-flash"
    assert manifest["project"] == DEFAULT_PROJECT
    assert manifest["location"] == DEFAULT_LOCATION
    assert manifest["row_count"] == 602
    assert manifest["request_count"] == 28_896
    assert manifest["shard_size"] == 3_612
    assert manifest["pilot_token_envelope"]["status"].startswith("pilot_required")
    assert manifest["gcs_lifecycle"] == GcsLifecycleContract().to_dict()
    with pytest.raises(VertexSensoryBatchError, match="manifest input_id_set"):
        guard_production_job_creation(manifest, RunCostLedger())


def test_prompt_utf8_diagnostics_and_measured_pilot_envelope() -> None:
    shards = build_requests((_row(),))
    diagnostics = prompt_envelope_diagnostics(shards)
    pilot = [shard[0] for shard in shards]

    assert diagnostics.request_count == 48
    assert diagnostics.max_utf8_bytes >= diagnostics.min_utf8_bytes > 0
    passed = validate_pilot_token_counts(
        {request.key: INPUT_TOKENS_PER_REQUEST for request in pilot},
        pilot,
    )
    assert passed["status"] == "pilot_passed"
    with pytest.raises(VertexSensoryBatchError, match="exceeded"):
        validate_pilot_token_counts(
            {request.key: INPUT_TOKENS_PER_REQUEST + 1 for request in pilot},
            pilot,
        )


def test_recorded_logprobs_whitespace_variants_use_logsumexp_and_hashes() -> None:
    request = build_requests((_row(),))[0][0].manifest_record()
    raw = _recorded_response()

    parsed = parse_response_line(raw, request)

    expected_a_logit = math.log(math.exp(-3.0) + math.exp(-4.0))
    logits = (expected_a_logit, -2.0, -0.2, -1.5, -3.5)
    denominator = sum(math.exp(value) for value in logits)
    assert parsed.selected_label == "C"
    assert parsed.probabilities == pytest.approx(
        tuple(math.exp(value) / denominator for value in logits)
    )
    assert math.fsum(parsed.probabilities) == pytest.approx(1.0)
    assert parsed.raw_response_sha256 == sha256_bytes(raw)
    assert len(parsed.response_sha256) == 64


def test_incomplete_a_to_e_logprobs_are_rejected_for_quarantine() -> None:
    request = build_requests((_row(),))[0][0].manifest_record()

    with pytest.raises(VertexSensoryBatchError, match="missing E"):
        parse_response_line(_recorded_response(include_e=False), request)


def test_projection_ready_requires_all_48_axes_and_preserves_response_hashes() -> None:
    request_shards = build_requests((_row(),))
    distributions = tuple(
        parse_response_line(_recorded_response(), request.manifest_record())
        for shard in request_shards
        for request in shard
    )

    projected = project_ready_records(distributions, expected_cocktails=1)

    assert len(projected) == 1
    assert len(projected[0]["axes"]) == 48
    assert len(projected[0]["raw_probabilities"]) == 240
    assert len(projected[0]["source_sha256"]) == 64
    assert (
        projected[0]["axes"][0]["response_sha256"] == distributions[0].response_sha256
    )
    with pytest.raises(VertexSensoryBatchError, match="exactly 48"):
        project_ready_records(distributions[:-1], expected_cocktails=1)


def test_cost_estimate_and_soft_hard_one_job_guards() -> None:
    estimate = estimate_cost(REQUEST_COUNT)

    assert BATCH_INPUT_USD_PER_MILLION == Decimal("0.15")
    assert BATCH_OUTPUT_USD_PER_MILLION == Decimal("1.25")
    assert estimate.input_tokens == REQUEST_COUNT * 845
    assert estimate.output_tokens == REQUEST_COUNT * OUTPUT_TOKENS_PER_REQUEST
    assert estimate.estimated_cost_usd == Decimal("4.818408")
    assert HISTORICAL_RESERVE_USD == Decimal("0.50")
    assert SOFT_STOP_USD == Decimal("7.50")
    assert HARD_CREATION_BLOCK_USD == Decimal("10.00")
    assert guard_job_creation(RunCostLedger(), estimate) == Decimal("5.318408")

    with pytest.raises(VertexSensoryBatchError, match="soft stop"):
        guard_job_creation(
            RunCostLedger(historical_cost_usd="2.20"),
            estimate,
        )
    with pytest.raises(VertexSensoryBatchError, match="hard creation block"):
        guard_job_creation(
            RunCostLedger(historical_cost_usd="4.70"),
            estimate,
            allow_soft_stop_override=True,
        )
    active = RunCostLedger(
        jobs=(
            JobLedgerEntry(
                run_id="run",
                job_name="projects/p/locations/l/batchPredictionJobs/j",
                state="JOB_STATE_RUNNING",
                estimated_cost_usd="4.98",
                created_at="now",
                updated_at="now",
            ),
        )
    )
    with pytest.raises(VertexSensoryBatchError, match="RUNNING"):
        guard_job_creation(active, estimate)
    unknown = update_job_state(
        active,
        job_name=active.jobs[0].job_name,
        state=None,
        updated_at="later",
    )
    assert unknown.jobs[0].state == "UNKNOWN"
    with pytest.raises(VertexSensoryBatchError, match="UNKNOWN"):
        guard_job_creation(unknown, estimate)


def test_adc_allowlist_ignores_api_keys_and_credential_file_paths() -> None:
    environment = adc_environment(
        {
            "GOOGLE_CLOUD_PROJECT": "project",
            "GOOGLE_CLOUD_QUOTA_PROJECT": "quota",
            "GOOGLE_APPLICATION_CREDENTIALS": "/do/not/read.json",
            "GEMINI_API_KEY": "secret",
            "GOOGLE_API_KEY": "secret",
        }
    )

    assert set(environment) == ADC_ENV_ALLOWLIST
    assert "API_KEY" not in repr(environment)
    assert "CREDENTIALS" not in repr(environment)


def test_gcs_contract_and_atomic_outputs_are_create_only(tmp_path: Path) -> None:
    assert GCS_LIFECYCLE_DAYS == 1
    metadata = gcs_run_metadata(
        run_id="run-1",
        manifest_sha256="a" * 64,
        shard_index=7,
        object_sha256="b" * 64,
    )
    assert metadata["run-id"] == "run-1"
    assert metadata["shard-index"] == "7"

    target = tmp_path / "artifact.json"
    atomic_create(target, b"first")
    with pytest.raises(VertexSensoryBatchError, match="refusing to replace"):
        atomic_create(target, b"second")
    assert target.read_bytes() == b"first"
