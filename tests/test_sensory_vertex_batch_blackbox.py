from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

from app.sensory_embedding.registry import SENSORY_V2_REGISTRY
from app.sensory_embedding.vertex_batch import (
    AXIS_REGISTRY_FILE_SHA256,
    AXIS_REGISTRY_PATH,
    BATCH_INPUT_USD_PER_MILLION,
    BATCH_OUTPUT_USD_PER_MILLION,
    CORPUS_ROWS,
    HARD_CREATION_BLOCK_USD,
    HISTORICAL_RESERVE_USD,
    INPUT_TOKENS_PER_REQUEST,
    OUTPUT_TOKENS_PER_REQUEST,
    REQUEST_CONFIG,
    REQUEST_COUNT,
    SHARD_COUNT,
    SHARD_SIZE,
    SOFT_STOP_USD,
    FrozenCocktail,
    ParsedDistribution,
    RunCostLedger,
    SensoryBatchRequest,
    VertexSensoryBatchError,
    atomic_create,
    build_manifest,
    build_requests,
    estimate_cost,
    frozen_csv_bytes,
    guard_job_creation,
    guard_production_job_creation,
    json_bytes,
    jsonl_bytes,
    load_cohort_ids_csv,
    load_prompt_axis_registry,
    load_source_csv,
    parse_recorded_outputs,
    parse_response_line,
    project_ready_records,
    sha256_bytes,
    validate_pilot_token_counts,
)
from scripts.sensory_vertex_batch import main as batch_cli

_MONOTONIC_RUBRIC = "A=연상되거나 감지되지 않음, B=약함, C=보통, D=강함, E=매우 강함."
_ACTUAL_SOURCE_CANDIDATES = (
    Path(__file__).resolve().parents[3]
    / "cocktail-mate-server"
    / "sensory-graph-lab"
    / "data"
    / "generated"
    / "latest"
    / "normalized_cocktails.csv",
    Path(__file__).resolve().parents[1]
    / "sensory-graph-lab"
    / "data"
    / "generated"
    / "latest"
    / "normalized_cocktails.csv",
)
_LEGACY_COHORT_CANDIDATES = (
    Path(__file__).resolve().parents[3]
    / "cocktail-mate-server"
    / "taste-data"
    / "cocktail-taste-descriptions.csv",
    Path(__file__).resolve().parents[1]
    / "taste-data"
    / "cocktail-taste-descriptions.csv",
)
ACTUAL_SOURCE = next(
    (candidate for candidate in _ACTUAL_SOURCE_CANDIDATES if candidate.is_file()),
    _ACTUAL_SOURCE_CANDIDATES[0],
)
LEGACY_COHORT_SOURCE = next(
    (candidate for candidate in _LEGACY_COHORT_CANDIDATES if candidate.is_file()),
    _LEGACY_COHORT_CANDIDATES[0],
)
ACTUAL_SOURCE_SHA256 = (
    "ee68629bbe2122509729bf748a78b6c1b6e7ce1fda0a8df5bb34d76dcf8ad68e"
)
LEGACY_COHORT_SOURCE_SHA256 = (
    "8755a91cfd2709b87fad3a05e5daef158d7ea589cb08e6c3f09ab4ecabd4ab6f"
)
CURRENT_COHORT_ID_SET_SHA256 = (
    "56e77646b60ad9b45cbdcd43f4807dde994ef40b1d5e4461dbfa41ca2d59c05f"
)
ACTUAL_FROZEN_SHA256 = (
    "4a51835460938ddebc11507d34da2835796e4f73179b15279ddd94253523560b"
)


def _require_actual_source() -> None:
    missing = [
        path for path in (ACTUAL_SOURCE, LEGACY_COHORT_SOURCE) if not path.is_file()
    ]
    if missing:
        pytest.skip(f"local cohort audit sources are unavailable: {missing}")


def _legacy_cohort_ids() -> frozenset[int]:
    _require_actual_source()
    with LEGACY_COHORT_SOURCE.open(encoding="utf-8-sig", newline="") as source:
        return frozenset(int(row["cocktail_id"]) for row in csv.DictReader(source))


@pytest.fixture(scope="module")
def actual_corpus() -> tuple[
    tuple[FrozenCocktail, ...],
    tuple[tuple[SensoryBatchRequest, ...], ...],
]:
    _require_actual_source()
    cohort_ids = load_cohort_ids_csv(LEGACY_COHORT_SOURCE)
    rows = load_source_csv(
        ACTUAL_SOURCE,
        included_cocktail_ids=cohort_ids,
    )
    return rows, build_requests(rows)


@pytest.fixture(scope="module")
def production_manifest(
    actual_corpus: tuple[
        tuple[FrozenCocktail, ...],
        tuple[tuple[SensoryBatchRequest, ...], ...],
    ],
) -> dict[str, object]:
    rows, shards = actual_corpus
    pilot_requests = tuple(shard[0] for shard in shards)
    pilot = validate_pilot_token_counts(
        {request.key: 100 for request in pilot_requests},
        pilot_requests,
    )
    return build_manifest(
        rows,
        shards,
        input_sha256=sha256_bytes(ACTUAL_SOURCE.read_bytes()),
        run_id="blackbox-audit-20260806",
        created_at="2026-08-06T00:00:00+00:00",
        sdk_version="blackbox-recorded",
        pilot_token_envelope=pilot,
        cohort_source_sha256=LEGACY_COHORT_SOURCE_SHA256,
        cohort_id_set_sha256=CURRENT_COHORT_ID_SET_SHA256,
    )


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for child in value.values() for key in _all_mapping_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _all_mapping_keys(child)}
    return set()


def _response_line(
    *,
    selected: str = "C",
    missing_label: str | None = None,
    snake_case: bool = False,
) -> bytes:
    candidates: list[dict[str, object]] = [
        {"token": "A", "logProbability": -3.0},
        {"token": " A ", "logProbability": -4.0},
        {"token": "B", "logProbability": -2.0},
        {"token": " C\n", "logProbability": -0.2},
        {"token": "D", "logProbability": -1.5},
        {"token": "E", "logProbability": -3.5},
        {"token": "explanation", "logProbability": -0.1},
    ]
    candidates = [
        candidate
        for candidate in candidates
        if str(candidate["token"]).strip() != missing_label
    ]
    if snake_case:
        candidates = [
            {
                "token": candidate["token"],
                "log_probability": candidate["logProbability"],
            }
            for candidate in candidates
        ]
        response: dict[str, object] = {
            "candidates": [
                {
                    "content": {"parts": [{"text": f" {selected}\n"}]},
                    "logprobs_result": {"top_candidates": [{"candidates": candidates}]},
                }
            ]
        }
        record: dict[str, object] = response
    else:
        response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": f" {selected}\n"}],
                    },
                    "finishReason": "STOP",
                    "logprobsResult": {
                        "topCandidates": [{"candidates": candidates}],
                        "chosenCandidates": [
                            {"token": selected, "logProbability": -0.2}
                        ],
                    },
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 412,
                "candidatesTokenCount": 1,
                "totalTokenCount": 413,
            },
        }
        record = {
            "status": "",
            "processed_time": "2026-08-06T00:00:00Z",
            "request": {"contents": [{"role": "user", "parts": [{"text": "..."}]}]},
            "response": response,
        }
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()


def _partial_manifest(
    row: FrozenCocktail,
) -> tuple[
    dict[str, object],
    tuple[tuple[SensoryBatchRequest, ...], ...],
]:
    shards = build_requests((row,))
    manifest = build_manifest(
        (row,),
        shards,
        input_sha256=sha256_bytes(frozen_csv_bytes((row,))),
        run_id="blackbox-partial",
        created_at="2026-08-06T00:00:00+00:00",
        sdk_version="blackbox-recorded",
    )
    return manifest, shards


def test_actual_622_row_normalized_source_is_filtered_by_the_frozen_602_id_cohort(
    actual_corpus: tuple[
        tuple[FrozenCocktail, ...],
        tuple[tuple[SensoryBatchRequest, ...], ...],
    ],
) -> None:
    cohort_rows, _ = actual_corpus
    with ACTUAL_SOURCE.open(encoding="utf-8-sig", newline="") as source:
        normalized_rows = list(csv.DictReader(source))
    with LEGACY_COHORT_SOURCE.open(
        encoding="utf-8-sig",
        newline="",
    ) as source:
        legacy_rows = list(csv.DictReader(source))
    normalized_by_id = {int(row["cocktail_id"]): row for row in normalized_rows}
    normalized_ids = set(normalized_by_id)
    cohort_ids = _legacy_cohort_ids()
    extra_ids = normalized_ids - cohort_ids

    assert sha256_bytes(ACTUAL_SOURCE.read_bytes()) == ACTUAL_SOURCE_SHA256
    assert (
        sha256_bytes(LEGACY_COHORT_SOURCE.read_bytes()) == LEGACY_COHORT_SOURCE_SHA256
    )
    assert len(normalized_rows) == 622
    assert len(legacy_rows) == len(cohort_ids) == 602
    assert cohort_ids < normalized_ids
    assert len(extra_ids) == 20
    assert not (cohort_ids - normalized_ids)
    assert {normalized_by_id[cocktail_id]["base_tag"] for cocktail_id in extra_ids} == {
        "non_alcoholic"
    }
    assert {row.cocktail_id for row in cohort_rows} == cohort_ids
    assert len(cohort_rows) == CORPUS_ROWS == 602
    assert all(raw["normalized_recipe_json"].strip() for raw in normalized_rows)
    assert frozen_csv_bytes(cohort_rows).startswith(
        b"cocktail_id,recipe_facts,recipe_source_column\n"
    )
    assert sha256_bytes(frozen_csv_bytes(cohort_rows)) == ACTUAL_FROZEN_SHA256
    with pytest.raises(VertexSensoryBatchError, match="expected 602.*got 622"):
        load_source_csv(ACTUAL_SOURCE)


def test_actual_source_is_reduced_to_the_minimal_recipe_fact_schema(
    actual_corpus: tuple[
        tuple[FrozenCocktail, ...],
        tuple[tuple[SensoryBatchRequest, ...], ...],
    ],
) -> None:
    rows, _ = actual_corpus
    allowed_root = {
        "ingredients",
        "method",
        "mixing_ice",
        "serving_ice",
        "garnish",
        "estimated_pre_dilution_abv_status",
        "carbonation",
        "estimated_pre_dilution_abv",
    }
    allowed_ingredient = {
        "canonical_name",
        "category",
        "normalized_amount_ratio",
        "normalized_amount_ml",
        "presence_only",
    }
    forbidden = {
        "cocktail_id",
        "cocktail_name",
        "cocktail_name_en",
        "ingredient_id",
        "ingredient_order",
        "display_name_ko",
        "amount",
        "unit",
        "normalization_status",
        "ratio_status",
        "normalized_volume_ml",
        "dilution_status",
    }

    for row in rows:
        assert isinstance(row.recipe_facts, dict)
        assert set(row.recipe_facts) <= allowed_root
        assert forbidden.isdisjoint(_all_mapping_keys(row.recipe_facts))
        ingredients = row.recipe_facts["ingredients"]
        assert isinstance(ingredients, list) and ingredients
        assert all(
            isinstance(ingredient, dict)
            and set(ingredient) <= allowed_ingredient
            and type(ingredient.get("presence_only")) is bool
            for ingredient in ingredients
        )


def test_target_does_not_derive_cohort_from_base_tag_or_the_legacy_file_at_runtime() -> (
    None
):
    module_source = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "sensory_embedding"
        / "vertex_batch.py"
    ).read_text(encoding="utf-8")
    script_source = (
        Path(__file__).resolve().parents[1] / "scripts" / "sensory_vertex_batch.py"
    ).read_text(encoding="utf-8")

    assert "base_tag" not in module_source
    assert "base_tag" not in script_source
    assert "cocktail-taste-descriptions.csv" not in module_source
    assert "cocktail-taste-descriptions.csv" not in script_source
    assert "taste-data" not in module_source
    assert "taste-data" not in script_source


def test_cli_explicitly_loads_and_applies_the_cohort_allowlist() -> None:
    script_source = (
        Path(__file__).resolve().parents[1] / "scripts" / "sensory_vertex_batch.py"
    ).read_text(encoding="utf-8")

    assert "load_cohort_ids_csv" in script_source
    assert "included_cocktail_ids" in script_source


def test_current_requests_are_exactly_28896_in_eight_equal_axis_modulo_shards(
    actual_corpus: tuple[
        tuple[FrozenCocktail, ...],
        tuple[tuple[SensoryBatchRequest, ...], ...],
    ],
) -> None:
    _, shards = actual_corpus

    assert len(shards) == SHARD_COUNT == 8
    assert [len(shard) for shard in shards] == [SHARD_SIZE] * 8 == [3612] * 8
    assert sum(len(shard) for shard in shards) == REQUEST_COUNT == 28_896
    for shard_index, shard in enumerate(shards):
        assert {request.axis_order for request in shard} == set(
            range(shard_index, 48, 8)
        )
        assert all(request.shard_index == request.axis_order % 8 for request in shard)


def test_every_actual_vertex_record_is_identity_free_and_uses_one_recipe_only(
    actual_corpus: tuple[
        tuple[FrozenCocktail, ...],
        tuple[tuple[SensoryBatchRequest, ...], ...],
    ],
) -> None:
    rows, shards = actual_corpus
    forbidden_recipe_keys = {
        "cocktail_id",
        "cocktail_name",
        "cocktail_name_en",
        "ingredient_id",
        "ingredient_order",
        "display_name_ko",
    }

    for shard in shards:
        for request in shard:
            record = request.vertex_record()
            assert set(record) == {"request"}
            vertex_request = record["request"]
            assert isinstance(vertex_request, dict)
            assert set(vertex_request) == {"contents", "generationConfig"}
            assert forbidden_recipe_keys.isdisjoint(_all_mapping_keys(record))
            encoded_recipe = request.prompt.split("레시피 사실: ", 1)[1].split(
                "\n평가 축:", 1
            )[0]
            assert json.loads(encoded_recipe) == rows[request.row_index].recipe_facts
            assert forbidden_recipe_keys.isdisjoint(
                _all_mapping_keys(json.loads(encoded_recipe))
            )


def test_every_axis_uses_the_same_monotonic_rubric_and_exact_vertex_config(
    actual_corpus: tuple[
        tuple[FrozenCocktail, ...],
        tuple[tuple[SensoryBatchRequest, ...], ...],
    ],
) -> None:
    _, shards = actual_corpus
    first_cocktail_requests = sorted(
        (request for shard in shards for request in shard if request.row_index == 0),
        key=lambda request: request.axis_order,
    )

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
    assert len(first_cocktail_requests) == 48
    assert [request.axis_id for request in first_cocktail_requests] == [
        axis.axis_id for axis in SENSORY_V2_REGISTRY.axes
    ]
    for request in first_cocktail_requests:
        vertex_request = request.vertex_record()["request"]
        assert isinstance(vertex_request, dict)
        assert request.prompt.count(_MONOTONIC_RUBRIC) == 1
        assert vertex_request["generationConfig"] == REQUEST_CONFIG


def test_registry_is_hash_pinned_and_a_modified_copy_is_rejected(
    tmp_path: Path,
) -> None:
    axes = load_prompt_axis_registry()
    registry_bytes = AXIS_REGISTRY_PATH.read_bytes()
    tampered_path = tmp_path / "sensory-axis-registry.csv"
    tampered_path.write_bytes(registry_bytes.replace(b"Sweetness", b"SweetnesS", 1))

    assert len(axes) == 48
    assert hashlib.sha256(registry_bytes).hexdigest() == AXIS_REGISTRY_FILE_SHA256
    with pytest.raises(VertexSensoryBatchError, match="SHA-256 mismatch"):
        load_prompt_axis_registry(tampered_path)


def test_request_and_manifest_hashes_are_deterministic_and_content_bound(
    actual_corpus: tuple[
        tuple[FrozenCocktail, ...],
        tuple[tuple[SensoryBatchRequest, ...], ...],
    ],
    production_manifest: dict[str, object],
) -> None:
    rows, shards = actual_corpus
    pilot = production_manifest["pilot_token_envelope"]
    assert isinstance(pilot, Mapping)
    rebuilt = build_manifest(
        rows,
        shards,
        input_sha256=sha256_bytes(ACTUAL_SOURCE.read_bytes()),
        run_id="blackbox-audit-20260806",
        created_at="2026-08-06T00:00:00+00:00",
        sdk_version="blackbox-recorded",
        pilot_token_envelope=pilot,
        cohort_source_sha256=LEGACY_COHORT_SOURCE_SHA256,
        cohort_id_set_sha256=CURRENT_COHORT_ID_SET_SHA256,
    )
    manifest_shards = production_manifest["shards"]
    manifest_requests = production_manifest["requests"]

    assert isinstance(manifest_shards, list)
    assert isinstance(manifest_requests, list)
    assert json_bytes(rebuilt) == json_bytes(production_manifest)
    for shard, manifest_shard in zip(shards, manifest_shards, strict=True):
        assert isinstance(manifest_shard, dict)
        assert manifest_shard["sha256"] == sha256_bytes(jsonl_bytes(shard))
    flattened = [request for shard in shards for request in shard]
    for request, manifest_request in zip(flattened, manifest_requests, strict=True):
        assert isinstance(manifest_request, dict)
        assert manifest_request["prompt_sha256"] == sha256_bytes(
            request.prompt.encode("utf-8")
        )


def test_manifest_binds_the_exact_frozen_current_cohort_id_set(
    production_manifest: dict[str, object],
) -> None:
    assert production_manifest["input_id_set_sha256"] == CURRENT_COHORT_ID_SET_SHA256
    assert production_manifest["id_set_sha256"] == CURRENT_COHORT_ID_SET_SHA256
    assert production_manifest["input_row_count"] == 602
    assert production_manifest["request_count"] == 28_896
    assert production_manifest["shard_size"] == 3_612


def test_manifest_binds_allowlist_source_and_membership_provenance(
    production_manifest: dict[str, object],
) -> None:
    cohort_hashes = {
        key: value
        for key, value in production_manifest.items()
        if "cohort" in key and "sha256" in key
    }

    assert CURRENT_COHORT_ID_SET_SHA256 in cohort_hashes.values()
    assert LEGACY_COHORT_SOURCE_SHA256 in cohort_hashes.values()


def test_cost_math_and_exact_soft_and_hard_gate_boundaries() -> None:
    estimate = estimate_cost(28_896)
    soft_history = SOFT_STOP_USD - HISTORICAL_RESERVE_USD - estimate.estimated_cost_usd
    hard_history = (
        HARD_CREATION_BLOCK_USD - HISTORICAL_RESERVE_USD - estimate.estimated_cost_usd
    )

    assert OUTPUT_TOKENS_PER_REQUEST == 32
    assert estimate.request_count == 28_896 == 602 * 48
    assert estimate.input_tokens == 24_417_120 == estimate.request_count * 845
    assert estimate.output_tokens == 924_672 == estimate.request_count * 32
    assert BATCH_INPUT_USD_PER_MILLION == Decimal("0.15")
    assert BATCH_OUTPUT_USD_PER_MILLION == Decimal("1.25")
    assert estimate.input_cost_usd == Decimal("3.662568")
    assert estimate.output_cost_usd == Decimal("1.155840")
    assert estimate.estimated_cost_usd == Decimal("4.818408")

    assert guard_job_creation(
        RunCostLedger(historical_cost_usd=str(soft_history - Decimal("0.000001"))),
        estimate,
    ) == SOFT_STOP_USD - Decimal("0.000001")
    with pytest.raises(VertexSensoryBatchError, match="soft stop"):
        guard_job_creation(
            RunCostLedger(historical_cost_usd=str(soft_history)),
            estimate,
        )
    assert (
        guard_job_creation(
            RunCostLedger(historical_cost_usd=str(soft_history)),
            estimate,
            allow_soft_stop_override=True,
        )
        == SOFT_STOP_USD
    )
    with pytest.raises(VertexSensoryBatchError, match="hard creation block"):
        guard_job_creation(
            RunCostLedger(historical_cost_usd=str(hard_history)),
            estimate,
            allow_soft_stop_override=True,
        )


def test_pilot_must_cover_exact_keys_and_reject_any_count_above_845(
    actual_corpus: tuple[
        tuple[FrozenCocktail, ...],
        tuple[tuple[SensoryBatchRequest, ...], ...],
    ],
) -> None:
    _, shards = actual_corpus
    pilot_requests = tuple(shard[0] for shard in shards)
    counts = {request.key: INPUT_TOKENS_PER_REQUEST for request in pilot_requests}

    evidence = validate_pilot_token_counts(counts, pilot_requests)

    assert evidence["status"] == "pilot_passed"
    assert evidence["measured_request_count"] == 8
    assert evidence["measured_max_tokens"] == 845
    assert len(str(evidence["token_counts_sha256"])) == 64
    with pytest.raises(VertexSensoryBatchError, match="exactly match"):
        validate_pilot_token_counts(
            {
                key: value
                for key, value in counts.items()
                if key != pilot_requests[0].key
            },
            pilot_requests,
        )
    over = dict(counts)
    over[pilot_requests[0].key] = INPUT_TOKENS_PER_REQUEST + 1
    with pytest.raises(VertexSensoryBatchError, match="exceeded"):
        validate_pilot_token_counts(over, pilot_requests)


def test_production_guard_rejects_manifest_request_identity_tampering(
    production_manifest: dict[str, object],
) -> None:
    baseline = guard_production_job_creation(
        production_manifest,
        RunCostLedger(),
    )
    assert baseline == (
        HISTORICAL_RESERVE_USD + estimate_cost(REQUEST_COUNT).estimated_cost_usd
    )
    tampered = deepcopy(production_manifest)
    requests = tampered["requests"]
    assert isinstance(requests, list) and isinstance(requests[0], dict)
    requests[0]["cocktail_id"] = 999_999

    with pytest.raises(VertexSensoryBatchError, match="manifest|identity|cocktail"):
        guard_production_job_creation(tampered, RunCostLedger())


def test_production_guard_rejects_row_to_existing_cocktail_remapping(
    production_manifest: dict[str, object],
) -> None:
    tampered = deepcopy(production_manifest)
    requests = tampered["requests"]
    assert isinstance(requests, list) and isinstance(requests[0], dict)
    original_id = requests[0]["cocktail_id"]
    replacement_id = next(
        request["cocktail_id"]
        for request in requests
        if isinstance(request, dict) and request["cocktail_id"] != original_id
    )
    requests[0]["cocktail_id"] = replacement_id

    with pytest.raises(VertexSensoryBatchError, match="identity|cohort|row"):
        guard_production_job_creation(tampered, RunCostLedger())


def test_production_guard_rejects_tampered_shard_identity(
    production_manifest: dict[str, object],
) -> None:
    tampered = deepcopy(production_manifest)
    shards = tampered["shards"]
    assert isinstance(shards, list) and isinstance(shards[0], dict)
    shards[0]["shard_index"] = 7
    shards[0]["filename"] = "requests-07.jsonl"

    with pytest.raises(VertexSensoryBatchError, match="manifest|shard"):
        guard_production_job_creation(tampered, RunCostLedger())


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("schema_version", 999),
        ("prompt_axis_registry_file_sha256", "0" * 64),
    ],
)
def test_production_guard_rejects_static_manifest_contract_tampering(
    production_manifest: dict[str, object],
    field: str,
    tampered_value: object,
) -> None:
    tampered = deepcopy(production_manifest)
    tampered[field] = tampered_value

    with pytest.raises(VertexSensoryBatchError, match="manifest"):
        guard_production_job_creation(tampered, RunCostLedger())


def test_production_guard_rejects_tampered_negative_pilot_evidence(
    production_manifest: dict[str, object],
) -> None:
    tampered = deepcopy(production_manifest)
    pilot = tampered["pilot_token_envelope"]
    assert isinstance(pilot, dict)
    pilot["measured_max_tokens"] = -1
    pilot["measured_request_count"] = 0
    pilot.pop("token_counts_sha256")

    with pytest.raises(VertexSensoryBatchError, match="pilot"):
        guard_production_job_creation(tampered, RunCostLedger())


@pytest.mark.parametrize("snake_case", [False, True])
def test_realistic_vertex_batch_response_shapes_parse_complete_a_to_e(
    actual_corpus: tuple[
        tuple[FrozenCocktail, ...],
        tuple[tuple[SensoryBatchRequest, ...], ...],
    ],
    snake_case: bool,
) -> None:
    _, shards = actual_corpus
    request = shards[0][0].manifest_record()
    raw = _response_line(snake_case=snake_case)

    parsed = parse_response_line(raw, request)

    expected_a = math.log(math.exp(-3.0) + math.exp(-4.0))
    logits = (expected_a, -2.0, -0.2, -1.5, -3.5)
    normalizer = sum(math.exp(value) for value in logits)
    assert parsed.selected_label == "C"
    assert parsed.probabilities == pytest.approx(
        tuple(math.exp(value) / normalizer for value in logits)
    )
    assert math.fsum(parsed.probabilities) == pytest.approx(1.0)
    assert parsed.raw_response_sha256 == sha256_bytes(raw)


def test_canonical_response_hash_ignores_json_format_but_raw_hash_does_not(
    actual_corpus: tuple[
        tuple[FrozenCocktail, ...],
        tuple[tuple[SensoryBatchRequest, ...], ...],
    ],
) -> None:
    _, shards = actual_corpus
    request = shards[0][0].manifest_record()
    first = _response_line()
    decoded = json.loads(first)
    second = json.dumps(
        decoded,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode()

    first_parsed = parse_response_line(first, request)
    second_parsed = parse_response_line(second, request)

    assert first_parsed.response_sha256 == second_parsed.response_sha256
    assert first_parsed.raw_response_sha256 != second_parsed.raw_response_sha256


def test_incomplete_logprobs_are_quarantined_without_imputation(
    actual_corpus: tuple[
        tuple[FrozenCocktail, ...],
        tuple[tuple[SensoryBatchRequest, ...], ...],
    ],
    tmp_path: Path,
) -> None:
    rows, _ = actual_corpus
    manifest, shards = _partial_manifest(rows[0])
    output_paths: list[Path] = []
    for shard_index, shard in enumerate(shards):
        lines = [
            _response_line(
                missing_label="E" if shard_index == 3 and index == 2 else None
            )
            for index, _ in enumerate(shard)
        ]
        path = tmp_path / f"responses-{shard_index:02d}.jsonl"
        path.write_bytes(b"\n".join(lines) + b"\n")
        output_paths.append(path)

    parsed, quarantined = parse_recorded_outputs(manifest, output_paths)

    assert len(parsed) == 47
    assert len(quarantined) == 1
    assert quarantined[0].key == shards[3][2].key
    assert "missing E" in quarantined[0].reason
    assert len(quarantined[0].raw_response_sha256) == 64
    with pytest.raises(VertexSensoryBatchError, match="exactly 48"):
        project_ready_records(parsed, expected_cocktails=1)


def test_projection_requires_registry_axis_order_not_only_48_axis_ids() -> None:
    distributions = tuple(
        ParsedDistribution(
            key=f"bad-order-{axis.axis_order}",
            cocktail_id=1,
            axis_order=0,
            axis_id=axis.axis_id,
            selected_label="C",
            probabilities=(0.1, 0.2, 0.4, 0.2, 0.1),
            response_sha256="a" * 64,
            raw_response_sha256="b" * 64,
        )
        for axis in SENSORY_V2_REGISTRY.axes
    )

    with pytest.raises(VertexSensoryBatchError, match="ordered axes|axis_order"):
        project_ready_records(distributions, expected_cocktails=1)


def test_atomic_create_and_cli_outputs_are_create_only(
    actual_corpus: tuple[
        tuple[FrozenCocktail, ...],
        tuple[tuple[SensoryBatchRequest, ...], ...],
    ],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows, _ = actual_corpus
    direct = tmp_path / "nested" / "artifact.bin"
    atomic_create(direct, b"first")
    with pytest.raises(VertexSensoryBatchError, match="refusing to replace"):
        atomic_create(direct, b"second")
    assert direct.read_bytes() == b"first"

    source = tmp_path / "one-row.csv"
    source.write_bytes(frozen_csv_bytes((rows[0],)))
    output_dir = tmp_path / "batch"
    arguments = [
        "build",
        "--input",
        str(source),
        "--output-dir",
        str(output_dir),
        "--run-id",
        "blackbox-cli",
        "--created-at",
        "2026-08-06T00:00:00+00:00",
        "--allow-partial",
    ]
    assert batch_cli(arguments) == 0
    capsys.readouterr()
    paths = tuple(
        output_dir / f"requests-{shard_index:02d}.jsonl" for shard_index in range(8)
    ) + (output_dir / "manifest.json",)
    before = {path: sha256_bytes(path.read_bytes()) for path in paths}
    assert all(path.is_file() for path in paths)
    assert all(len(path.read_bytes().splitlines()) == 6 for path in paths[:-1])

    assert batch_cli(arguments) == 1
    capsys.readouterr()
    assert {path: sha256_bytes(path.read_bytes()) for path in paths} == before


def test_full_cli_freeze_requires_the_exact_hash_pinned_cohort(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    frozen = tmp_path / "frozen.csv"
    valid_arguments = [
        "freeze",
        "--input",
        str(ACTUAL_SOURCE),
        "--cohort-ids",
        str(LEGACY_COHORT_SOURCE),
        "--output",
        str(frozen),
    ]

    assert batch_cli(valid_arguments) == 0
    capsys.readouterr()
    assert sha256_bytes(frozen.read_bytes()) == ACTUAL_FROZEN_SHA256
    assert len(frozen.read_bytes().splitlines()) == 603

    missing_allowlist_output = tmp_path / "missing-allowlist.csv"
    assert (
        batch_cli(
            [
                "freeze",
                "--input",
                str(ACTUAL_SOURCE),
                "--output",
                str(missing_allowlist_output),
            ]
        )
        == 1
    )
    capsys.readouterr()
    assert not missing_allowlist_output.exists()

    with LEGACY_COHORT_SOURCE.open(
        encoding="utf-8-sig",
        newline="",
    ) as source:
        cohort_rows = list(csv.DictReader(source))
        fieldnames = list(cohort_rows[0])
    cohort_rows[0]["cocktail_id"] = "71"
    wrong_cohort = tmp_path / "wrong-cohort.csv"
    with wrong_cohort.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cohort_rows)

    wrong_output = tmp_path / "wrong-frozen.csv"
    assert (
        batch_cli(
            [
                "freeze",
                "--input",
                str(ACTUAL_SOURCE),
                "--cohort-ids",
                str(wrong_cohort),
                "--output",
                str(wrong_output),
            ]
        )
        == 1
    )
    capsys.readouterr()
    assert not wrong_output.exists()


def test_target_files_have_no_network_google_sdk_database_or_credential_imports() -> (
    None
):
    targets = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "sensory_embedding"
        / "vertex_batch.py",
        Path(__file__).resolve().parents[1] / "scripts" / "sensory_vertex_batch.py",
    )
    banned_import_roots = {
        "aiohttp",
        "boto3",
        "google",
        "httpx",
        "pymongo",
        "pymysql",
        "psycopg",
        "requests",
        "socket",
        "sqlalchemy",
        "urllib",
        "vertexai",
    }
    banned_source_fragments = {
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "create_engine(",
        "urlopen(",
        "https://",
    }

    for target in targets:
        source = target.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.partition(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.partition(".")[0])
        assert banned_import_roots.isdisjoint(imported_roots)
        assert all(fragment not in source for fragment in banned_source_fragments)
