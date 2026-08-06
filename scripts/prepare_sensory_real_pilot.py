"""Prepare a deterministic real-cocktail Vertex teacher pilot without I/O.

This command is local-only: it reads two hash-pinned 602-row CSV artifacts,
selects ten recipes by deterministic recipe-feature k-center coverage, and
creates request/manifest/ledger files. It never imports a cloud SDK, credentials,
database code, or a network client.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.sensory_embedding.contracts import canonical_sha256
from app.sensory_embedding.registry import SENSORY_V2_REGISTRY
from app.sensory_embedding.vertex_batch import (
    AXIS_REGISTRY_FILE_SHA256,
    COHORT_ID_SET_SHA256,
    COHORT_SOURCE_FILE_SHA256,
    CORPUS_ROWS,
    HARD_CREATION_BLOCK_USD,
    HISTORICAL_RESERVE_USD,
    MODEL,
    PLANNING_INPUT_TOKEN_ENVELOPE,
    PROMPT_SHA256,
    REQUEST_CONFIG_SHA256,
    SOFT_STOP_USD,
    FrozenCocktail,
    GcsLifecycleContract,
    RunCostLedger,
    SensoryBatchRequest,
    VertexSensoryBatchError,
    atomic_create,
    build_requests,
    estimate_cost,
    guard_job_creation,
    id_set_sha256,
    json_bytes,
    jsonl_bytes,
    load_source_csv,
    prompt_envelope_diagnostics,
    sha256_bytes,
)
from app.sensory_embedding.vertex_live import (
    LIVE_PILOT_APPROVAL_MARKER,
    LIVE_PILOT_APPROVAL_MARKER_SHA256,
    LIVE_PILOT_FROZEN_SOURCE_SHA256,
    LIVE_PILOT_MANIFEST_TYPE,
    LIVE_PILOT_SELECTED_IDS,
    LIVE_PILOT_SELECTED_ID_SET_SHA256,
    LIVE_PILOT_SHARD_SIZE,
    LIVE_PILOT_STATUS,
    LOCATION,
    PILOT_RUN_SCOPE,
    PROJECT,
)

PILOT_COCKTAIL_COUNT = 10
PILOT_REQUEST_COUNT = PILOT_COCKTAIL_COUNT * 48
FULL_REQUEST_COUNT = CORPUS_ROWS * 48
SELECTION_POLICY = "recipe_feature_jaccard_k_center_v1"


def _scalar_feature(prefix: str, value: object) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (str, int, float, bool)) and not isinstance(value, complex):
        return f"{prefix}:{json.dumps(value, ensure_ascii=True, sort_keys=True)}"
    return None


def recipe_features(row: FrozenCocktail) -> frozenset[str]:
    """Extract only normalized categorical recipe facts used for selection."""

    payload = row.recipe_facts
    if not isinstance(payload, Mapping):
        raise VertexSensoryBatchError("frozen recipe facts must be an object")
    features: set[str] = set()
    for field in ("method", "mixing_ice", "serving_ice", "carbonation"):
        feature = _scalar_feature(field, payload.get(field))
        if feature is not None:
            features.add(feature)
    ingredients = payload.get("ingredients")
    if not isinstance(ingredients, list) or not ingredients:
        raise VertexSensoryBatchError("frozen recipe must have ingredients")
    features.add(f"ingredient_count:{min(len(ingredients), 8)}")
    for ingredient in ingredients:
        if not isinstance(ingredient, Mapping):
            raise VertexSensoryBatchError("frozen ingredient must be an object")
        for field in ("canonical_name", "category"):
            feature = _scalar_feature(field, ingredient.get(field))
            if feature is not None:
                features.add(f"ingredient_{feature}")
        if ingredient.get("presence_only") is True:
            features.add("ingredient_presence_only:true")
    abv = payload.get("estimated_pre_dilution_abv")
    if isinstance(abv, (int, float)) and not isinstance(abv, bool):
        numeric = float(abv)
        if math.isfinite(numeric):
            features.add(f"abv_bin_5:{int(max(0.0, numeric) // 5.0) * 5}")
    if not features:
        raise VertexSensoryBatchError("recipe produced no selection features")
    return frozenset(features)


def _jaccard_distance(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return 1.0 - len(left & right) / len(union)


def select_representative_rows(
    rows: Sequence[FrozenCocktail],
    *,
    count: int = PILOT_COCKTAIL_COUNT,
) -> tuple[FrozenCocktail, ...]:
    """Select a central seed, then deterministic farthest-first coverage."""

    ordered = tuple(sorted(rows, key=lambda row: row.cocktail_id))
    if len(ordered) < count or count <= 0:
        raise VertexSensoryBatchError("representative count exceeds source rows")
    features = tuple(recipe_features(row) for row in ordered)
    distances = tuple(
        tuple(_jaccard_distance(left, right) for right in features) for left in features
    )
    central_index = min(
        range(len(ordered)),
        key=lambda index: (
            math.fsum(distances[index]),
            ordered[index].cocktail_id,
        ),
    )
    selected = [central_index]
    selected_set = {central_index}
    while len(selected) < count:
        candidate = min(
            (index for index in range(len(ordered)) if index not in selected_set),
            key=lambda index: (
                -min(distances[index][chosen] for chosen in selected),
                ordered[index].cocktail_id,
            ),
        )
        selected.append(candidate)
        selected_set.add(candidate)
    return tuple(
        sorted((ordered[index] for index in selected), key=lambda row: row.cocktail_id)
    )


def _load_names(path: Path) -> tuple[dict[int, tuple[str, str]], bytes]:
    try:
        payload = path.read_bytes()
        if sha256_bytes(payload) != COHORT_SOURCE_FILE_SHA256:
            raise VertexSensoryBatchError("cohort source file SHA-256 mismatch")
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required = {"cocktail_id", "cocktail_name_ko", "cocktail_name_en"}
            if not required <= set(reader.fieldnames or ()):
                raise VertexSensoryBatchError("cohort CSV name columns are missing")
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise VertexSensoryBatchError(f"cannot read cohort CSV {path}") from error
    names: dict[int, tuple[str, str]] = {}
    for raw in raw_rows:
        try:
            cocktail_id = int(raw["cocktail_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise VertexSensoryBatchError("invalid cohort cocktail_id") from error
        if cocktail_id in names or cocktail_id <= 0:
            raise VertexSensoryBatchError("cohort IDs must be positive and unique")
        ko = raw["cocktail_name_ko"].strip()
        en = raw["cocktail_name_en"].strip()
        if not ko and not en:
            raise VertexSensoryBatchError("cohort must contain at least one name")
        names[cocktail_id] = (ko, en)
    if len(names) != CORPUS_ROWS or id_set_sha256(names) != COHORT_ID_SET_SHA256:
        raise VertexSensoryBatchError("cohort names do not match pinned 602 IDs")
    return names, payload


def _flatten_requests(
    rows: Sequence[FrozenCocktail],
) -> tuple[SensoryBatchRequest, ...]:
    shards = build_requests(rows)
    return tuple(
        sorted(
            (request for shard in shards for request in shard),
            key=lambda request: (request.row_index, request.axis_order),
        )
    )


def prepare_real_pilot(
    *,
    frozen_source: Path,
    cohort_source: Path,
    output_dir: Path,
    run_id: str,
    created_at: str,
    user_approval_marker: str,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")
    frozen_payload = frozen_source.read_bytes()
    if sha256_bytes(frozen_payload) != LIVE_PILOT_FROZEN_SOURCE_SHA256:
        raise VertexSensoryBatchError("frozen 602 source SHA-256 mismatch")
    rows = load_source_csv(frozen_source, expected_rows=CORPUS_ROWS)
    if id_set_sha256(row.cocktail_id for row in rows) != COHORT_ID_SET_SHA256:
        raise VertexSensoryBatchError("frozen source does not match pinned 602 IDs")
    names, cohort_payload = _load_names(cohort_source)
    selected = select_representative_rows(rows)
    if tuple(row.cocktail_id for row in selected) != LIVE_PILOT_SELECTED_IDS:
        raise VertexSensoryBatchError(
            "deterministic selection drifted from the reviewed pilot IDs"
        )
    if user_approval_marker != LIVE_PILOT_APPROVAL_MARKER:
        raise VertexSensoryBatchError(
            "pilot preparation requires the exact user approval marker"
        )
    shards = build_requests(selected)
    if any(len(shard) != LIVE_PILOT_SHARD_SIZE for shard in shards):
        raise VertexSensoryBatchError("pilot must have eight 60-record shards")
    requests = _flatten_requests(selected)
    if len(requests) != PILOT_REQUEST_COUNT:
        raise VertexSensoryBatchError("pilot must contain exactly 480 requests")
    request_payload = jsonl_bytes(requests)
    request_sha256 = sha256_bytes(request_payload)
    pilot_estimate = estimate_cost(PILOT_REQUEST_COUNT)
    full_estimate = estimate_cost(FULL_REQUEST_COUNT)
    projected_pilot = guard_job_creation(RunCostLedger(), pilot_estimate)
    projected_pilot_and_full = (
        HISTORICAL_RESERVE_USD
        + pilot_estimate.estimated_cost_usd
        + full_estimate.estimated_cost_usd
    )
    if projected_pilot_and_full >= SOFT_STOP_USD:
        raise VertexSensoryBatchError("pilot plus full plan exceeds soft stop")

    selected_records = [
        {
            "selection_order": order,
            "cocktail_id": row.cocktail_id,
            "cocktail_name_ko": names[row.cocktail_id][0],
            "cocktail_name_en": names[row.cocktail_id][1],
            "recipe_feature_sha256": canonical_sha256(sorted(recipe_features(row))),
        }
        for order, row in enumerate(selected, start=1)
    ]
    request_records = [
        request.manifest_record() for shard in shards for request in shard
    ]
    selected_ids_hash = id_set_sha256(row.cocktail_id for row in selected)
    if selected_ids_hash != LIVE_PILOT_SELECTED_ID_SET_SHA256:
        raise VertexSensoryBatchError("selected pilot ID hash drifted")
    diagnostics = prompt_envelope_diagnostics(build_requests(selected)).to_dict()
    shard_records = [
        {
            "shard_index": shard_index,
            "filename": f"requests-{shard_index:02d}.jsonl",
            "record_count": len(shard),
            "axis_orders": list(range(shard_index, 48, 8)),
            "sha256": sha256_bytes(jsonl_bytes(shard)),
        }
        for shard_index, shard in enumerate(shards)
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_type": LIVE_PILOT_MANIFEST_TYPE,
        "run_id": run_id,
        "status": LIVE_PILOT_STATUS,
        "run_scope": PILOT_RUN_SCOPE,
        "full_production_authorized": False,
        "created_at": created_at,
        "network_calls": 0,
        "database_reads": 0,
        "credential_reads": 0,
        "model": MODEL,
        "project": PROJECT,
        "location": LOCATION,
        "row_count": PILOT_COCKTAIL_COUNT,
        "request_count": PILOT_REQUEST_COUNT,
        "shard_count": 8,
        "shard_size": LIVE_PILOT_SHARD_SIZE,
        "parent_cohort_row_count": CORPUS_ROWS,
        "parent_cohort_id_set_sha256": COHORT_ID_SET_SHA256,
        "parent_cohort_source_sha256": sha256_bytes(cohort_payload),
        "parent_frozen_source_sha256": sha256_bytes(frozen_payload),
        "selected_id_set_sha256": selected_ids_hash,
        "selected_cocktail_ids": [row.cocktail_id for row in selected],
        "selection_policy": SELECTION_POLICY,
        "registry_sha256": SENSORY_V2_REGISTRY.registry_sha256,
        "prompt_axis_registry_file_sha256": AXIS_REGISTRY_FILE_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "request_config_sha256": REQUEST_CONFIG_SHA256,
        "historical_reserve_usd": str(HISTORICAL_RESERVE_USD),
        "soft_stop_usd": str(SOFT_STOP_USD),
        "hard_creation_block_usd": str(HARD_CREATION_BLOCK_USD),
        "user_approval_marker_sha256": LIVE_PILOT_APPROVAL_MARKER_SHA256,
        "estimated_cost_usd": str(pilot_estimate.estimated_cost_usd),
        "gcs_lifecycle": GcsLifecycleContract().to_dict(),
        "shards": shard_records,
        "requests": request_records,
        "selection": {
            "policy": SELECTION_POLICY,
            "manual_cherry_pick": False,
            "source_row_count": CORPUS_ROWS,
            "selected_row_count": PILOT_COCKTAIL_COUNT,
            "selected_id_set_sha256": selected_ids_hash,
            "records": selected_records,
        },
        "parent_cohort": {
            "row_count": CORPUS_ROWS,
            "id_set_sha256": COHORT_ID_SET_SHA256,
            "cohort_source_sha256": sha256_bytes(cohort_payload),
            "frozen_source_sha256": sha256_bytes(frozen_payload),
        },
        "request_contract": {
            "record_count": PILOT_REQUEST_COUNT,
            "axis_count": len(SENSORY_V2_REGISTRY.axes),
            "request_config_sha256": REQUEST_CONFIG_SHA256,
            "input_filename": "requests-480.jsonl",
            "input_sha256": request_sha256,
            "prompt_envelope_diagnostics": diagnostics,
            "combined_input_filename": "requests-480.jsonl",
        },
        "cost": {
            "planning_input_tokens_per_request": PLANNING_INPUT_TOKEN_ENVELOPE,
            "planning_output_tokens_per_request": 32,
            "estimated_input_tokens": pilot_estimate.input_tokens,
            "estimated_output_tokens": pilot_estimate.output_tokens,
            "estimated_input_cost_usd": str(pilot_estimate.input_cost_usd),
            "estimated_output_cost_usd": str(pilot_estimate.output_cost_usd),
            "estimated_cost_usd": str(pilot_estimate.estimated_cost_usd),
            "historical_reserve_usd": str(HISTORICAL_RESERVE_USD),
            "projected_with_reserve_usd": str(projected_pilot),
            "full_run_estimated_cost_usd": str(full_estimate.estimated_cost_usd),
            "pilot_plus_full_plus_reserve_usd": str(projected_pilot_and_full),
            "soft_stop_usd": str(SOFT_STOP_USD),
            "hard_limit_usd": str(HARD_CREATION_BLOCK_USD),
            "pilot_gate_passed": projected_pilot < SOFT_STOP_USD,
            "pilot_plus_full_gate_passed": (
                projected_pilot_and_full < SOFT_STOP_USD
                and projected_pilot_and_full < HARD_CREATION_BLOCK_USD
            ),
            "measured_token_gate": "pending_provider_measurement",
        },
    }
    manifest_payload = json_bytes(manifest)
    ledger = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "PREPARED_NOT_SUBMITTED",
        "jobs": [],
        "actual_cost_usd": "0",
        "reserved_pilot_upper_bound_usd": str(pilot_estimate.estimated_cost_usd),
        "projected_pilot_with_historical_reserve_usd": str(projected_pilot),
        "projected_pilot_plus_full_with_historical_reserve_usd": str(
            projected_pilot_and_full
        ),
        "soft_stop_usd": str(SOFT_STOP_USD),
        "hard_limit_usd": str(HARD_CREATION_BLOCK_USD),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for shard_index, shard in enumerate(shards):
        atomic_create(
            output_dir / f"requests-{shard_index:02d}.jsonl",
            jsonl_bytes(shard),
        )
    atomic_create(output_dir / "requests-480.jsonl", request_payload)
    atomic_create(output_dir / "pilot-manifest.json", manifest_payload)
    atomic_create(output_dir / "cost-ledger.json", json_bytes(ledger))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-source", required=True, type=Path)
    parser.add_argument("--cohort-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--user-approval-marker", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manifest = prepare_real_pilot(
            frozen_source=arguments.frozen_source,
            cohort_source=arguments.cohort_source,
            output_dir=arguments.output_dir,
            run_id=arguments.run_id,
            created_at=arguments.created_at,
            user_approval_marker=arguments.user_approval_marker,
        )
    except (FileExistsError, OSError, VertexSensoryBatchError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": str(arguments.output_dir),
                "record_count": PILOT_REQUEST_COUNT,
                "network_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
