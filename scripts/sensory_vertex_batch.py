"""Create and inspect sensory Vertex Batch artifacts without network access."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.sensory_embedding.vertex_batch import (
    CORPUS_ROWS,
    SHARD_COUNT,
    ParsedDistribution,
    VertexSensoryBatchError,
    atomic_create,
    build_manifest,
    build_requests,
    frozen_csv_bytes,
    id_set_sha256,
    json_bytes,
    jsonl_bytes,
    load_cohort_ids_csv,
    load_source_csv,
    parse_recorded_outputs,
    project_ready_records,
    records_jsonl_bytes,
    sha256_bytes,
    utc_now,
    validate_pilot_token_counts,
)

DEFAULT_ARTIFACT_DIR = Path("sensory-batch")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze, build, parse, and aggregate local sensory-48 Vertex Batch "
            "artifacts. This command never calls Vertex, GCS, or a database."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser(
        "freeze",
        help="Create a canonical name-free recipe-facts CSV snapshot.",
    )
    freeze.add_argument("--input", type=Path, required=True)
    freeze.add_argument(
        "--cohort-ids",
        type=Path,
        help="CSV containing the exact current-catalog cocktail_id allowlist.",
    )
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--allow-partial", action="store_true")

    build = commands.add_parser(
        "build",
        help="Create eight request JSONL shards and a local manifest.",
    )
    build.add_argument("--input", type=Path, required=True)
    build.add_argument(
        "--cohort-ids",
        type=Path,
        help="CSV containing the exact current-catalog cocktail_id allowlist.",
    )
    build.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    build.add_argument("--run-id", required=True)
    build.add_argument("--created-at", default=None)
    build.add_argument("--sdk-version", default="not-installed-offline")
    build.add_argument(
        "--pilot-token-counts",
        type=Path,
        help=(
            "Optional JSON object of measured request-key token counts. A production "
            "job remains blocked until a reviewed pilot supplies this evidence."
        ),
    )
    build.add_argument("--allow-partial", action="store_true")

    parse = commands.add_parser(
        "parse",
        help="Parse eight recorded response shards; quarantine invalid A-E evidence.",
    )
    parse.add_argument("--manifest", type=Path, required=True)
    parse.add_argument(
        "--response",
        type=Path,
        action="append",
        required=True,
        help="Response shard in index order; repeat exactly eight times.",
    )
    parse.add_argument("--output", type=Path, required=True)
    parse.add_argument("--quarantine", type=Path, required=True)
    parse.add_argument("--summary", type=Path, required=True)

    project = commands.add_parser(
        "project",
        help="Aggregate parsed records into projection-ready 48xA-E distributions.",
    )
    project.add_argument("--input", type=Path, required=True)
    project.add_argument("--output", type=Path, required=True)
    project.add_argument("--allow-partial", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VertexSensoryBatchError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise VertexSensoryBatchError(f"{path} must contain a JSON object")
    return value


def _freeze(arguments: argparse.Namespace) -> dict[str, object]:
    if not arguments.allow_partial and arguments.cohort_ids is None:
        raise VertexSensoryBatchError(
            "full freeze requires --cohort-ids for the exact 602-ID cohort"
        )
    cohort_ids = (
        load_cohort_ids_csv(arguments.cohort_ids)
        if arguments.cohort_ids is not None
        else None
    )
    rows = load_source_csv(
        arguments.input,
        expected_rows=None if arguments.allow_partial else CORPUS_ROWS,
        included_cocktail_ids=cohort_ids,
    )
    payload = frozen_csv_bytes(rows)
    atomic_create(arguments.output, payload)
    return {
        "network_calls": 0,
        "row_count": len(rows),
        "source_sha256": sha256_bytes(arguments.input.read_bytes()),
        "cohort_source_sha256": (
            sha256_bytes(arguments.cohort_ids.read_bytes())
            if arguments.cohort_ids is not None
            else None
        ),
        "cohort_id_set_sha256": (
            id_set_sha256(cohort_ids) if cohort_ids is not None else None
        ),
        "sha256": sha256_bytes(payload),
        "output": str(arguments.output),
    }


def _build(arguments: argparse.Namespace) -> dict[str, object]:
    if not arguments.allow_partial and arguments.cohort_ids is None:
        raise VertexSensoryBatchError(
            "full build requires --cohort-ids for the exact 602-ID cohort"
        )
    cohort_ids = (
        load_cohort_ids_csv(arguments.cohort_ids)
        if arguments.cohort_ids is not None
        else None
    )
    source_bytes = arguments.input.read_bytes()
    rows = load_source_csv(
        arguments.input,
        expected_rows=None if arguments.allow_partial else CORPUS_ROWS,
        included_cocktail_ids=cohort_ids,
    )
    shards = build_requests(rows)
    pilot: dict[str, object] | None = None
    if arguments.pilot_token_counts is not None:
        raw_counts = _read_json(arguments.pilot_token_counts)
        counts: dict[str, int] = {}
        for key, value in raw_counts.items():
            if type(value) is not int:
                raise VertexSensoryBatchError(
                    "pilot token-count values must be integers"
                )
            counts[key] = value
        # The pilot may be a reviewed subset; validate exactly that subset.
        all_requests = {request.key: request for shard in shards for request in shard}
        unknown = set(counts) - set(all_requests)
        if unknown or not counts:
            raise VertexSensoryBatchError(
                "pilot token-count keys must be a non-empty request subset"
            )
        pilot = validate_pilot_token_counts(
            counts,
            [all_requests[key] for key in counts],
        )

    manifest = build_manifest(
        rows,
        shards,
        input_sha256=sha256_bytes(source_bytes),
        run_id=arguments.run_id,
        created_at=arguments.created_at or utc_now(),
        sdk_version=arguments.sdk_version,
        pilot_token_envelope=pilot,
        cohort_source_sha256=(
            sha256_bytes(arguments.cohort_ids.read_bytes())
            if arguments.cohort_ids is not None
            else None
        ),
        cohort_id_set_sha256=(
            id_set_sha256(cohort_ids) if cohort_ids is not None else None
        ),
    )
    paths = [
        arguments.output_dir / f"requests-{index:02d}.jsonl"
        for index in range(SHARD_COUNT)
    ]
    manifest_path = arguments.output_dir / "manifest.json"
    existing = [path for path in (*paths, manifest_path) if path.exists()]
    if existing:
        raise VertexSensoryBatchError(
            f"refusing to replace existing artifacts: {existing}"
        )
    for path, shard in zip(paths, shards, strict=True):
        atomic_create(path, jsonl_bytes(shard))
    atomic_create(manifest_path, json_bytes(manifest))
    return {
        "network_calls": 0,
        "row_count": len(rows),
        "request_count": sum(len(shard) for shard in shards),
        "shard_record_counts": [len(shard) for shard in shards],
        "manifest": str(manifest_path),
    }


def _parse(arguments: argparse.Namespace) -> dict[str, object]:
    manifest = _read_json(arguments.manifest)
    if len(arguments.response) != SHARD_COUNT:
        raise VertexSensoryBatchError("repeat --response exactly eight times")
    output_paths = (arguments.output, arguments.quarantine, arguments.summary)
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise VertexSensoryBatchError(
            f"refusing to replace existing artifacts: {existing}"
        )
    parsed, quarantined = parse_recorded_outputs(manifest, arguments.response)
    atomic_create(
        arguments.output,
        records_jsonl_bytes(record.to_dict() for record in parsed),
    )
    atomic_create(
        arguments.quarantine,
        records_jsonl_bytes(record.to_dict() for record in quarantined),
    )
    summary = {
        "schema_version": 1,
        "manifest_sha256": sha256_bytes(arguments.manifest.read_bytes()),
        "response_shard_sha256": [
            sha256_bytes(path.read_bytes()) for path in arguments.response
        ],
        "expected_records": manifest.get("request_count"),
        "accepted_records": len(parsed),
        "quarantined_records": len(quarantined),
        "complete": not quarantined and len(parsed) == manifest.get("request_count"),
    }
    atomic_create(arguments.summary, json_bytes(summary))
    return {"network_calls": 0, **summary}


def _load_distributions(path: Path) -> tuple[ParsedDistribution, ...]:
    records: list[ParsedDistribution] = []
    try:
        lines = path.read_bytes().splitlines()
    except OSError as error:
        raise VertexSensoryBatchError(f"cannot read parsed records {path}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise TypeError("record is not an object")
            if tuple(raw["labels"]) != ("A", "B", "C", "D", "E"):
                raise ValueError("labels are not ordered A-E")
            records.append(
                ParsedDistribution(
                    key=str(raw["key"]),
                    cocktail_id=int(raw["cocktail_id"]),
                    axis_order=int(raw["axis_order"]),
                    axis_id=str(raw["axis_id"]),
                    selected_label=str(raw["selected_label"]),
                    probabilities=tuple(float(value) for value in raw["probabilities"]),
                    response_sha256=str(raw["response_sha256"]),
                    raw_response_sha256=str(raw["raw_response_sha256"]),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise VertexSensoryBatchError(
                f"{path}:{line_number}: invalid parsed distribution: {error}"
            ) from error
    return tuple(records)


def _project(arguments: argparse.Namespace) -> dict[str, object]:
    distributions = _load_distributions(arguments.input)
    records = project_ready_records(
        distributions,
        expected_cocktails=None if arguments.allow_partial else CORPUS_ROWS,
    )
    atomic_create(arguments.output, records_jsonl_bytes(records))
    return {
        "network_calls": 0,
        "cocktail_count": len(records),
        "output": str(arguments.output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    handlers = {
        "freeze": _freeze,
        "build": _build,
        "parse": _parse,
        "project": _project,
    }
    try:
        result = handlers[arguments.command](arguments)
    except (OSError, VertexSensoryBatchError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
