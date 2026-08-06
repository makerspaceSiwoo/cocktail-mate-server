from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest

from app.sensory_embedding import (
    SENSORY_V2_LEVELS,
    SENSORY_V2_REGISTRY,
    teacher_source_sha256,
)
from app.sensory_embedding.vertex_batch import (
    COHORT_ID_SET_SHA256,
    RESULT_SCHEMA_VERSION,
    id_set_sha256,
)
from app.vector_similarity import graph48_ids_sha256
from scripts.build_sensory_artifacts import (
    CANONICAL_RUN_FILENAME,
    DIRECTED_FILENAME,
    GRAPH48_FILENAME,
    MANIFEST_FILENAME,
    PREFERENCE48_FILENAME,
    RAW240_FILENAME,
    SPHERICAL_FILENAME,
    UNION_FILENAME,
    SensoryArtifactError,
    build_sensory_artifacts,
    load_projection_ready_jsonl,
)


def _projection_record(cocktail_id: int) -> dict[str, object]:
    axes: list[dict[str, object]] = []
    raw: list[float] = []
    for axis in SENSORY_V2_REGISTRY.axes:
        primary = (cocktail_id + axis.axis_order) % 5
        secondary = (primary + 1) % 5
        probabilities = [0.0] * 5
        probabilities[primary] = 0.8
        probabilities[secondary] = 0.2
        raw.extend(probabilities)
        axes.append(
            {
                "axis_order": axis.axis_order,
                "axis_id": axis.axis_id,
                "probabilities": probabilities,
                "response_sha256": hashlib.sha256(
                    f"response:{cocktail_id}:{axis.axis_order}".encode()
                ).hexdigest(),
                "raw_response_sha256": hashlib.sha256(
                    f"raw:{cocktail_id}:{axis.axis_order}".encode()
                ).hexdigest(),
            }
        )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "cocktail_id": cocktail_id,
        "registry_sha256": SENSORY_V2_REGISTRY.registry_sha256,
        "labels": list(SENSORY_V2_LEVELS),
        "axes": axes,
        "raw_probabilities": raw,
        "source_sha256": teacher_source_sha256(
            SENSORY_V2_REGISTRY.registry_sha256,
            raw,
        ),
    }


def _write_projection_jsonl(path: Path, ids: range = range(1, 7)) -> None:
    path.write_text(
        "".join(
            json.dumps(
                _projection_record(cocktail_id),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for cocktail_id in ids
        ),
        encoding="utf-8",
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def test_partial_pipeline_writes_separate_vectors_canonical_topology_and_public_s2(
    tmp_path: Path,
) -> None:
    source = tmp_path / "projection-ready.jsonl"
    output = tmp_path / "artifacts"
    _write_projection_jsonl(source)
    source_before = source.read_bytes()

    manifest = build_sensory_artifacts(
        source,
        output,
        run_id="synthetic-partial-01",
        allow_partial=True,
        clusters=2,
        seed=17,
        iterations=2,
        multistarts=1,
        report_only=True,
    )

    assert source.read_bytes() == source_before
    assert set(path.name for path in output.iterdir()) == {
        RAW240_FILENAME,
        GRAPH48_FILENAME,
        PREFERENCE48_FILENAME,
        CANONICAL_RUN_FILENAME,
        DIRECTED_FILENAME,
        UNION_FILENAME,
        SPHERICAL_FILENAME,
        MANIFEST_FILENAME,
    }
    raw_rows = _csv_rows(output / RAW240_FILENAME)
    graph_rows = _csv_rows(output / GRAPH48_FILENAME)
    preference_rows = _csv_rows(output / PREFERENCE48_FILENAME)
    directed_rows = _csv_rows(output / DIRECTED_FILENAME)
    union_rows = _csv_rows(output / UNION_FILENAME)

    assert len(raw_rows) == len(graph_rows) == len(preference_rows) == 6
    assert len(directed_rows) == 30
    assert 15 <= len(union_rows) <= 30
    assert len(raw_rows[0]) == 7 + 240
    assert len(graph_rows[0]) == len(preference_rows[0]) == 10 + 48
    assert all(
        math.isclose(
            math.sqrt(
                math.fsum(
                    float(row[axis.axis_id]) ** 2 for axis in SENSORY_V2_REGISTRY.axes
                )
            ),
            1.0,
            abs_tol=1e-12,
        )
        for row in graph_rows
    )
    assert any(
        not math.isclose(
            float(graph_rows[0][axis.axis_id]),
            float(preference_rows[0][axis.axis_id]),
        )
        for axis in SENSORY_V2_REGISTRY.axes
    )

    public = json.loads((output / SPHERICAL_FILENAME).read_text(encoding="utf-8"))
    graph = public["graph"]
    assert public["public_hub_node_count"] == 0
    assert public["public_hub_edge_count"] == 0
    assert len(graph["nodes"]) == 6
    assert len(graph["edges"]) == len(union_rows)
    assert len(graph["directed_neighbors"]) == len(directed_rows)
    assert all(node["node_kind"] == "cocktail" for node in graph["nodes"])
    assert all(edge["edge_kind"] == "cocktail_knn" for edge in graph["edges"])
    assert "__spherical_graph_hub__:" not in json.dumps(public)
    assert all(
        math.isclose(
            math.sqrt(node["x"] ** 2 + node["y"] ** 2 + node["z"] ** 2),
            1.0,
            abs_tol=1e-12,
        )
        for node in graph["nodes"]
    )
    assert [
        (
            row["source_id"],
            row["target_id"],
            int(row["rank"]),
            float(row["cosine"]),
        )
        for row in directed_rows
    ] == [
        (
            row["source_id"],
            row["target_id"],
            row["rank"],
            row["similarity"],
        )
        for row in graph["directed_neighbors"]
    ]

    stored_manifest = json.loads(
        (output / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest == stored_manifest
    assert manifest["mode"] == "partial-test"
    assert manifest["database_reads"] == 0
    assert manifest["database_writes"] == 0
    assert manifest["network_calls"] == 0
    cohort = cast(dict[str, Any], manifest["cohort"])
    assert (
        cohort["ids_sha256"]
        == id_set_sha256(range(1, 7))
        != graph48_ids_sha256(range(1, 7))
        == cohort["graph48_ids_sha256"]
    )
    files = cast(dict[str, dict[str, Any]], manifest["files"])
    for filename, metadata in files.items():
        payload = (output / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
        assert len(payload) == metadata["bytes"]

    with pytest.raises(FileExistsError, match="refusing to replace"):
        build_sensory_artifacts(
            source,
            output,
            run_id="synthetic-partial-02",
            allow_partial=True,
            clusters=2,
            iterations=1,
            multistarts=1,
        )


def test_production_mode_rejects_partial_before_creating_output(tmp_path: Path) -> None:
    source = tmp_path / "projection-ready.jsonl"
    output = tmp_path / "artifacts"
    _write_projection_jsonl(source)

    with pytest.raises(SensoryArtifactError, match="current 602-cocktail cohort"):
        build_sensory_artifacts(
            source,
            output,
            run_id="not-production",
            clusters=2,
            iterations=1,
            multistarts=1,
        )

    assert not output.exists()


def test_projection_input_rejects_tampered_axis_and_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "projection-ready.jsonl"
    record = _projection_record(1)
    axes = record["axes"]
    assert isinstance(axes, list)
    axes[0]["axis_id"] = "wrong"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(SensoryArtifactError, match="registry order"):
        load_projection_ready_jsonl(source, allow_partial=True)

    _write_projection_jsonl(source)
    records = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["source_sha256"] = "0" * 64
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(SensoryArtifactError, match="source_sha256 mismatch"):
        load_projection_ready_jsonl(source, allow_partial=True)


def test_actual_current_cohort_uses_integer_id_hash_not_graph_string_hash() -> None:
    candidates = (
        Path(__file__).resolve().parents[3]
        / "cocktail-mate-server"
        / "taste-data"
        / "cocktail-taste-descriptions.csv",
        Path(__file__).resolve().parents[1]
        / "taste-data"
        / "cocktail-taste-descriptions.csv",
    )
    cohort_path = next((path for path in candidates if path.is_file()), None)
    if cohort_path is None:
        pytest.skip("local current-cohort audit CSV is unavailable")
    with cohort_path.open(encoding="utf-8-sig", newline="") as source:
        ids = tuple(int(row["cocktail_id"]) for row in csv.DictReader(source))

    assert len(ids) == 602
    assert id_set_sha256(ids) == COHORT_ID_SET_SHA256
    assert graph48_ids_sha256(ids) != COHORT_ID_SET_SHA256
