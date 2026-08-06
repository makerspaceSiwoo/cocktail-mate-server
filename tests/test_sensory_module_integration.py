from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import app.spherical_graph.pipeline as spherical_pipeline
from app.embedding_pipeline.core import file_sha256, save_vector_artifact
from app.sensory_embedding import (
    AxisSoftLabels,
    CocktailEmbedding,
    PositiveSelection,
    SENSORY_V2_REGISTRY,
    SensoryPositiveQueryEncoder,
    TeacherSoftLabelProjector,
    legacy_32_cocktail_adapter,
    legacy_32_contract,
    sensory_48_contract,
)
from app.spherical_graph import (
    CosineKMedoidsClusterer,
    SimilarityProvider,
    SphericalGraph,
    SphericalGraphConfig,
    VectorRecord,
)
from app.spherical_graph.adapters import (
    VectorSimilarityCosineProvider,
    build_spherical_graph_from_canonical,
    sensory_embeddings_to_mapping,
    sensory_embeddings_to_records,
)
from app.vector_similarity import (
    EXACT_COSINE_PROVIDER_ID,
    CanonicalNeighborArtifact,
    DenseVector,
    build_canonical_neighbor_artifact,
    cosine_similarity,
    rank_by_inner_product,
)
from scripts.build_spherical_graph import (
    build_graph_payload,
    load_csv_vector_artifact,
    main as build_spherical_graph_main,
)

SPACE_SHA256 = hashlib.sha256(b"integration-space").hexdigest()


def _write_embedding_csv(
    path: Path,
    rows: list[tuple[object, ...]],
    *,
    columns: tuple[str, ...] = (
        "cocktail_id",
        "cocktail_name_ko",
        "embedding",
    ),
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _sensory_embeddings() -> tuple[CocktailEmbedding, ...]:
    contract = sensory_48_contract(
        registry=SENSORY_V2_REGISTRY,
        space_sha256=SPACE_SHA256,
    )
    projector = TeacherSoftLabelProjector(contract)
    embeddings: list[CocktailEmbedding] = []
    for index in range(6):
        axes = tuple(
            AxisSoftLabels.from_values(
                axis.axis_id,
                (
                    (0.0, 0.0, 0.0, 1.0, 0.0)
                    if axis.axis_order == index
                    else (
                        (0.0, 1.0, 0.0, 0.0, 0.0)
                        if axis.axis_order == (index + 1) % 6
                        else (1.0, 0.0, 0.0, 0.0, 0.0)
                    )
                ),
            )
            for axis in SENSORY_V2_REGISTRY.axes
        )
        embeddings.append(projector.project(index + 1, axes).embedding)
    return tuple(embeddings)


def _legacy_embeddings() -> tuple[CocktailEmbedding, ...]:
    adapter = legacy_32_cocktail_adapter(legacy_32_contract(space_sha256=SPACE_SHA256))
    embeddings: list[CocktailEmbedding] = []
    for index in range(6):
        angle = 2.0 * math.pi * index / 6
        vector = [0.0] * 32
        vector[0] = math.cos(angle)
        vector[1] = math.sin(angle)
        embeddings.append(adapter.adapt(index + 1, vector))
    return tuple(embeddings)


def _assert_graph_matches_canonical(
    graph: SphericalGraph,
    artifact: CanonicalNeighborArtifact,
) -> None:
    assert tuple(
        (
            int(row.source_id),
            int(row.target_id),
            row.rank,
            row.similarity,
        )
        for row in graph.directed_neighbors
    ) == tuple(
        (row.source_id, row.target_id, row.rank, row.score)
        for row in artifact.directed_neighbors
    )
    assert tuple(
        (
            int(edge.source_id),
            int(edge.target_id),
            edge.similarity,
            edge.source_rank,
            edge.target_rank,
            edge.is_mutual,
        )
        for edge in graph.edges
        if edge.edge_kind == "cocktail_knn"
    ) == tuple(
        (
            edge.left_id,
            edge.right_id,
            edge.score,
            edge.left_rank,
            edge.right_rank,
            edge.mutual,
        )
        for edge in artifact.undirected_edges
    )


def _assert_serialized_graph_matches_canonical(payload: dict[str, Any]) -> None:
    canonical = payload["canonical_neighbors"]
    graph = payload["graph"]
    assert [
        (
            row["source_id"],
            row["target_id"],
            row["rank"],
            row["score"],
        )
        for row in canonical["directed_neighbors"]
    ] == [
        (
            int(row["source_id"]),
            int(row["target_id"]),
            row["rank"],
            row["similarity"],
        )
        for row in graph["directed_neighbors"]
    ]
    assert [
        (
            edge["left_id"],
            edge["right_id"],
            edge["score"],
            edge["left_rank"],
            edge["right_rank"],
            edge["mutual"],
        )
        for edge in canonical["undirected_edges"]
    ] == [
        (
            int(edge["source_id"]),
            int(edge["target_id"]),
            edge["similarity"],
            edge["source_rank"],
            edge["target_rank"],
            edge["is_mutual"],
        )
        for edge in graph["edges"]
        if edge["edge_kind"] == "cocktail_knn"
    ]


def test_sensory_query_flows_from_module1_to_module2_inner_product() -> None:
    embeddings = _sensory_embeddings()
    encoder = SensoryPositiveQueryEncoder(
        SENSORY_V2_REGISTRY,
        embeddings[0].contract,
    )

    query = encoder.encode((PositiveSelection("sweetness"),))
    matches = rank_by_inner_product(query.values, embeddings, k=2)

    assert math.fsum(query.values) == pytest.approx(1.0, abs=1e-15)
    assert [match.id for match in matches] == [1, 6]
    assert [match.score for match in matches] == [1.0, 0.2]
    assert all(match.negative_inner_product == -match.score for match in matches)


@pytest.mark.parametrize(
    "embeddings",
    [_sensory_embeddings(), _legacy_embeddings()],
    ids=["future-48d", "current-legacy-32d"],
)
def test_module1_records_flow_through_canonical_module2_artifact_to_module3(
    embeddings: tuple[CocktailEmbedding, ...],
) -> None:
    records = sensory_embeddings_to_records(tuple(reversed(embeddings)))
    mapping = sensory_embeddings_to_mapping(tuple(reversed(embeddings)))
    canonical = build_canonical_neighbor_artifact(embeddings, k=2)

    assert [record.node_id for record in records] == [
        str(index) for index in range(1, 7)
    ]
    assert mapping == {record.node_id: record.vector for record in records}

    result = build_spherical_graph_from_canonical(
        records,
        canonical,
        clusterer=CosineKMedoidsClusterer(cluster_count=2, seed=17),
        config=SphericalGraphConfig(k=2, seed=17, layout_iterations=30),
    )

    assert result.similarity_provider == canonical.provider_id
    assert len(result.directed_neighbors) == 6 * 2
    assert len(result.components) == 2
    assert all(node.node_kind == "cocktail" for node in result.nodes)
    assert all(edge.edge_kind == "cocktail_knn" for edge in result.edges)
    assert {node.node_id for node in result.nodes if node.visible} == set(mapping)
    _assert_graph_matches_canonical(result, canonical)


def test_numeric_ties_and_asymmetric_union_are_identical_across_modules() -> None:
    records = tuple(
        DenseVector.from_values(cocktail_id, (1.0, 0.0))
        for cocktail_id in (10, 6, 5, 4, 3, 2, 1)
    )
    canonical = build_canonical_neighbor_artifact(records)
    graph_records = tuple(
        VectorRecord(node_id=str(record.id), vector=record.values) for record in records
    )

    result = build_spherical_graph_from_canonical(
        graph_records,
        canonical,
        clusterer=CosineKMedoidsClusterer(cluster_count=2, seed=29),
        config=SphericalGraphConfig(
            k=5,
            seed=29,
            layout_iterations=30,
            report_only=True,
        ),
    )

    assert [row.target_id for row in canonical.recommendations_for(1)] == [
        2,
        3,
        4,
        5,
        6,
    ]
    assert [row.target_id for row in canonical.recommendations_for(10)] == [
        1,
        2,
        3,
        4,
        5,
    ]
    _assert_graph_matches_canonical(result, canonical)
    visible_degree = {cocktail_id: 0 for cocktail_id in canonical.vector_ids}
    for edge in result.edges:
        if edge.edge_kind == "cocktail_knn":
            visible_degree[int(edge.source_id)] += 1
            visible_degree[int(edge.target_id)] += 1
    assert visible_degree[1] == 6
    asymmetric = next(
        edge
        for edge in result.edges
        if edge.edge_kind == "cocktail_knn"
        and (edge.source_id, edge.target_id) == ("1", "10")
    )
    assert asymmetric.source_rank is None
    assert asymmetric.target_rank == 1
    assert not asymmetric.is_mutual


def test_production_builder_never_calls_module3_rank_or_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tuple(
        DenseVector.from_values(cocktail_id, (1.0, float(cocktail_id)))
        for cocktail_id in range(1, 8)
    )
    canonical = build_canonical_neighbor_artifact(records)
    graph_records = tuple(
        VectorRecord(node_id=str(record.id), vector=record.values) for record in records
    )

    def fail_if_called(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("module3 topology derivation must not run")

    monkeypatch.setattr(spherical_pipeline, "_directed_neighbors", fail_if_called)
    monkeypatch.setattr(spherical_pipeline, "_union_edges", fail_if_called)

    result = build_spherical_graph_from_canonical(
        graph_records,
        canonical,
        config=SphericalGraphConfig(k=5, seed=31, layout_iterations=10),
    )

    _assert_graph_matches_canonical(result, canonical)


def test_canonical_builder_rejects_vector_set_mismatch() -> None:
    records = tuple(
        DenseVector.from_values(cocktail_id, (1.0, float(cocktail_id)))
        for cocktail_id in range(1, 8)
    )
    canonical = build_canonical_neighbor_artifact(records)
    graph_records = [
        VectorRecord(node_id=str(record.id), vector=record.values) for record in records
    ]
    graph_records[0] = VectorRecord(node_id="1", vector=(2.0, 1.0))

    with pytest.raises(ValueError, match="vector-set SHA-256"):
        build_spherical_graph_from_canonical(
            tuple(graph_records),
            canonical,
            config=SphericalGraphConfig(k=5, layout_iterations=10),
        )


def test_module2_similarity_provider_delegates_exact_pair_scores() -> None:
    provider = VectorSimilarityCosineProvider()
    assert isinstance(provider, SimilarityProvider)
    vectors = np.asarray(
        [[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]],
        dtype=np.float64,
    )

    matrix = provider.pairwise_cosine(("1", "2", "3"), vectors)

    assert matrix[0, 1] == pytest.approx(cosine_similarity((1.0, 0.0), (1.0, 1.0)))
    assert matrix[0, 2] == -1.0
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), np.ones(3))


def test_embedding_adapter_rejects_mixed_contracts() -> None:
    sensory = _sensory_embeddings()[0]
    legacy = _legacy_embeddings()[0]

    with pytest.raises(ValueError, match="one embedding contract"):
        sensory_embeddings_to_records((sensory, legacy))


def test_core_spherical_graph_import_does_not_load_optional_modules() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import sys; import app.spherical_graph; "
                "assert 'app.sensory_embedding' not in sys.modules; "
                "assert 'app.vector_similarity' not in sys.modules"
            ),
        ),
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_cli_builds_atomic_hashed_json_without_mutating_small_npz(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "current-32.npz"
    output_path = tmp_path / "nested" / "spherical-graph.json"
    cocktail_ids = np.arange(1, 9, dtype=np.int64)
    vectors = np.zeros((8, 32), dtype=np.float32)
    for index in range(8):
        angle = 2.0 * math.pi * index / 8
        vectors[index, 0] = math.cos(angle)
        vectors[index, 1] = math.sin(angle)
    save_vector_artifact(
        input_path,
        cocktail_ids=cocktail_ids,
        cocktail_names=tuple(f"cocktail-{index}" for index in cocktail_ids),
        vectors=vectors,
        metadata={"space": "current-32d-test"},
    )
    input_sha256 = file_sha256(input_path)

    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "scripts.build_spherical_graph",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--k",
            "2",
            "--seed",
            "19",
            "--iterations",
            "30",
        ),
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert file_sha256(input_path) == input_sha256
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source"] == {
        "path": str(input_path.resolve()),
        "format": "npz",
        "sha256": input_sha256,
        "row_count": 8,
        "vector_dimension": 32,
        "metadata": {"space": "current-32d-test"},
    }
    assert len(payload["cocktails"]) == 8
    assert payload["graph"]["k"] == 2
    assert payload["graph"]["seed"] == 19
    assert payload["graph"]["similarity_provider"] == (
        "vector_similarity_exact_cosine_v1"
    )
    assert payload["canonical_neighbor_provenance"]["provider_id"] == (
        EXACT_COSINE_PROVIDER_ID
    )
    _assert_serialized_graph_matches_canonical(payload)
    assert len(payload["graph"]["components"]) == 7
    assert all(node["node_kind"] == "cocktail" for node in payload["graph"]["nodes"])
    assert all(
        edge["edge_kind"] == "cocktail_knn" for edge in payload["graph"]["edges"]
    )
    assert list(output_path.parent.glob(f".{output_path.name}.*")) == []


def test_cli_reads_strict_csv_and_keeps_source_immutable(tmp_path: Path) -> None:
    input_path = tmp_path / "current-32.csv"
    output_path = tmp_path / "spherical-graph.json"
    rows: list[tuple[object, ...]] = []
    for index in range(8):
        angle = 2.0 * math.pi * index / 8
        vector = [0.0] * 32
        vector[0] = math.cos(angle)
        vector[1] = math.sin(angle)
        rows.append(
            (
                index + 1,
                f"cocktail-{index + 1}",
                json.dumps(vector),
            )
        )
    _write_embedding_csv(input_path, rows)
    input_sha256 = file_sha256(input_path)

    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "scripts.build_spherical_graph",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--k",
            "2",
            "--seed",
            "23",
            "--iterations",
            "30",
        ),
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert file_sha256(input_path) == input_sha256
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source"] == {
        "path": str(input_path.resolve()),
        "format": "csv",
        "sha256": input_sha256,
        "row_count": 8,
        "vector_dimension": 32,
        "metadata": {},
    }
    assert payload["canonical_neighbor_provenance"]["provider_id"] == (
        EXACT_COSINE_PROVIDER_ID
    )
    _assert_serialized_graph_matches_canonical(payload)
    assert [row["cocktail_id"] for row in payload["cocktails"]] == list(range(1, 9))
    assert len(payload["graph"]["components"]) == 7
    assert all(node["node_kind"] == "cocktail" for node in payload["graph"]["nodes"])
    assert all(
        edge["edge_kind"] == "cocktail_knn" for edge in payload["graph"]["edges"]
    )


@pytest.mark.parametrize(
    "columns",
    [
        ("cocktail_id", "cocktail_name_ko"),
        ("cocktail_id", "cocktail_name_ko", "embedding", "extra"),
        ("cocktail_name_ko", "cocktail_id", "embedding"),
    ],
)
def test_csv_rejects_missing_extra_or_reordered_columns(
    tmp_path: Path,
    columns: tuple[str, ...],
) -> None:
    path = tmp_path / "invalid-columns.csv"
    _write_embedding_csv(path, [], columns=columns)

    with pytest.raises(ValueError, match="expected columns"):
        load_csv_vector_artifact(path)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [(0, "zero", "[1, 0]")],
            "positive",
        ),
        (
            [(1, "first", "[1, 0]"), (1, "duplicate", "[0, 1]")],
            "duplicate",
        ),
        (
            [(1, "   ", "[1, 0]")],
            "non-empty",
        ),
        (
            [(1, "malformed", "[1, 0]", "extra")],
            "column count",
        ),
    ],
)
def test_csv_rejects_invalid_ids_names_and_row_width(
    tmp_path: Path,
    rows: list[tuple[object, ...]],
    message: str,
) -> None:
    path = tmp_path / "invalid-identity.csv"
    _write_embedding_csv(path, rows)

    with pytest.raises(ValueError, match=message):
        load_csv_vector_artifact(path)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [(1, "not-array", json.dumps("vector"))],
            "JSON array",
        ),
        (
            [(1, "zero", "[0, 0]")],
            "non-zero",
        ),
        (
            [(1, "nan", "[NaN, 1]")],
            "finite",
        ),
        (
            [(1, "bool", "[true, 1]")],
            "finite",
        ),
        (
            [(1, "two", "[1, 0]"), (2, "three", "[1, 0, 0]")],
            "dimensions",
        ),
    ],
)
def test_csv_rejects_invalid_vector_json(
    tmp_path: Path,
    rows: list[tuple[object, ...]],
    message: str,
) -> None:
    path = tmp_path / "invalid-vector.csv"
    _write_embedding_csv(path, rows)

    with pytest.raises(ValueError, match=message):
        load_csv_vector_artifact(path)


def test_csv_requires_at_least_k_plus_one_rows(tmp_path: Path) -> None:
    path = tmp_path / "too-small.csv"
    _write_embedding_csv(
        path,
        [
            (1, "one", "[1, 0]"),
            (2, "two", "[0, 1]"),
        ],
    )

    with pytest.raises(ValueError, match=r"k \+ 1"):
        build_graph_payload(
            path,
            k=2,
            clusters=0,
            seed=1,
            iterations=1,
        )


def test_cli_refuses_to_replace_its_input(tmp_path: Path) -> None:
    path = tmp_path / "artifact.npz"

    with pytest.raises(ValueError, match="must differ"):
        build_spherical_graph_main(("--input", str(path), "--output", str(path)))
