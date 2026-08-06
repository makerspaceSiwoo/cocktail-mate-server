from __future__ import annotations

import json
import math
from collections.abc import Sequence

import numpy as np
import pytest

from app.spherical_graph import (
    CosineKMedoidsClusterer,
    ExactCosineSimilarity,
    GraphEdge,
    SphericalGraph,
    SphericalGraphConfig,
    VectorRecord,
    build_spherical_graph,
)


def _clustered_records() -> list[VectorRecord]:
    records: list[VectorRecord] = []
    angles = (-0.13, -0.08, -0.04, 0.0, 0.03, 0.07, 0.12)
    for index, angle in enumerate(angles):
        records.append(
            VectorRecord(
                f"alpha-{index}",
                (
                    float(math.cos(angle)),
                    0.0,
                    float(math.sin(angle)),
                    0.0,
                ),
            )
        )
        records.append(
            VectorRecord(
                f"beta-{index}",
                (
                    0.0,
                    float(math.cos(angle)),
                    0.0,
                    float(math.sin(angle)),
                ),
            )
        )
    return records


def _config() -> SphericalGraphConfig:
    return SphericalGraphConfig(
        seed=104729,
        layout_iterations=500,
        multistart_count=2,
    )


def _connected_two_cluster_records() -> list[VectorRecord]:
    records: list[VectorRecord] = []
    for index, angle in enumerate((-0.08, -0.02, 0.03, 0.09)):
        records.append(
            VectorRecord(
                f"left-{index}",
                (
                    float(math.cos(angle)),
                    0.0,
                    float(math.sin(angle)),
                    0.0,
                ),
            )
        )
        records.append(
            VectorRecord(
                f"right-{index}",
                (
                    0.0,
                    float(math.cos(angle)),
                    0.0,
                    float(math.sin(angle)),
                ),
            )
        )
    return records


def _angle(
    coordinates: dict[str, np.ndarray],
    left: str,
    right: str,
) -> float:
    dot = float(np.dot(coordinates[left], coordinates[right]))
    return math.acos(max(-1.0, min(1.0, dot)))


def test_union_top_five_components_and_private_hubs_are_not_published() -> None:
    result = build_spherical_graph(_clustered_records(), config=_config())

    assert len(result.directed_neighbors) == 14 * 5
    assert len(result.components) == 2
    assert {len(component.member_ids) for component in result.components} == {7}

    public_node_ids = {row["node_id"] for row in result.node_rows()}
    assert public_node_ids == {record.node_id for record in _clustered_records()}
    assert all(row["edge_kind"] == "cocktail_knn" for row in result.edge_rows())
    assert all(node.node_kind == "cocktail" for node in result.nodes)
    assert all(edge.edge_kind == "cocktail_knn" for edge in result.edges)
    assert result.layout_report["private_hub_count"] == 2
    assert result.layout_report["private_hub_edge_count"] == 15
    assert "__spherical_graph_hub__" not in json.dumps(result.to_dict())
    assert result.clusterer == "union_graph_connected_components_v1"
    assert (
        result.layout_report["clustering_policy"]
        == "union_graph_connected_components_v1"
    )
    assert result.layout_report["acceptance_passed"]


def test_coordinates_are_unit_norm_and_build_is_bitwise_deterministic() -> None:
    first = build_spherical_graph(_clustered_records(), config=_config())
    second = build_spherical_graph(
        list(reversed(_clustered_records())),
        config=_config(),
    )

    assert first.to_dict() == second.to_dict()
    for node in first.nodes:
        norm = math.sqrt(node.x * node.x + node.y * node.y + node.z * node.z)
        assert math.isclose(norm, 1.0, abs_tol=1e-12)
    assert first.layout_report["unit_norm_max_error"] <= 1e-12
    json.dumps(first.to_dict(), allow_nan=False)
    assert all(
        all(not isinstance(value, (list, dict, tuple)) for value in row.values())
        for row in first.node_rows(include_hidden=True)
    )
    assert all(
        all(not isinstance(value, (list, dict, tuple)) for value in row.values())
        for row in first.edge_rows(include_hidden=True)
    )


def test_graph_nonneighbors_are_ranked_without_similarity_matrix_forces() -> None:
    result = build_spherical_graph(_clustered_records(), config=_config())
    coordinates = {
        node.node_id: np.asarray([node.x, node.y, node.z])
        for node in result.nodes
        if node.node_kind == "cocktail"
    }
    public_edges = {
        _edge_key(edge.source_id, edge.target_id)
        for edge in result.edges
        if edge.edge_kind == "cocktail_knn"
    }
    high_nonedge = ("alpha-0", "alpha-6")
    assert _edge_key(*high_nonedge) not in public_edges
    high_angle = _angle(coordinates, *high_nonedge)
    low_angle = _angle(coordinates, "alpha-0", "beta-0")

    assert high_angle < low_angle
    assert result.layout_report["algorithm"] == "graph_only_spherical_force_v3"
    assert result.layout_report["multistart_count"] == 2
    assert (
        result.layout_report["negative_sampling_policy"]
        == "deterministic graph nonedges only; no similarity data"
    )


def test_edge_target_distance_is_monotonic_and_provider_protocol_is_pluggable() -> None:
    class DelegatingProvider:
        provider_id = "delegating-test-provider"

        def pairwise_cosine(
            self,
            node_ids: Sequence[str],
            vectors: np.ndarray,
        ) -> np.ndarray:
            normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            similarities: np.ndarray = normalized @ normalized.T
            return similarities

    class PrefixClusterer:
        clusterer_id = "prefix-high-dimensional-test-clusterer"

        def clusters(
            self,
            node_ids: Sequence[str],
            similarities: np.ndarray,
            union_edges: Sequence[GraphEdge],
        ) -> tuple[tuple[str, ...], tuple[str, ...]]:
            assert similarities.shape == (len(node_ids), len(node_ids))
            assert union_edges
            return (
                tuple(node_id for node_id in node_ids if node_id.startswith("alpha")),
                tuple(node_id for node_id in node_ids if node_id.startswith("beta")),
            )

    result = build_spherical_graph(
        _clustered_records(),
        similarity_provider=DelegatingProvider(),
        clusterer=PrefixClusterer(),
        config=_config(),
    )
    assert result.similarity_provider == "delegating-test-provider"
    assert result.clusterer == "prefix-high-dimensional-test-clusterer"

    ordered = sorted(
        result.directed_neighbors,
        key=lambda neighbor: neighbor.similarity,
        reverse=True,
    )
    distances = [neighbor.target_distance for neighbor in ordered]
    assert distances == sorted(distances)
    assert all(
        0.0 <= neighbor.target_distance <= math.pi
        for neighbor in result.directed_neighbors
    )


def test_cosine_k_medoids_splits_a_connected_union_graph_by_high_dimensional_data() -> (
    None
):
    records = _connected_two_cluster_records()
    result = build_spherical_graph(
        records,
        clusterer=CosineKMedoidsClusterer(
            cluster_count=2,
            max_iterations=25,
            seed=17,
        ),
        config=SphericalGraphConfig(
            k=5,
            seed=23,
            layout_iterations=150,
            multistart_count=2,
            report_only=True,
        ),
    )

    visible_edges = {
        _edge_key(edge.source_id, edge.target_id)
        for edge in result.edges
        if edge.visible
    }
    _assert_connected({record.node_id for record in records}, visible_edges)
    assert {component.member_ids for component in result.components} == {
        tuple(f"left-{index}" for index in range(4)),
        tuple(f"right-{index}" for index in range(4)),
    }
    assert result.clusterer.startswith("cosine_k_medoids_v1:k=2:")


def test_cosine_k_medoids_is_input_order_independent_and_handles_duplicate_vectors() -> (
    None
):
    records = _connected_two_cluster_records()
    ids = tuple(record.node_id for record in records)
    vectors = np.asarray([record.vector for record in records])
    similarities = ExactCosineSimilarity().pairwise_cosine(ids, vectors)
    clusterer = CosineKMedoidsClusterer(cluster_count=2, seed=101)

    forward = clusterer.clusters(ids, similarities, ())
    order = tuple(reversed(range(len(ids))))
    reverse = clusterer.clusters(
        tuple(ids[index] for index in order),
        similarities[np.ix_(order, order)],
        (),
    )
    assert forward == reverse

    duplicate_clusters = CosineKMedoidsClusterer(
        cluster_count=2,
        seed=3,
    ).clusters(
        ("a", "b", "c"),
        np.ones((3, 3)),
        (),
    )
    assert len(duplicate_clusters) == 2
    assert all(duplicate_clusters)
    assert sorted(node for group in duplicate_clusters for node in group) == [
        "a",
        "b",
        "c",
    ]


def test_cosine_k_medoids_rejects_invalid_cluster_counts() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CosineKMedoidsClusterer(cluster_count=0)
    with pytest.raises(ValueError, match="positive integer"):
        CosineKMedoidsClusterer(cluster_count=1, max_iterations=0)

    clusterer = CosineKMedoidsClusterer(cluster_count=3)
    with pytest.raises(ValueError, match="cannot exceed"):
        clusterer.clusters(
            ("a", "b"),
            np.eye(2),
            (),
        )


def node_ids(result: SphericalGraph) -> set[str]:
    return {node.node_id for node in result.nodes if node.node_kind == "cocktail"}


def _edge_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _assert_connected(
    nodes: set[str],
    edges: set[tuple[str, str]],
) -> None:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited: set[str] = set()
    stack = [min(nodes)]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency[node] - visited)
    assert visited == nodes
