from __future__ import annotations

import json
import inspect
import math

import numpy as np
import pytest

from app.spherical_graph import (
    ComponentSummary,
    DirectedNeighbor,
    GraphEdge,
    SphericalGraphConfig,
    SphericalLayoutQualityError,
    VectorRecord,
    build_spherical_graph,
    build_spherical_graph_from_topology,
    layout_spherical_graph,
    similarity_to_target_distance,
)


def _edge(
    left: str,
    right: str,
    *,
    similarity: float,
    left_rank: int | None,
    right_rank: int | None,
) -> GraphEdge:
    return GraphEdge(
        source_id=left,
        target_id=right,
        edge_kind="cocktail_knn",
        similarity=similarity,
        target_distance=similarity_to_target_distance(similarity),
        source_rank=left_rank,
        target_rank=right_rank,
        is_mutual=left_rank is not None and right_rank is not None,
        is_bridge=False,
        visible=True,
        recommendable=True,
    )


def _private_edge(
    left: str,
    right: str,
    *,
    kind: str,
    angle: float,
) -> GraphEdge:
    return GraphEdge(
        source_id=left,
        target_id=right,
        edge_kind=kind,  # type: ignore[arg-type]
        similarity=math.cos(angle),
        target_distance=angle,
        source_rank=None,
        target_rank=None,
        is_mutual=False,
        is_bridge=kind == "hub_mst",
        visible=False,
        recommendable=False,
    )


def _fixed_two_cluster_topology() -> tuple[
    tuple[str, ...],
    tuple[DirectedNeighbor, ...],
    tuple[GraphEdge, ...],
    tuple[ComponentSummary, ...],
    tuple[GraphEdge, ...],
    np.ndarray,
]:
    groups = (
        tuple(f"a-{index}" for index in range(6)),
        tuple(f"b-{index}" for index in range(6)),
    )
    node_ids = tuple(node_id for group in groups for node_id in group)
    directed: list[DirectedNeighbor] = []
    edges: list[GraphEdge] = []
    for group in groups:
        for source in group:
            targets = [target for target in group if target != source]
            for rank, target in enumerate(targets, start=1):
                score = math.cos(0.18 + 0.025 * rank)
                directed.append(
                    DirectedNeighbor(
                        source_id=source,
                        target_id=target,
                        rank=rank,
                        similarity=score,
                        target_distance=similarity_to_target_distance(score),
                    )
                )
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                left_rank = [item for item in group if item != left].index(right) + 1
                right_rank = [item for item in group if item != right].index(left) + 1
                # Canonical opposite rows must agree on one symmetric weight.
                score = math.cos(0.18 + 0.025 * min(left_rank, right_rank))
                for row_index, row in enumerate(directed):
                    if {row.source_id, row.target_id} == {left, right}:
                        directed[row_index] = DirectedNeighbor(
                            source_id=row.source_id,
                            target_id=row.target_id,
                            rank=row.rank,
                            similarity=score,
                            target_distance=similarity_to_target_distance(score),
                        )
                edges.append(
                    _edge(
                        left,
                        right,
                        similarity=score,
                        left_rank=left_rank,
                        right_rank=right_rank,
                    )
                )

    components = (
        ComponentSummary(
            component_id="cluster-a",
            hub_id="__spherical_graph_hub__:a",
            member_ids=groups[0],
            medoid_id=groups[0][0],
            centroid=(),
        ),
        ComponentSummary(
            component_id="cluster-b",
            hub_id="__spherical_graph_hub__:b",
            member_ids=groups[1],
            medoid_id=groups[1][0],
            centroid=(),
        ),
    )
    private_edges = tuple(
        _private_edge(
            component.hub_id,
            member_id,
            kind="hub_anchor",
            angle=0.20 + 0.01 * member_index,
        )
        for component, group in zip(components, groups, strict=True)
        for member_index, member_id in enumerate(group)
    ) + (
        _private_edge(
            components[0].hub_id,
            components[1].hub_id,
            kind="hub_mst",
            angle=2.20,
        ),
    )
    audit = np.full((len(node_ids), len(node_ids)), -0.8, dtype=np.float64)
    np.fill_diagonal(audit, 1.0)
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    for group in groups:
        for left in group:
            for right in group:
                audit[index[left], index[right]] = 0.9
    return (
        node_ids,
        tuple(directed),
        tuple(edges),
        components,
        private_edges,
        audit,
    )


def _layout(
    *,
    audit: np.ndarray | None,
    report_only: bool = False,
) -> object:
    node_ids, directed, edges, components, private_edges, _ = (
        _fixed_two_cluster_topology()
    )
    return build_spherical_graph_from_topology(
        tuple(reversed(node_ids)),
        directed_neighbors=directed,
        cocktail_edges=edges,
        components=components,
        private_hub_edges=private_edges,
        topology_provider_id="fixed-test-topology",
        clustering_policy="fixed-test-clusters",
        audit_similarities=audit,
        audit_node_ids=node_ids,
        config=SphericalGraphConfig(
            k=5,
            seed=41,
            layout_iterations=120,
            multistart_count=2,
            report_only=report_only,
        ),
    )


def test_graph_only_layout_reports_and_passes_all_acceptance_metrics() -> None:
    *_, audit = _fixed_two_cluster_topology()
    graph = _layout(audit=audit)
    report = graph.layout_report  # type: ignore[union-attr]

    assert report["mean_recall_at_5"] >= 0.60
    assert report["node_coverage_at_5"] >= 0.90
    assert report["bottom_decile_false_close_count"] == 0
    assert report["union_edge_rmse_radians"] <= 0.40
    assert report["unit_norm_max_error"] <= 1e-12
    assert report["acceptance_passed"]
    assert len(report["coordinate_sha256"]) == 64


def test_force_entrypoint_signature_is_graph_only() -> None:
    parameters = set(inspect.signature(layout_spherical_graph).parameters)
    assert parameters == {
        "node_ids",
        "cocktail_edges",
        "components",
        "private_hub_edges",
        "config",
    }
    assert parameters.isdisjoint(
        {
            "records",
            "vectors",
            "directed_neighbors",
            "similarities",
            "audit_similarities",
        }
    )


def test_canonical_positive_integer_node_ids_sort_numerically() -> None:
    graph = build_spherical_graph(
        (
            VectorRecord("10", (0.0, 1.0)),
            VectorRecord("2", (0.8, 0.2)),
            VectorRecord("1", (1.0, 0.0)),
        ),
        config=SphericalGraphConfig(
            k=1,
            seed=7,
            layout_iterations=20,
            multistart_count=1,
            report_only=True,
        ),
    )

    assert [node.node_id for node in graph.nodes] == ["1", "2", "10"]
    assert [component.member_ids for component in graph.components] == [
        ("1", "2", "10")
    ]


def test_audit_matrix_cannot_change_coordinates_or_multistart_selection() -> None:
    *_, audit = _fixed_two_cluster_topology()
    first = _layout(audit=audit, report_only=True)
    second = _layout(audit=-audit, report_only=True)

    assert first.nodes == second.nodes  # type: ignore[union-attr]
    assert (
        first.layout_report["coordinate_sha256"]  # type: ignore[union-attr]
        == second.layout_report["coordinate_sha256"]  # type: ignore[union-attr]
    )
    assert (
        first.layout_report["selected_start"]  # type: ignore[union-attr]
        == second.layout_report["selected_start"]  # type: ignore[union-attr]
    )


def test_hubs_participate_in_forces_but_never_enter_public_payload() -> None:
    graph = _layout(audit=None)
    serialized = json.dumps(graph.to_dict())  # type: ignore[union-attr]

    assert "__spherical_graph_hub__" not in serialized
    assert all(node.node_kind == "cocktail" for node in graph.nodes)  # type: ignore[union-attr]
    assert all(edge.edge_kind == "cocktail_knn" for edge in graph.edges)  # type: ignore[union-attr]
    assert graph.layout_report["private_hub_count"] == 2  # type: ignore[union-attr]
    assert graph.layout_report["private_hub_edge_count"] == 13  # type: ignore[union-attr]


def test_production_default_is_sixteen_starts_and_tests_can_request_fewer() -> None:
    assert SphericalGraphConfig().multistart_count == 16
    graph = _layout(audit=None)
    assert graph.layout_report["multistart_count"] == 2  # type: ignore[union-attr]
    assert len(graph.layout_report["multistart_seeds"]) == 2  # type: ignore[union-attr]


def test_report_only_exposes_failures_and_enforced_mode_raises() -> None:
    node_ids, directed, edges, components, private_edges, audit = (
        _fixed_two_cluster_topology()
    )
    bad_edges = tuple(
        _edge(
            edge.source_id,
            edge.target_id,
            similarity=-1.0,
            left_rank=edge.source_rank,
            right_rank=edge.target_rank,
        )
        for edge in edges
    )
    bad_directed = tuple(
        DirectedNeighbor(
            source_id=row.source_id,
            target_id=row.target_id,
            rank=row.rank,
            similarity=-1.0,
            target_distance=math.pi,
        )
        for row in directed
    )
    kwargs = {
        "directed_neighbors": bad_directed,
        "cocktail_edges": bad_edges,
        "components": components,
        "private_hub_edges": private_edges,
        "topology_provider_id": "bad-fixed-topology",
        "clustering_policy": "fixed-test-clusters",
        "audit_similarities": audit,
        "audit_node_ids": node_ids,
    }
    reported = build_spherical_graph_from_topology(
        node_ids,
        **kwargs,  # type: ignore[arg-type]
        config=SphericalGraphConfig(
            k=5,
            seed=3,
            layout_iterations=10,
            multistart_count=1,
            report_only=True,
        ),
    )
    assert not reported.layout_report["acceptance_passed"]

    with pytest.raises(SphericalLayoutQualityError, match="report_only=True"):
        build_spherical_graph_from_topology(
            node_ids,
            **kwargs,  # type: ignore[arg-type]
            config=SphericalGraphConfig(
                k=5,
                seed=3,
                layout_iterations=10,
                multistart_count=1,
            ),
        )
