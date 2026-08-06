"""Canonical cocktail topology preparation and graph-only S² layout."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np

from app.spherical_graph.clustering import (
    HighDimensionalClusterer,
    UnionGraphComponentClusterer,
)
from app.spherical_graph.models import (
    ComponentSummary,
    DirectedNeighbor,
    GraphEdge,
    GraphNode,
    SphericalGraph,
    VectorRecord,
)
from app.spherical_graph.similarity import (
    ExactCosineSimilarity,
    SimilarityProvider,
    validate_similarity_matrix,
)


_HUB_PREFIX = "__spherical_graph_hub__:"
_MEAN_RECALL_AT_5_MIN = 0.60
_NODE_COVERAGE_AT_5_MIN = 0.90
_BOTTOM_DECILE_FALSE_CLOSE_MAX = 0
_UNION_EDGE_RMSE_MAX = 0.40
_UNIT_NORM_MAX_ERROR = 1e-12


class SphericalLayoutQualityError(ValueError):
    """Raised when an enforced layout misses an acceptance threshold."""


@dataclass(frozen=True, slots=True)
class SphericalGraphConfig:
    """Deterministic graph-only force-layout configuration.

    ``multistart_count`` is 16 for production. Tests may explicitly use fewer
    starts. ``report_only`` records quality failures instead of raising.
    """

    k: int = 5
    seed: int = 20260806
    layout_iterations: int = 450
    multistart_count: int = 16
    learning_rate: float = 0.045
    edge_weight: float = 1.0
    hub_edge_weight: float = 0.30
    nonedge_margin: float = 0.52
    nonedge_samples_per_node: int = 18
    nonedge_repulsion_weight: float = 0.30
    ranking_margin: float = 0.06
    ranking_weight: float = 0.30
    ranking_nonedges_per_edge: int = 3
    initialization_weight: float = 0.006
    cluster_initial_radius: float = 0.30
    constraint_projection_passes: int = 4
    constraint_projection_slack: float = 1e-6
    report_only: bool = False


def similarity_to_target_distance(similarity: float) -> float:
    """Convert a cosine directly to its clamped angular distance in radians."""

    if not math.isfinite(similarity):
        raise ValueError("similarity must be finite")
    return math.acos(max(-1.0, min(1.0, similarity)))


def build_spherical_graph(
    records: Sequence[VectorRecord] | Mapping[str, Sequence[float]],
    *,
    similarity_provider: SimilarityProvider | None = None,
    clusterer: HighDimensionalClusterer | None = None,
    config: SphericalGraphConfig = SphericalGraphConfig(),
) -> SphericalGraph:
    """Precompute topology from vectors, then invoke the graph-only layout.

    This compatibility helper owns all vector and full-matrix work. The force
    layout it calls receives no vectors and no similarity matrix.
    """

    _validate_config(config)
    node_ids, vectors = _coerce_records(records)
    if len(node_ids) <= config.k:
        raise ValueError("record count must be greater than k")

    provider = similarity_provider or ExactCosineSimilarity()
    provider_id = getattr(provider, "provider_id", None)
    if not isinstance(provider_id, str) or not provider_id:
        raise ValueError("similarity provider must expose a non-empty provider_id")
    similarities = validate_similarity_matrix(
        provider.pairwise_cosine(node_ids, vectors),
        row_count=len(node_ids),
    )
    directed, ranks = _directed_neighbors(node_ids, similarities, config.k)
    cocktail_edges = _union_edges(node_ids, similarities, ranks)
    components, private_hub_edges, clusterer_id = _prepare_clusters_and_hubs(
        node_ids=node_ids,
        vectors=vectors,
        similarities=similarities,
        cocktail_edges=cocktail_edges,
        clusterer=clusterer,
    )
    return build_spherical_graph_from_topology(
        node_ids,
        directed_neighbors=directed,
        cocktail_edges=cocktail_edges,
        components=components,
        private_hub_edges=private_hub_edges,
        topology_provider_id=provider_id,
        clustering_policy=clusterer_id,
        audit_similarities=similarities,
        audit_node_ids=node_ids,
        config=config,
    )


def build_spherical_graph_from_topology(
    node_ids: Sequence[str],
    *,
    directed_neighbors: Sequence[DirectedNeighbor],
    cocktail_edges: Sequence[GraphEdge],
    components: Sequence[ComponentSummary],
    private_hub_edges: Sequence[GraphEdge],
    topology_provider_id: str,
    clustering_policy: str,
    audit_similarities: (np.ndarray | Mapping[tuple[str, str], float] | None) = None,
    audit_node_ids: Sequence[str] | None = None,
    config: SphericalGraphConfig = SphericalGraphConfig(),
) -> SphericalGraph:
    """Lay out an already-fixed weighted graph on S².

    The force solver consumes only cocktail IDs, union edges, a fixed cluster
    partition, and a fixed private hub graph. Canonical directed neighbors are
    used after layout for exhaustive Recall@5 reporting. Optional similarities
    are likewise audit-only and cannot affect coordinates or multistart choice.
    """

    _validate_config(config)
    ordered_ids = _coerce_node_ids(node_ids)
    if len(ordered_ids) <= config.k:
        raise ValueError("node count must be greater than k")
    if not topology_provider_id:
        raise ValueError("topology_provider_id must be non-empty")
    if not clustering_policy:
        raise ValueError("clustering_policy must be non-empty")

    directed = tuple(directed_neighbors)
    edges = tuple(cocktail_edges)
    fixed_components = _validate_components(ordered_ids, components)
    fixed_hubs = tuple(private_hub_edges)
    _validate_supplied_topology(
        node_ids=ordered_ids,
        directed=directed,
        cocktail_edges=edges,
        k=config.k,
    )
    _validate_private_hub_graph(
        node_ids=ordered_ids,
        components=fixed_components,
        private_hub_edges=fixed_hubs,
        cocktail_edges=edges,
    )

    coordinate_map, layout_diagnostics = layout_spherical_graph(
        node_ids=ordered_ids,
        cocktail_edges=edges,
        components=fixed_components,
        private_hub_edges=fixed_hubs,
        config=config,
    )
    nodes = _public_graph_nodes(
        node_ids=ordered_ids,
        coordinates=coordinate_map,
        components=fixed_components,
    )
    report = _evaluate_layout(
        node_ids=ordered_ids,
        coordinate_map=coordinate_map,
        directed_neighbors=directed,
        cocktail_edges=edges,
        audit_similarities=audit_similarities,
        audit_node_ids=audit_node_ids,
        clustering_policy=clustering_policy,
        layout_diagnostics=layout_diagnostics,
        config=config,
    )
    _enforce_quality(report, report_only=config.report_only)

    # Hubs took part in every force iteration, but are deliberately discarded.
    return SphericalGraph(
        k=config.k,
        seed=config.seed,
        similarity_provider=topology_provider_id,
        clusterer=clustering_policy,
        directed_neighbors=directed,
        nodes=nodes,
        edges=edges,
        components=fixed_components,
        layout_report=report,
    )


def prepare_spherical_graph_topology(
    records: Sequence[VectorRecord] | Mapping[str, Sequence[float]],
    *,
    directed_neighbors: Sequence[DirectedNeighbor],
    cocktail_edges: Sequence[GraphEdge],
    clusterer: HighDimensionalClusterer | None = None,
    similarity_provider: SimilarityProvider | None = None,
) -> tuple[
    tuple[str, ...],
    tuple[ComponentSummary, ...],
    tuple[GraphEdge, ...],
    str,
    np.ndarray,
]:
    """Adapt canonical rows into fixed cluster/hub metadata before layout.

    This helper is intentionally outside the graph-only force boundary. Its
    returned matrix is for optional post-layout audit only.
    """

    node_ids, vectors = _coerce_records(records)
    provider = similarity_provider or ExactCosineSimilarity()
    similarities = validate_similarity_matrix(
        provider.pairwise_cosine(node_ids, vectors),
        row_count=len(node_ids),
    )
    directed = tuple(directed_neighbors)
    edges = tuple(cocktail_edges)
    _validate_supplied_topology(
        node_ids=node_ids,
        directed=directed,
        cocktail_edges=edges,
        k=len(directed) // len(node_ids),
    )
    components, hidden_edges, clusterer_id = _prepare_clusters_and_hubs(
        node_ids=node_ids,
        vectors=vectors,
        similarities=similarities,
        cocktail_edges=edges,
        clusterer=clusterer,
    )
    return node_ids, components, hidden_edges, clusterer_id, similarities


def _prepare_clusters_and_hubs(
    *,
    node_ids: tuple[str, ...],
    vectors: np.ndarray,
    similarities: np.ndarray,
    cocktail_edges: tuple[GraphEdge, ...],
    clusterer: HighDimensionalClusterer | None,
) -> tuple[tuple[ComponentSummary, ...], tuple[GraphEdge, ...], str]:
    normalized_vectors = _normalize_rows(vectors)
    selected_clusterer = clusterer or UnionGraphComponentClusterer()
    clusterer_id = getattr(selected_clusterer, "clusterer_id", None)
    if not isinstance(clusterer_id, str) or not clusterer_id:
        raise ValueError("clusterer must expose a non-empty clusterer_id")
    members = _validated_clusters(
        node_ids,
        selected_clusterer.clusters(
            node_ids,
            similarities.copy(),
            cocktail_edges,
        ),
    )
    components = _component_summaries(
        node_ids,
        normalized_vectors,
        similarities,
        members,
    )
    return (
        components,
        _hidden_hub_edges(components, node_ids, normalized_vectors),
        clusterer_id,
    )


def _validate_config(config: SphericalGraphConfig) -> None:
    integer_positive = {
        "k": config.k,
        "layout_iterations": config.layout_iterations,
        "multistart_count": config.multistart_count,
        "nonedge_samples_per_node": config.nonedge_samples_per_node,
        "ranking_nonedges_per_edge": config.ranking_nonedges_per_edge,
    }
    if any(type(value) is not int or value <= 0 for value in integer_positive.values()):
        raise ValueError("layout counts and k must be positive integers")
    if type(config.seed) is not int:
        raise ValueError("seed must be an integer")
    if (
        type(config.constraint_projection_passes) is not int
        or config.constraint_projection_passes < 0
    ):
        raise ValueError("constraint_projection_passes must be non-negative")
    positive = (
        config.learning_rate,
        config.edge_weight,
        config.hub_edge_weight,
        config.nonedge_repulsion_weight,
        config.ranking_weight,
        config.cluster_initial_radius,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError(
            "layout rates, weights, and radius must be finite and positive"
        )
    if not 0.0 < config.nonedge_margin < math.pi:
        raise ValueError("nonedge_margin must be within (0, pi)")
    if not math.isfinite(config.ranking_margin) or config.ranking_margin < 0.0:
        raise ValueError("ranking_margin must be finite and non-negative")
    if (
        not math.isfinite(config.initialization_weight)
        or config.initialization_weight < 0.0
    ):
        raise ValueError("initialization_weight must be finite and non-negative")
    if (
        not math.isfinite(config.constraint_projection_slack)
        or config.constraint_projection_slack < 0.0
    ):
        raise ValueError("constraint_projection_slack must be non-negative")
    if type(config.report_only) is not bool:
        raise ValueError("report_only must be a boolean")


def _coerce_node_ids(node_ids: Sequence[str]) -> tuple[str, ...]:
    ordered = tuple(sorted((str(node_id) for node_id in node_ids), key=_node_id_key))
    if not ordered:
        raise ValueError("at least one cocktail node is required")
    if any(not node_id or node_id.startswith(_HUB_PREFIX) for node_id in ordered):
        raise ValueError("node IDs must be non-empty and outside the hub namespace")
    if len(set(ordered)) != len(ordered):
        raise ValueError("node IDs must be unique")
    return ordered


def _coerce_records(
    records: Sequence[VectorRecord] | Mapping[str, Sequence[float]],
) -> tuple[tuple[str, ...], np.ndarray]:
    if isinstance(records, Mapping):
        source: Iterable[VectorRecord] = (
            VectorRecord(str(node_id), tuple(vector))
            for node_id, vector in records.items()
        )
    else:
        source = records
    rows = sorted(source, key=lambda record: _node_id_key(record.node_id))
    node_ids = _coerce_node_ids(tuple(record.node_id for record in rows))
    dimensions = {len(record.vector) for record in rows}
    if len(dimensions) != 1 or next(iter(dimensions)) <= 0:
        raise ValueError("all source vectors must share a positive dimension")
    vectors = np.asarray([record.vector for record in rows], dtype=np.float64)
    if vectors.ndim != 2 or not np.all(np.isfinite(vectors)):
        raise ValueError("source vectors must form a finite two-dimensional matrix")
    if np.any(np.linalg.norm(vectors, axis=1) <= 1e-12):
        raise ValueError("source vectors must be non-zero")
    return node_ids, vectors


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return cast(np.ndarray, vectors / norms)


def _node_id_key(node_id: str) -> tuple[int, int, str]:
    """Sort canonical positive integer IDs numerically, all other IDs lexically."""

    try:
        numeric_id = int(node_id)
    except ValueError:
        return (1, 0, node_id)
    if numeric_id > 0 and str(numeric_id) == node_id:
        return (0, numeric_id, "")
    return (1, 0, node_id)


def _ordered_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if _node_id_key(left) < _node_id_key(right) else (right, left)


def _directed_neighbors(
    node_ids: tuple[str, ...],
    similarities: np.ndarray,
    k: int,
) -> tuple[tuple[DirectedNeighbor, ...], dict[tuple[str, str], int]]:
    rows: list[DirectedNeighbor] = []
    ranks: dict[tuple[str, str], int] = {}
    for source_index, source_id in enumerate(node_ids):
        candidates = [
            target_index
            for target_index in range(len(node_ids))
            if target_index != source_index
        ]
        candidates.sort(
            key=lambda target: (
                -float(similarities[source_index, target]),
                _node_id_key(node_ids[target]),
            )
        )
        for rank, target_index in enumerate(candidates[:k], start=1):
            target_id = node_ids[target_index]
            similarity = float(similarities[source_index, target_index])
            rows.append(
                DirectedNeighbor(
                    source_id=source_id,
                    target_id=target_id,
                    rank=rank,
                    similarity=similarity,
                    target_distance=similarity_to_target_distance(similarity),
                )
            )
            ranks[(source_id, target_id)] = rank
    return tuple(rows), ranks


def _union_edges(
    node_ids: tuple[str, ...],
    similarities: np.ndarray,
    directed_rank: dict[tuple[str, str], int],
) -> tuple[GraphEdge, ...]:
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    pairs = {
        _ordered_pair(source_id, target_id) for source_id, target_id in directed_rank
    }
    edges: list[GraphEdge] = []
    for source_id, target_id in sorted(
        pairs,
        key=lambda pair: (_node_id_key(pair[0]), _node_id_key(pair[1])),
    ):
        source_rank = directed_rank.get((source_id, target_id))
        target_rank = directed_rank.get((target_id, source_id))
        similarity = float(similarities[index[source_id], index[target_id]])
        edges.append(
            GraphEdge(
                source_id=source_id,
                target_id=target_id,
                edge_kind="cocktail_knn",
                similarity=similarity,
                target_distance=similarity_to_target_distance(similarity),
                source_rank=source_rank,
                target_rank=target_rank,
                is_mutual=source_rank is not None and target_rank is not None,
                is_bridge=False,
                visible=True,
                recommendable=True,
            )
        )
    return tuple(edges)


def _validate_supplied_topology(
    *,
    node_ids: tuple[str, ...],
    directed: tuple[DirectedNeighbor, ...],
    cocktail_edges: tuple[GraphEdge, ...],
    k: int,
) -> None:
    known = set(node_ids)
    if len(directed) != len(node_ids) * k:
        raise ValueError("canonical topology must have exactly k rows per node")
    directions: dict[tuple[str, str], DirectedNeighbor] = {}
    ranks: dict[str, list[int]] = {node_id: [] for node_id in node_ids}
    for row in directed:
        direction = (row.source_id, row.target_id)
        if (
            row.source_id not in known
            or row.target_id not in known
            or row.source_id == row.target_id
            or direction in directions
        ):
            raise ValueError("canonical directed endpoints are invalid or repeated")
        if not 1 <= row.rank <= k:
            raise ValueError("canonical directed rank is invalid")
        if not math.isclose(
            row.target_distance,
            similarity_to_target_distance(row.similarity),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("canonical directed angular target is invalid")
        directions[direction] = row
        ranks[row.source_id].append(row.rank)
    if any(sorted(value) != list(range(1, k + 1)) for value in ranks.values()):
        raise ValueError("each canonical source must have ranks 1 through k")

    expected_pairs = {
        frozenset((source_id, target_id)) for source_id, target_id in directions
    }
    seen: set[frozenset[str]] = set()
    for edge in cocktail_edges:
        pair = frozenset((edge.source_id, edge.target_id))
        source_row = directions.get((edge.source_id, edge.target_id))
        target_row = directions.get((edge.target_id, edge.source_id))
        expected_source_rank = source_row.rank if source_row else None
        expected_target_rank = target_row.rank if target_row else None
        expected_score = (
            source_row.similarity
            if source_row is not None
            else target_row.similarity
            if target_row is not None
            else math.nan
        )
        if (
            edge.edge_kind != "cocktail_knn"
            or edge.source_id not in known
            or edge.target_id not in known
            or edge.source_id == edge.target_id
            or pair in seen
            or not edge.visible
            or not edge.recommendable
            or edge.is_bridge
            or edge.source_rank != expected_source_rank
            or edge.target_rank != expected_target_rank
            or edge.is_mutual
            != (expected_source_rank is not None and expected_target_rank is not None)
            or not math.isclose(edge.similarity, expected_score, abs_tol=1e-15)
            or not math.isclose(
                edge.target_distance,
                similarity_to_target_distance(edge.similarity),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("canonical cocktail edge is invalid")
        if (
            source_row is not None
            and target_row is not None
            and not math.isclose(
                source_row.similarity,
                target_row.similarity,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("opposite canonical rows disagree on edge weight")
        seen.add(pair)
    if seen != expected_pairs:
        raise ValueError("cocktail edges must equal the either-direction union")


def _validated_clusters(
    node_ids: tuple[str, ...],
    value: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    known = set(node_ids)
    components = [tuple(sorted(members, key=_node_id_key)) for members in value]
    if not components or any(not members for members in components):
        raise ValueError("clusterer must return non-empty clusters")
    flattened = [node_id for members in components for node_id in members]
    if len(flattened) != len(set(flattened)) or set(flattened) != known:
        raise ValueError("clusterer must return an exhaustive node partition")
    return tuple(sorted(components, key=lambda members: _node_id_key(members[0])))


def _component_summaries(
    node_ids: tuple[str, ...],
    normalized_vectors: np.ndarray,
    similarities: np.ndarray,
    component_members: tuple[tuple[str, ...], ...],
) -> tuple[ComponentSummary, ...]:
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    summaries: list[ComponentSummary] = []
    for component_index, member_ids in enumerate(component_members):
        member_indices = np.asarray([index[node_id] for node_id in member_ids])
        local_similarities = similarities[np.ix_(member_indices, member_indices)]
        medoid_position = min(
            range(len(member_ids)),
            key=lambda position: (
                -float(local_similarities[position].mean()),
                _node_id_key(member_ids[position]),
            ),
        )
        centroid = normalized_vectors[member_indices].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= 1e-12:
            centroid = normalized_vectors[member_indices[medoid_position]]
        else:
            centroid = centroid / norm
        summaries.append(
            ComponentSummary(
                component_id=f"component-{component_index:04d}",
                hub_id=f"{_HUB_PREFIX}{component_index:04d}",
                member_ids=member_ids,
                medoid_id=member_ids[medoid_position],
                centroid=tuple(float(value) for value in centroid),
            )
        )
    return tuple(summaries)


def _validate_components(
    node_ids: tuple[str, ...],
    components: Sequence[ComponentSummary],
) -> tuple[ComponentSummary, ...]:
    ordered = tuple(sorted(components, key=lambda component: component.component_id))
    if not ordered:
        raise ValueError("at least one precomputed component is required")
    if len({component.component_id for component in ordered}) != len(ordered):
        raise ValueError("component IDs must be unique")
    if len({component.hub_id for component in ordered}) != len(ordered):
        raise ValueError("private hub IDs must be unique")
    if any(
        not component.hub_id.startswith(_HUB_PREFIX)
        or component.medoid_id not in component.member_ids
        for component in ordered
    ):
        raise ValueError("component hub namespace or medoid is invalid")
    _validated_clusters(node_ids, [component.member_ids for component in ordered])
    return ordered


def _hidden_hub_edges(
    components: tuple[ComponentSummary, ...],
    node_ids: tuple[str, ...],
    normalized_vectors: np.ndarray,
) -> tuple[GraphEdge, ...]:
    vector_by_id = dict(zip(node_ids, normalized_vectors, strict=True))
    edges: list[GraphEdge] = []
    for component in components:
        centroid = np.asarray(component.centroid, dtype=np.float64)
        for member_id in component.member_ids:
            score = float(np.clip(np.dot(centroid, vector_by_id[member_id]), -1.0, 1.0))
            edges.append(
                _private_edge(
                    component.hub_id,
                    member_id,
                    edge_kind="hub_anchor",
                    similarity=score,
                    bridge=False,
                )
            )
    if len(components) == 1:
        return tuple(edges)

    candidates: list[tuple[float, str, str, float]] = []
    for left_index, left in enumerate(components):
        for right in components[left_index + 1 :]:
            score = float(
                np.clip(
                    np.dot(np.asarray(left.centroid), np.asarray(right.centroid)),
                    -1.0,
                    1.0,
                )
            )
            candidates.append(
                (
                    similarity_to_target_distance(score),
                    left.component_id,
                    right.component_id,
                    score,
                )
            )
    candidates.sort()
    parent = {
        component.component_id: component.component_id for component in components
    }

    def find(component_id: str) -> str:
        while parent[component_id] != component_id:
            parent[component_id] = parent[parent[component_id]]
            component_id = parent[component_id]
        return component_id

    by_id = {component.component_id: component for component in components}
    for _, left_id, right_id, score in candidates:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root == right_root:
            continue
        parent[right_root] = left_root
        edges.append(
            _private_edge(
                by_id[left_id].hub_id,
                by_id[right_id].hub_id,
                edge_kind="hub_mst",
                similarity=score,
                bridge=True,
            )
        )
        if sum(edge.edge_kind == "hub_mst" for edge in edges) == len(components) - 1:
            break
    return tuple(edges)


def _private_edge(
    source_id: str,
    target_id: str,
    *,
    edge_kind: str,
    similarity: float,
    bridge: bool,
) -> GraphEdge:
    return GraphEdge(
        source_id=source_id,
        target_id=target_id,
        edge_kind=cast(Any, edge_kind),
        similarity=similarity,
        target_distance=similarity_to_target_distance(similarity),
        source_rank=None,
        target_rank=None,
        is_mutual=False,
        is_bridge=bridge,
        visible=False,
        recommendable=False,
    )


def _validate_private_hub_graph(
    *,
    node_ids: tuple[str, ...],
    components: tuple[ComponentSummary, ...],
    private_hub_edges: tuple[GraphEdge, ...],
    cocktail_edges: tuple[GraphEdge, ...],
) -> None:
    cocktail_ids = set(node_ids)
    hub_ids = {component.hub_id for component in components}
    component_by_node = {
        member: component for component in components for member in component.member_ids
    }
    component_by_hub = {component.hub_id: component for component in components}
    anchored_members: dict[str, set[str]] = {hub_id: set() for hub_id in hub_ids}
    hub_edge_count = 0
    for edge in private_hub_edges:
        if edge.visible or edge.recommendable:
            raise ValueError("private hub edges must be hidden and non-recommendable")
        if not math.isclose(
            edge.target_distance,
            similarity_to_target_distance(edge.similarity),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("private hub angular target is invalid")
        endpoints = {edge.source_id, edge.target_id}
        if edge.edge_kind == "hub_anchor":
            hubs = endpoints & hub_ids
            cocktails = endpoints & cocktail_ids
            if len(hubs) != 1 or len(cocktails) != 1:
                raise ValueError("hub anchor must join one hub to one cocktail")
            hub_id = next(iter(hubs))
            cocktail_id = next(iter(cocktails))
            if component_by_hub[hub_id] != component_by_node[cocktail_id]:
                raise ValueError("hub anchor crossed component boundaries")
            if cocktail_id in anchored_members[hub_id]:
                raise ValueError("private hub member anchor was repeated")
            anchored_members[hub_id].add(cocktail_id)
        elif edge.edge_kind == "hub_mst":
            if endpoints <= hub_ids and len(endpoints) == 2 and edge.is_bridge:
                hub_edge_count += 1
            else:
                raise ValueError("hub MST edge is invalid")
        else:
            raise ValueError("private graph contains a non-hub edge")
    if any(
        anchored_members[component.hub_id] != set(component.member_ids)
        for component in components
    ):
        raise ValueError("each private hub must anchor every component member")
    if hub_edge_count != max(0, len(hub_ids) - 1):
        raise ValueError("private hub graph must contain a hub spanning tree")

    adjacency: dict[str, set[str]] = {
        node_id: set() for node_id in tuple(node_ids) + tuple(sorted(hub_ids))
    }
    for edge in tuple(cocktail_edges) + private_hub_edges:
        adjacency[edge.source_id].add(edge.target_id)
        adjacency[edge.target_id].add(edge.source_id)
    visited: set[str] = set()
    stack = [min(adjacency)]
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        stack.extend(sorted(adjacency[node_id] - visited, reverse=True))
    if visited != set(adjacency):
        raise ValueError("private hub topology did not connect the augmented graph")


def layout_spherical_graph(
    *,
    node_ids: Sequence[str],
    cocktail_edges: Sequence[GraphEdge],
    components: Sequence[ComponentSummary],
    private_hub_edges: Sequence[GraphEdge],
    config: SphericalGraphConfig,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, Any]]:
    """Optimize fixed graph forces with no vector or similarity-matrix input.

    This is the S² layout entrypoint. It returns public cocktail coordinates and
    non-identifying diagnostics; private hub coordinates are discarded.
    """

    _validate_config(config)
    fixed_node_ids = _coerce_node_ids(node_ids)
    fixed_edges = tuple(cocktail_edges)
    fixed_components = _validate_components(fixed_node_ids, components)
    fixed_hub_edges = tuple(private_hub_edges)
    _validate_weighted_union_graph(fixed_node_ids, fixed_edges)
    # This also proves the caller supplied a connected fixed private graph.
    _validate_private_hub_graph(
        node_ids=fixed_node_ids,
        components=fixed_components,
        private_hub_edges=fixed_hub_edges,
        cocktail_edges=fixed_edges,
    )

    hub_ids = tuple(component.hub_id for component in fixed_components)
    all_ids = fixed_node_ids + hub_ids
    index = {node_id: position for position, node_id in enumerate(all_ids)}
    cocktail_count = len(fixed_node_ids)

    union_pairs = _edge_pairs(fixed_edges, index)
    union_targets = np.asarray(
        [max(-1.0, min(1.0, edge.similarity)) for edge in fixed_edges],
        dtype=np.float64,
    )
    private_pairs = _edge_pairs(fixed_hub_edges, index)
    private_targets = np.asarray(
        [max(-1.0, min(1.0, edge.similarity)) for edge in fixed_hub_edges],
        dtype=np.float64,
    )
    edge_neighbors = _edge_neighbor_sets(cocktail_count, union_pairs)
    nonedge_pairs, nonedges_by_node = _sample_graph_nonedges(
        fixed_node_ids,
        edge_neighbors,
        seed=config.seed,
        per_node=config.nonedge_samples_per_node,
    )
    ranking_triples = _graph_ranking_triples(
        edge_neighbors,
        nonedges_by_node,
        per_edge=config.ranking_nonedges_per_edge,
    )
    multistart_seeds = tuple(
        (config.seed + 104729 * start) % (2**63 - 1)
        for start in range(config.multistart_count)
    )

    best: np.ndarray | None = None
    best_key: tuple[float, int] | None = None
    candidate_objectives: list[float] = []
    for start, start_seed in enumerate(multistart_seeds):
        initial = _cluster_local_initialization(
            node_ids=fixed_node_ids,
            components=fixed_components,
            start_seed=start_seed,
            start_index=start,
            cluster_radius=config.cluster_initial_radius,
        )
        coordinates = initial.copy()
        for iteration in range(config.layout_iterations):
            gradient = np.zeros_like(coordinates)
            _add_edge_gradient(
                gradient,
                coordinates,
                union_pairs,
                union_targets,
                config.edge_weight,
            )
            _add_edge_gradient(
                gradient,
                coordinates,
                private_pairs,
                private_targets,
                config.hub_edge_weight,
            )
            _add_nonedge_gradient(
                gradient,
                coordinates,
                nonedge_pairs,
                margin=config.nonedge_margin,
                weight=config.nonedge_repulsion_weight,
            )
            _add_ranking_gradient(
                gradient,
                coordinates,
                ranking_triples,
                margin=config.ranking_margin,
                weight=config.ranking_weight,
            )
            gradient += 2.0 * config.initialization_weight * (coordinates - initial)
            gradient -= (
                np.sum(gradient * coordinates, axis=1, keepdims=True) * coordinates
            )
            norms = np.linalg.norm(gradient, axis=1, keepdims=True)
            gradient /= np.maximum(1.0, norms / 8.0)
            progress = iteration / max(1, config.layout_iterations - 1)
            rate = config.learning_rate * (1.0 - 0.82 * progress)
            coordinates = _normalize_rows_3d(coordinates - rate * gradient)

        _project_graph_nonedge_constraints(
            coordinates,
            nonedge_pairs,
            margin=config.nonedge_margin + config.constraint_projection_slack,
            passes=config.constraint_projection_passes,
        )
        objective = _graph_only_objective(
            coordinates,
            union_pairs,
            union_targets,
            private_pairs,
            private_targets,
            ranking_triples,
            config,
        )
        candidate_objectives.append(objective)
        key = (objective, start)
        if best_key is None or key < best_key:
            best_key = key
            best = coordinates.copy()

    assert best is not None and best_key is not None
    coordinates = best
    coordinate_map = {
        node_id: (
            float(coordinates[position, 0]),
            float(coordinates[position, 1]),
            float(coordinates[position, 2]),
        )
        for position, node_id in enumerate(fixed_node_ids)
    }
    return coordinate_map, {
        "algorithm": "graph_only_spherical_force_v3",
        "iterations": config.layout_iterations,
        "multistart_count": config.multistart_count,
        "multistart_seeds": list(multistart_seeds),
        "selected_start": best_key[1],
        "candidate_objectives": candidate_objectives,
        "private_hub_count": len(hub_ids),
        "private_hub_edge_count": len(fixed_hub_edges),
        "sampled_nonedge_count": len(nonedge_pairs),
        "ranking_constraint_count": len(ranking_triples),
        "edge_target_policy": "acos(clamped edge cosine), radians",
        "negative_sampling_policy": (
            "deterministic graph nonedges only; no similarity data"
        ),
    }


def _validate_weighted_union_graph(
    node_ids: tuple[str, ...],
    edges: tuple[GraphEdge, ...],
) -> None:
    known = set(node_ids)
    seen: set[frozenset[str]] = set()
    for edge in edges:
        pair = frozenset((edge.source_id, edge.target_id))
        if (
            edge.edge_kind != "cocktail_knn"
            or edge.source_id not in known
            or edge.target_id not in known
            or edge.source_id == edge.target_id
            or pair in seen
            or not edge.visible
            or not edge.recommendable
            or edge.is_bridge
            or not math.isfinite(edge.similarity)
            or not math.isclose(
                edge.target_distance,
                similarity_to_target_distance(edge.similarity),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("weighted cocktail union edge is invalid")
        seen.add(pair)


def _edge_pairs(
    edges: tuple[GraphEdge, ...],
    index: dict[str, int],
) -> np.ndarray:
    return np.asarray(
        [(index[edge.source_id], index[edge.target_id]) for edge in edges],
        dtype=np.int64,
    ).reshape((-1, 2))


def _edge_neighbor_sets(
    node_count: int,
    pairs: np.ndarray,
) -> dict[int, set[int]]:
    neighbors: dict[int, set[int]] = {position: set() for position in range(node_count)}
    for raw_left, raw_right in pairs:
        left = int(raw_left)
        right = int(raw_right)
        neighbors[left].add(right)
        neighbors[right].add(left)
    return neighbors


def _sample_graph_nonedges(
    node_ids: tuple[str, ...],
    edge_neighbors: dict[int, set[int]],
    *,
    seed: int,
    per_node: int,
) -> tuple[np.ndarray, dict[int, tuple[int, ...]]]:
    selected: set[tuple[int, int]] = set()
    by_node: dict[int, tuple[int, ...]] = {}
    for source, source_id in enumerate(node_ids):
        candidates = [
            target
            for target in range(len(node_ids))
            if target != source and target not in edge_neighbors[source]
        ]
        candidates.sort(
            key=lambda target: (
                _stable_pair_key(seed, source_id, node_ids[target]),
                node_ids[target],
            )
        )
        chosen = tuple(candidates[:per_node])
        by_node[source] = chosen
        for target in chosen:
            selected.add((source, target) if source < target else (target, source))
    return (
        np.asarray(sorted(selected), dtype=np.int64).reshape((-1, 2)),
        by_node,
    )


def _stable_pair_key(seed: int, left: str, right: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{left}\0{right}".encode("utf-8")).digest()


def _graph_ranking_triples(
    edge_neighbors: dict[int, set[int]],
    nonedges_by_node: dict[int, tuple[int, ...]],
    *,
    per_edge: int,
) -> np.ndarray:
    triples = [
        (center, near, far)
        for center in sorted(edge_neighbors)
        for near in sorted(edge_neighbors[center])
        for far in nonedges_by_node[center][:per_edge]
    ]
    return np.asarray(triples, dtype=np.int64).reshape((-1, 3))


def _cluster_local_initialization(
    *,
    node_ids: tuple[str, ...],
    components: tuple[ComponentSummary, ...],
    start_seed: int,
    start_index: int,
    cluster_radius: float,
) -> np.ndarray:
    node_index = {node_id: position for position, node_id in enumerate(node_ids)}
    coordinates = np.zeros((len(node_ids) + len(components), 3), dtype=np.float64)
    rng = np.random.default_rng(start_seed)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for component_index, component in enumerate(components):
        center = _fibonacci_point(component_index, len(components))
        coordinates[len(node_ids) + component_index] = center
        tangent_u, tangent_v = _tangent_basis(center)
        phase = float(rng.uniform(0.0, 2.0 * math.pi)) + start_index * golden_angle
        member_count = len(component.member_ids)
        for member_position, node_id in enumerate(component.member_ids):
            radius = cluster_radius * (
                0.35 + 0.65 * math.sqrt((member_position + 0.5) / member_count)
            )
            angle = phase + golden_angle * member_position
            tangent = math.cos(angle) * tangent_u + math.sin(angle) * tangent_v
            point = math.cos(radius) * center + math.sin(radius) * tangent
            coordinates[node_index[node_id]] = point / np.linalg.norm(point)
    return coordinates


def _fibonacci_point(index: int, total: int) -> np.ndarray:
    if total <= 1:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    y = 1.0 - 2.0 * index / (total - 1)
    radius = math.sqrt(max(0.0, 1.0 - y * y))
    angle = math.pi * (3.0 - math.sqrt(5.0)) * index
    return np.asarray(
        [math.cos(angle) * radius, y, math.sin(angle) * radius],
        dtype=np.float64,
    )


def _tangent_basis(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.zeros(3, dtype=np.float64)
    axis[int(np.argmin(np.abs(center)))] = 1.0
    tangent_u = axis - float(np.dot(axis, center)) * center
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(center, tangent_u)
    tangent_v /= np.linalg.norm(tangent_v)
    return tangent_u, tangent_v


def _add_edge_gradient(
    gradient: np.ndarray,
    coordinates: np.ndarray,
    pairs: np.ndarray,
    targets: np.ndarray,
    weight: float,
) -> None:
    if not len(pairs):
        return
    left = pairs[:, 0]
    right = pairs[:, 1]
    dots = np.sum(coordinates[left] * coordinates[right], axis=1)
    coefficient = 2.0 * weight * (dots - targets)
    np.add.at(gradient, left, coefficient[:, None] * coordinates[right])
    np.add.at(gradient, right, coefficient[:, None] * coordinates[left])


def _add_nonedge_gradient(
    gradient: np.ndarray,
    coordinates: np.ndarray,
    pairs: np.ndarray,
    *,
    margin: float,
    weight: float,
) -> None:
    if not len(pairs):
        return
    left = pairs[:, 0]
    right = pairs[:, 1]
    dots = np.sum(coordinates[left] * coordinates[right], axis=1)
    maximum = math.cos(margin)
    active = dots > maximum
    if not np.any(active):
        return
    active_left = left[active]
    active_right = right[active]
    coefficient = 2.0 * weight * (dots[active] - maximum)
    np.add.at(
        gradient,
        active_left,
        coefficient[:, None] * coordinates[active_right],
    )
    np.add.at(
        gradient,
        active_right,
        coefficient[:, None] * coordinates[active_left],
    )


def _add_ranking_gradient(
    gradient: np.ndarray,
    coordinates: np.ndarray,
    triples: np.ndarray,
    *,
    margin: float,
    weight: float,
) -> None:
    if not len(triples):
        return
    centers = triples[:, 0]
    near = triples[:, 1]
    far = triples[:, 2]
    near_dot = np.sum(coordinates[centers] * coordinates[near], axis=1)
    far_dot = np.sum(coordinates[centers] * coordinates[far], axis=1)
    hinge = margin + far_dot - near_dot
    active = hinge > 0.0
    if not np.any(active):
        return
    coefficient = 2.0 * weight * hinge[active]
    active_centers = centers[active]
    active_near = near[active]
    active_far = far[active]
    np.add.at(
        gradient,
        active_centers,
        coefficient[:, None] * (coordinates[active_far] - coordinates[active_near]),
    )
    np.add.at(
        gradient,
        active_near,
        -coefficient[:, None] * coordinates[active_centers],
    )
    np.add.at(
        gradient,
        active_far,
        coefficient[:, None] * coordinates[active_centers],
    )


def _normalize_rows_3d(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms <= 1e-15):
        raise ValueError("force layout produced a zero coordinate")
    return cast(np.ndarray, result / norms[:, None])


def _project_graph_nonedge_constraints(
    coordinates: np.ndarray,
    pairs: np.ndarray,
    *,
    margin: float,
    passes: int,
) -> None:
    for _ in range(passes):
        for raw_left, raw_right in pairs:
            left = int(raw_left)
            right = int(raw_right)
            dot = float(
                np.clip(np.dot(coordinates[left], coordinates[right]), -1.0, 1.0)
            )
            angle = math.acos(dot)
            if angle >= margin:
                continue
            if angle <= 1e-10:
                left_tangent, _ = _tangent_basis(coordinates[left])
                right_tangent = -left_tangent
            else:
                sine = math.sin(angle)
                left_tangent = (coordinates[right] - dot * coordinates[left]) / sine
                right_tangent = (coordinates[left] - dot * coordinates[right]) / sine
            half = 0.5 * (margin - angle)
            cosine = math.cos(half)
            sine = math.sin(half)
            new_left = cosine * coordinates[left] - sine * left_tangent
            new_right = cosine * coordinates[right] - sine * right_tangent
            coordinates[left] = new_left / np.linalg.norm(new_left)
            coordinates[right] = new_right / np.linalg.norm(new_right)


def _graph_only_objective(
    coordinates: np.ndarray,
    union_pairs: np.ndarray,
    union_targets: np.ndarray,
    private_pairs: np.ndarray,
    private_targets: np.ndarray,
    ranking_triples: np.ndarray,
    config: SphericalGraphConfig,
) -> float:
    union_rmse = _angular_rmse(coordinates, union_pairs, union_targets)
    private_rmse = _angular_rmse(coordinates, private_pairs, private_targets)
    ranking_fraction = _count_ranking_violations(
        coordinates,
        ranking_triples,
        config.ranking_margin,
    ) / max(1, len(ranking_triples))
    return union_rmse + 0.15 * private_rmse + 0.10 * ranking_fraction


def _angular_rmse(
    coordinates: np.ndarray,
    pairs: np.ndarray,
    target_cosines: np.ndarray,
) -> float:
    if not len(pairs):
        return 0.0
    actual = _pair_angles(coordinates, pairs)
    targets = np.arccos(np.clip(target_cosines, -1.0, 1.0))
    return float(np.sqrt(np.mean((actual - targets) ** 2)))


def _pair_angles(coordinates: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if not len(pairs):
        return np.asarray([], dtype=np.float64)
    dots = np.sum(
        coordinates[pairs[:, 0]] * coordinates[pairs[:, 1]],
        axis=1,
    )
    return cast(np.ndarray, np.arccos(np.clip(dots, -1.0, 1.0)))


def _count_ranking_violations(
    coordinates: np.ndarray,
    triples: np.ndarray,
    margin: float,
) -> int:
    if not len(triples):
        return 0
    centers = triples[:, 0]
    near = triples[:, 1]
    far = triples[:, 2]
    near_dot = np.sum(coordinates[centers] * coordinates[near], axis=1)
    far_dot = np.sum(coordinates[centers] * coordinates[far], axis=1)
    return int(np.sum(near_dot + 1e-12 < far_dot + margin))


def _public_graph_nodes(
    *,
    node_ids: tuple[str, ...],
    coordinates: dict[str, tuple[float, float, float]],
    components: tuple[ComponentSummary, ...],
) -> tuple[GraphNode, ...]:
    component_by_member = {
        member: component.component_id
        for component in components
        for member in component.member_ids
    }
    return tuple(
        GraphNode(
            node_id=node_id,
            node_kind="cocktail",
            component_id=component_by_member[node_id],
            x=coordinates[node_id][0],
            y=coordinates[node_id][1],
            z=coordinates[node_id][2],
            visible=True,
            recommendable=True,
        )
        for node_id in node_ids
    )


def _evaluate_layout(
    *,
    node_ids: tuple[str, ...],
    coordinate_map: dict[str, tuple[float, float, float]],
    directed_neighbors: tuple[DirectedNeighbor, ...],
    cocktail_edges: tuple[GraphEdge, ...],
    audit_similarities: np.ndarray | Mapping[tuple[str, str], float] | None,
    audit_node_ids: Sequence[str] | None,
    clustering_policy: str,
    layout_diagnostics: dict[str, Any],
    config: SphericalGraphConfig,
) -> dict[str, Any]:
    coordinates = np.asarray([coordinate_map[node_id] for node_id in node_ids])
    coordinate_top5 = _coordinate_neighbors(node_ids, coordinates, k=5)
    true_by_source: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for row in directed_neighbors:
        if row.rank <= 5:
            true_by_source[row.source_id].add(row.target_id)
    recalls = [
        len(true_by_source[node_id] & set(coordinate_top5[node_id]))
        / max(1, min(5, len(true_by_source[node_id])))
        for node_id in node_ids
    ]
    mean_recall = float(math.fsum(recalls) / len(recalls))
    coverage = float(sum(recall > 0.0 for recall in recalls) / len(recalls))

    index = {node_id: position for position, node_id in enumerate(node_ids)}
    edge_pairs = _edge_pairs(cocktail_edges, index)
    edge_targets = np.asarray(
        [max(-1.0, min(1.0, edge.similarity)) for edge in cocktail_edges]
    )
    edge_rmse = _angular_rmse(coordinates, edge_pairs, edge_targets)
    norm_error = float(np.max(np.abs(np.linalg.norm(coordinates, axis=1) - 1.0)))
    false_close_count = _bottom_decile_false_close_count(
        node_ids=node_ids,
        coordinates=coordinates,
        directed_neighbors=directed_neighbors,
        cocktail_edges=cocktail_edges,
        audit_similarities=audit_similarities,
        audit_node_ids=audit_node_ids,
    )
    checks: dict[str, bool | None] = {
        "mean_recall_at_5": mean_recall >= _MEAN_RECALL_AT_5_MIN,
        "node_coverage_at_5": coverage >= _NODE_COVERAGE_AT_5_MIN,
        "bottom_decile_false_close_count": (
            None
            if false_close_count is None
            else false_close_count <= _BOTTOM_DECILE_FALSE_CLOSE_MAX
        ),
        "union_edge_rmse_radians": edge_rmse <= _UNION_EDGE_RMSE_MAX,
        "unit_norm_max_error": norm_error <= _UNIT_NORM_MAX_ERROR,
    }
    evaluated_checks = [value for value in checks.values() if value is not None]
    report = dict(layout_diagnostics)
    report.update(
        {
            "seed": config.seed,
            "clustering_policy": clustering_policy,
            "mean_recall_at_5": mean_recall,
            "node_coverage_at_5": coverage,
            "nodes_with_true_top5_count": sum(recall > 0.0 for recall in recalls),
            "bottom_decile_false_close_count": false_close_count,
            "bottom_decile_false_close_policy": (
                "per-source cosine-bottom-decile graph nonneighbors closer than "
                "that source's farthest true top-5 coordinate neighbor"
            ),
            "audit_similarity_supplied": audit_similarities is not None,
            "union_edge_rmse_radians": edge_rmse,
            # Compatibility spelling used by the v1 report.
            "edge_target_rmse_radians": edge_rmse,
            "unit_norm_max_error": norm_error,
            "coordinate_sha256": _coordinate_hash(node_ids, coordinate_map),
            "acceptance_thresholds": {
                "mean_recall_at_5_min": _MEAN_RECALL_AT_5_MIN,
                "node_coverage_at_5_min": _NODE_COVERAGE_AT_5_MIN,
                "bottom_decile_false_close_count_max": (_BOTTOM_DECILE_FALSE_CLOSE_MAX),
                "union_edge_rmse_radians_max": _UNION_EDGE_RMSE_MAX,
                "unit_norm_max_error_max": _UNIT_NORM_MAX_ERROR,
            },
            "acceptance_checks": checks,
            "acceptance_passed": all(evaluated_checks),
            "report_only": config.report_only,
        }
    )
    return report


def _coordinate_neighbors(
    node_ids: tuple[str, ...],
    coordinates: np.ndarray,
    *,
    k: int,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    similarities = coordinates @ coordinates.T
    for source, source_id in enumerate(node_ids):
        candidates = [target for target in range(len(node_ids)) if target != source]
        candidates.sort(
            key=lambda target: (
                -float(similarities[source, target]),
                _node_id_key(node_ids[target]),
            )
        )
        result[source_id] = tuple(
            node_ids[target] for target in candidates[: min(k, len(candidates))]
        )
    return result


def _bottom_decile_false_close_count(
    *,
    node_ids: tuple[str, ...],
    coordinates: np.ndarray,
    directed_neighbors: tuple[DirectedNeighbor, ...],
    cocktail_edges: tuple[GraphEdge, ...],
    audit_similarities: np.ndarray | Mapping[tuple[str, str], float] | None,
    audit_node_ids: Sequence[str] | None,
) -> int | None:
    if audit_similarities is None:
        return None
    audit = _audit_matrix_in_node_order(
        node_ids=node_ids,
        audit_similarities=audit_similarities,
        audit_node_ids=audit_node_ids,
    )
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    union_neighbors: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in cocktail_edges:
        union_neighbors[edge.source_id].add(edge.target_id)
        union_neighbors[edge.target_id].add(edge.source_id)
    true_by_source: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for row in directed_neighbors:
        if row.rank <= 5:
            true_by_source[row.source_id].append(row.target_id)

    violation_count = 0
    for source_id in node_ids:
        source = index[source_id]
        nonneighbors = [
            target_id
            for target_id in node_ids
            if target_id != source_id and target_id not in union_neighbors[source_id]
        ]
        if not nonneighbors or not true_by_source[source_id]:
            continue
        nonneighbors.sort(
            key=lambda target_id: (
                float(audit[source, index[target_id]]),
                _node_id_key(target_id),
            )
        )
        bottom_count = max(1, math.ceil(len(nonneighbors) * 0.10))
        farthest_true_angle = max(
            _coordinate_angle(coordinates, source, index[target_id])
            for target_id in true_by_source[source_id]
        )
        violation_count += sum(
            _coordinate_angle(coordinates, source, index[target_id]) + 1e-12
            < farthest_true_angle
            for target_id in nonneighbors[:bottom_count]
        )
    return violation_count


def _coordinate_angle(
    coordinates: np.ndarray,
    left: int,
    right: int,
) -> float:
    return math.acos(
        float(
            np.clip(
                np.dot(coordinates[left], coordinates[right]),
                -1.0,
                1.0,
            )
        )
    )


def _audit_matrix_in_node_order(
    *,
    node_ids: tuple[str, ...],
    audit_similarities: np.ndarray | Mapping[tuple[str, str], float],
    audit_node_ids: Sequence[str] | None,
) -> np.ndarray:
    if isinstance(audit_similarities, Mapping):
        matrix = np.eye(len(node_ids), dtype=np.float64)
        for left_index, left in enumerate(node_ids):
            for right_index, right in enumerate(
                node_ids[left_index + 1 :],
                start=left_index + 1,
            ):
                pair = (left, right)
                reverse = (right, left)
                if pair in audit_similarities:
                    value = float(audit_similarities[pair])
                elif reverse in audit_similarities:
                    value = float(audit_similarities[reverse])
                else:
                    raise ValueError("audit similarity mapping is not exhaustive")
                if not math.isfinite(value):
                    raise ValueError("audit similarities must be finite")
                matrix[left_index, right_index] = value
                matrix[right_index, left_index] = value
        return matrix

    matrix = np.asarray(audit_similarities, dtype=np.float64)
    supplied_ids = (
        tuple(str(node_id) for node_id in audit_node_ids)
        if audit_node_ids is not None
        else node_ids
    )
    if len(set(supplied_ids)) != len(supplied_ids) or set(supplied_ids) != set(
        node_ids
    ):
        raise ValueError("audit_node_ids must match cocktail node IDs")
    if matrix.shape != (len(supplied_ids), len(supplied_ids)):
        raise ValueError("audit similarity matrix shape is invalid")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("audit similarities must be finite")
    source_index = {node_id: position for position, node_id in enumerate(supplied_ids)}
    order = [source_index[node_id] for node_id in node_ids]
    return cast(np.ndarray, matrix[np.ix_(order, order)])


def _coordinate_hash(
    node_ids: tuple[str, ...],
    coordinate_map: dict[str, tuple[float, float, float]],
) -> str:
    rows = [
        [
            node_id,
            *[float(value).hex() for value in coordinate_map[node_id]],
        ]
        for node_id in node_ids
    ]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _enforce_quality(report: dict[str, Any], *, report_only: bool) -> None:
    if report["acceptance_passed"] or report_only:
        return
    failed = [
        name for name, passed in report["acceptance_checks"].items() if passed is False
    ]
    raise SphericalLayoutQualityError(
        "S² layout missed acceptance thresholds: "
        + ", ".join(failed)
        + "; set report_only=True to inspect the complete report"
    )
