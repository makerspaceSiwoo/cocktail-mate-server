"""Injectable high-dimensional clustering boundary."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from app.spherical_graph.models import GraphEdge


def _node_id_key(node_id: str) -> tuple[int, int, str]:
    try:
        numeric_id = int(node_id)
    except ValueError:
        return (1, 0, node_id)
    if numeric_id > 0 and str(numeric_id) == node_id:
        return (0, numeric_id, "")
    return (1, 0, node_id)


@runtime_checkable
class HighDimensionalClusterer(Protocol):
    """Cluster from exact high-dimensional similarity and union-graph topology."""

    @property
    def clusterer_id(self) -> str:
        """Stable read-only identifier included in graph provenance."""
        ...

    def clusters(
        self,
        node_ids: Sequence[str],
        similarities: np.ndarray,
        union_edges: Sequence[GraphEdge],
    ) -> Sequence[Sequence[str]]:
        """Return a deterministic, exhaustive partition of ``node_ids``."""


class UnionGraphComponentClusterer:
    """Default policy: every union-graph connected component is one cluster."""

    clusterer_id = "union_graph_connected_components_v1"

    def clusters(
        self,
        node_ids: Sequence[str],
        similarities: np.ndarray,
        union_edges: Sequence[GraphEdge],
    ) -> tuple[tuple[str, ...], ...]:
        del similarities
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
        for edge in union_edges:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)
        remaining = set(node_ids)
        components: list[tuple[str, ...]] = []
        while remaining:
            start = min(remaining, key=_node_id_key)
            stack = [start]
            members: set[str] = set()
            while stack:
                node_id = stack.pop()
                if node_id in members:
                    continue
                members.add(node_id)
                stack.extend(
                    sorted(
                        adjacency[node_id] - members,
                        key=_node_id_key,
                        reverse=True,
                    )
                )
            remaining.difference_update(members)
            components.append(tuple(sorted(members, key=_node_id_key)))
        return tuple(sorted(components, key=lambda members: _node_id_key(members[0])))


@dataclass(frozen=True, slots=True)
class CosineKMedoidsClusterer:
    """Deterministic k-medoids over a supplied cosine-similarity matrix.

    A CLI can map ``--clusters N`` directly to
    ``CosineKMedoidsClusterer(cluster_count=N)`` and pass the instance to
    :func:`app.spherical_graph.build_spherical_graph` as ``clusterer=...``.
    The implementation never reads vectors, graph layout, or external state.
    """

    cluster_count: int = 7
    max_iterations: int = 100
    seed: int = 20260806

    def __post_init__(self) -> None:
        if type(self.cluster_count) is not int or self.cluster_count <= 0:
            raise ValueError("cluster_count must be a positive integer")
        if type(self.max_iterations) is not int or self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")

    @property
    def clusterer_id(self) -> str:
        return (
            "cosine_k_medoids_v1"
            f":k={self.cluster_count}"
            f":iterations={self.max_iterations}"
            f":seed={self.seed}"
        )

    def clusters(
        self,
        node_ids: Sequence[str],
        similarities: np.ndarray,
        union_edges: Sequence[GraphEdge],
    ) -> tuple[tuple[str, ...], ...]:
        del union_edges
        original_ids = tuple(node_ids)
        if not original_ids or len(set(original_ids)) != len(original_ids):
            raise ValueError("k-medoids node IDs must be non-empty and unique")
        if self.cluster_count > len(original_ids):
            raise ValueError("cluster_count cannot exceed the node count")

        matrix = np.asarray(similarities, dtype=np.float64)
        if matrix.shape != (len(original_ids), len(original_ids)):
            raise ValueError("k-medoids similarity matrix shape is invalid")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("k-medoids similarity matrix must be finite")
        if not np.allclose(matrix, matrix.T, atol=1e-9, rtol=0.0):
            raise ValueError("k-medoids similarity matrix must be symmetric")
        if np.any(matrix < -1.0 - 1e-9) or np.any(matrix > 1.0 + 1e-9):
            raise ValueError("k-medoids cosine similarity must be within [-1, 1]")

        order = tuple(
            sorted(
                range(len(original_ids)),
                key=lambda index: _node_id_key(original_ids[index]),
            )
        )
        ids = tuple(original_ids[index] for index in order)
        ordered = np.clip(matrix[np.ix_(order, order)], -1.0, 1.0)
        distances = np.clip(1.0 - ordered, 0.0, 2.0)
        medoids = self._initial_medoids(ids, distances)

        for _ in range(self.max_iterations):
            medoids, assignments = self._assign_with_empty_reseed(
                ids,
                distances,
                medoids,
            )
            updated = tuple(
                self._best_medoid(ids, distances, members) for members in assignments
            )
            if updated == medoids:
                break
            medoids = updated

        medoids, assignments = self._assign_with_empty_reseed(
            ids,
            distances,
            medoids,
        )
        del medoids
        clusters = [
            tuple(
                ids[index]
                for index in sorted(
                    members,
                    key=lambda index: _node_id_key(ids[index]),
                )
            )
            for members in assignments
        ]
        return tuple(sorted(clusters, key=lambda members: _node_id_key(members[0])))

    def _initial_medoids(
        self,
        ids: tuple[str, ...],
        distances: np.ndarray,
    ) -> tuple[int, ...]:
        first = random.Random(self.seed).randrange(len(ids))
        medoids = [first]
        while len(medoids) < self.cluster_count:
            candidates = [index for index in range(len(ids)) if index not in medoids]
            candidates.sort(
                key=lambda index: (
                    -min(float(distances[index, medoid]) for medoid in medoids),
                    _node_id_key(ids[index]),
                )
            )
            medoids.append(candidates[0])
        return tuple(medoids)

    def _assign_with_empty_reseed(
        self,
        ids: tuple[str, ...],
        distances: np.ndarray,
        medoids: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        current = medoids
        for _ in range(self.cluster_count + 1):
            assignments = self._assign(ids, distances, current)
            empty = [
                cluster_index
                for cluster_index, members in enumerate(assignments)
                if not members
            ]
            if not empty:
                return current, assignments

            occupied = set(current)
            for cluster_index in empty:
                candidates = [
                    node_index
                    for assigned_cluster, members in enumerate(assignments)
                    if len(members) > 1
                    for node_index in members
                    if node_index not in occupied
                    and node_index != current[assigned_cluster]
                ]
                if not candidates:
                    raise ValueError("k-medoids could not reseed an empty cluster")
                candidates.sort(
                    key=lambda node_index: (
                        -min(
                            float(distances[node_index, medoid]) for medoid in current
                        ),
                        _node_id_key(ids[node_index]),
                    )
                )
                replacement = candidates[0]
                mutable = list(current)
                mutable[cluster_index] = replacement
                current = tuple(mutable)
                occupied.add(replacement)
        raise ValueError("k-medoids empty-cluster reseed did not converge")

    @staticmethod
    def _assign(
        ids: tuple[str, ...],
        distances: np.ndarray,
        medoids: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        medoid_cluster = {
            medoid: cluster_index for cluster_index, medoid in enumerate(medoids)
        }
        assignments: list[list[int]] = [[] for _ in medoids]
        for node_index in range(len(ids)):
            if node_index in medoid_cluster:
                cluster_index = medoid_cluster[node_index]
            else:
                cluster_index = min(
                    range(len(medoids)),
                    key=lambda candidate: (
                        float(distances[node_index, medoids[candidate]]),
                        _node_id_key(ids[medoids[candidate]]),
                    ),
                )
            assignments[cluster_index].append(node_index)
        return tuple(tuple(members) for members in assignments)

    @staticmethod
    def _best_medoid(
        ids: tuple[str, ...],
        distances: np.ndarray,
        members: tuple[int, ...],
    ) -> int:
        if not members:
            raise ValueError("k-medoids cannot update an empty cluster")
        return min(
            members,
            key=lambda candidate: (
                float(np.sum(distances[candidate, list(members)])),
                _node_id_key(ids[candidate]),
            ),
        )
