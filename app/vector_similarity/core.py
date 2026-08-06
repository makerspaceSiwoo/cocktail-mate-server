"""Dependency-free exact cosine baseline and deterministic neighbor graph."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence

from .types import (
    AnnBackendUnavailableError,
    CANONICAL_NEIGHBOR_SCHEMA_VERSION,
    EXACT_COSINE_PROVIDER_ID,
    CanonicalNeighborArtifact,
    DenseVector,
    InnerProductMatch,
    LegacyDirectedNeighbor,
    LegacyUndirectedNeighborEdge,
    NeighborSearchBackend,
    PairSimilarity,
    SimilarityConfigurationError,
    VectorRecord,
    VectorValidationError,
)


def _finite_values(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    if not values:
        raise VectorValidationError(f"{label} vector must not be empty")
    result: list[float] = []
    for value in values:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise VectorValidationError(f"{label} vector must contain finite numbers")
        result.append(float(value))
    return tuple(result)


def _same_dimension(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    left_values = _finite_values(left, label="left")
    right_values = _finite_values(right, label="right")
    if len(left_values) != len(right_values):
        raise VectorValidationError("vector dimensions must match")
    return left_values, right_values


def negative_inner_product(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Return the lower-is-better distance convention used by inner-product ANN."""

    left_values, right_values = _same_dimension(left, right)
    try:
        distance = -math.fsum(
            left_value * right_value
            for left_value, right_value in zip(
                left_values,
                right_values,
                strict=True,
            )
        )
    except (OverflowError, ValueError) as exc:
        raise VectorValidationError("inner product is not finite") from exc
    if not math.isfinite(distance):
        raise VectorValidationError("inner product is not finite")
    return distance


def positive_score_from_distance(distance: float) -> float:
    """Convert negative-inner-product distance to a higher-is-better score."""

    if (
        not isinstance(distance, (int, float))
        or isinstance(distance, bool)
        or not math.isfinite(float(distance))
    ):
        raise VectorValidationError("distance must be a finite number")
    return -float(distance)


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Compute exact cosine; zero vectors fail because cosine is undefined."""

    left_values, right_values = _same_dimension(left, right)
    left_scale = max(abs(value) for value in left_values)
    right_scale = max(abs(value) for value in right_values)
    if left_scale == 0.0 or right_scale == 0.0:
        raise VectorValidationError("cosine similarity is undefined for a zero vector")
    left_scaled = tuple(value / left_scale for value in left_values)
    right_scaled = tuple(value / right_scale for value in right_values)
    left_norm = math.sqrt(math.fsum(value * value for value in left_scaled))
    right_norm = math.sqrt(math.fsum(value * value for value in right_scaled))
    score = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(
            left_scaled,
            right_scaled,
            strict=True,
        )
    ) / (left_norm * right_norm)
    # Clamp only floating-point roundoff beyond the mathematical cosine range.
    return min(1.0, max(-1.0, score))


def validate_records(
    records: Sequence[VectorRecord],
) -> tuple[DenseVector, ...]:
    """Materialize, validate, de-duplicate, and deterministically order records."""

    materialized: list[DenseVector] = []
    seen_ids: set[int] = set()
    dimension: int | None = None
    for record in records:
        vector = DenseVector.from_values(record.id, record.values)
        if vector.id in seen_ids:
            raise VectorValidationError(f"duplicate vector id: {vector.id}")
        seen_ids.add(vector.id)
        if dimension is None:
            dimension = len(vector.values)
        elif len(vector.values) != dimension:
            raise VectorValidationError("all vector dimensions must match")
        if not any(value != 0.0 for value in vector.values):
            raise VectorValidationError(
                f"cosine similarity is undefined for zero vector id {vector.id}"
            )
        materialized.append(vector)
    return tuple(sorted(materialized, key=lambda vector: vector.id))


def _validate_dense_records(
    records: Sequence[VectorRecord],
) -> tuple[DenseVector, ...]:
    materialized: list[DenseVector] = []
    seen_ids: set[int] = set()
    dimension: int | None = None
    for record in records:
        vector = DenseVector.from_values(record.id, record.values)
        if vector.id in seen_ids:
            raise VectorValidationError(f"duplicate vector id: {vector.id}")
        seen_ids.add(vector.id)
        if dimension is None:
            dimension = len(vector.values)
        elif len(vector.values) != dimension:
            raise VectorValidationError("all vector dimensions must match")
        materialized.append(vector)
    return tuple(sorted(materialized, key=lambda vector: vector.id))


def rank_by_inner_product(
    query: Sequence[float],
    records: Sequence[VectorRecord],
    *,
    k: int,
) -> tuple[InnerProductMatch, ...]:
    """Rank raw vectors like pgvector ``<#>`` and expose higher-is-better score."""

    _validate_k(k)
    query_values = _finite_values(query, label="query")
    vectors = _validate_dense_records(records)
    candidates: list[tuple[float, int]] = []
    for vector in vectors:
        if len(vector.values) != len(query_values):
            raise VectorValidationError("query and record dimensions must match")
        distance = negative_inner_product(query_values, vector.values)
        candidates.append((distance, vector.id))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return tuple(
        InnerProductMatch(
            id=id,
            rank=rank,
            negative_inner_product=distance,
            score=positive_score_from_distance(distance),
        )
        for rank, (distance, id) in enumerate(candidates[:k], start=1)
    )


def all_pair_similarities(
    records: Sequence[VectorRecord],
) -> tuple[PairSimilarity, ...]:
    """Return every unordered exact-cosine pair in identifier order."""

    vectors = validate_records(records)
    pairs: list[PairSimilarity] = []
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1 :]:
            score = cosine_similarity(left.values, right.values)
            pairs.append(
                PairSimilarity(
                    left_id=left.id,
                    right_id=right.id,
                    score=score,
                )
            )
    return tuple(pairs)


def _validate_k(k: int) -> None:
    if type(k) is not int or k <= 0:
        raise SimilarityConfigurationError("top-k must be a positive integer")


def directed_top_k(
    records: Sequence[VectorRecord],
    *,
    k: int = 5,
) -> tuple[LegacyDirectedNeighbor, ...]:
    """Rank every node's exact neighbors by score, then by target id."""

    _validate_k(k)
    vectors = validate_records(records)
    rows: list[LegacyDirectedNeighbor] = []
    for source in vectors:
        candidates = [
            (cosine_similarity(source.values, target.values), target.id)
            for target in vectors
            if target.id != source.id
        ]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for rank, (score, target_id) in enumerate(candidates[:k], start=1):
            rows.append(
                LegacyDirectedNeighbor(
                    source_id=source.id,
                    target_id=target_id,
                    rank=rank,
                    score=score,
                )
            )
    return tuple(rows)


def union_directed_neighbors(
    neighbors: Iterable[LegacyDirectedNeighbor],
) -> tuple[LegacyUndirectedNeighborEdge, ...]:
    """Union directed selections into exact, rank-aware undirected edges."""

    directed: dict[tuple[int, int], LegacyDirectedNeighbor] = {}
    pair_scores: dict[tuple[int, int], float] = {}
    for neighbor in neighbors:
        if (
            type(neighbor.source_id) is not int
            or type(neighbor.target_id) is not int
            or neighbor.source_id == neighbor.target_id
        ):
            raise VectorValidationError("directed neighbor endpoints are invalid")
        if type(neighbor.rank) is not int or neighbor.rank <= 0:
            raise VectorValidationError("directed neighbor rank is invalid")
        if not math.isfinite(neighbor.score):
            raise VectorValidationError("directed neighbor score is not finite")
        direction = (neighbor.source_id, neighbor.target_id)
        if direction in directed:
            raise VectorValidationError("directed neighbor was repeated")
        directed[direction] = neighbor
        pair = (
            min(neighbor.source_id, neighbor.target_id),
            max(neighbor.source_id, neighbor.target_id),
        )
        prior_score = pair_scores.get(pair)
        if prior_score is not None and not math.isclose(
            prior_score,
            neighbor.score,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise VectorValidationError("opposite exact-cosine scores disagree")
        pair_scores[pair] = neighbor.score

    edges: list[LegacyUndirectedNeighborEdge] = []
    for left_id, right_id in sorted(pair_scores):
        left_to_right = directed.get((left_id, right_id))
        right_to_left = directed.get((right_id, left_id))
        left_rank = left_to_right.rank if left_to_right is not None else None
        right_rank = right_to_left.rank if right_to_left is not None else None
        score = pair_scores[(left_id, right_id)]
        edges.append(
            LegacyUndirectedNeighborEdge(
                left_id=left_id,
                right_id=right_id,
                score=score,
                left_rank=left_rank,
                right_rank=right_rank,
                mutual=left_rank is not None and right_rank is not None,
            )
        )
    return tuple(edges)


def exact_top_k_union_graph(
    records: Sequence[VectorRecord],
    *,
    k: int = 5,
) -> tuple[LegacyUndirectedNeighborEdge, ...]:
    """Build directed exact top-k lists, then union them into undirected edges."""

    return union_directed_neighbors(directed_top_k(records, k=k))


def build_canonical_neighbor_artifact(
    records: Sequence[VectorRecord],
    *,
    k: int = 5,
) -> CanonicalNeighborArtifact:
    """Build the one exact-cosine source for recommendations and graph edges."""

    vectors = validate_records(records)
    directed = directed_top_k(vectors, k=k)
    return CanonicalNeighborArtifact(
        schema_version=CANONICAL_NEIGHBOR_SCHEMA_VERSION,
        provider_id=EXACT_COSINE_PROVIDER_ID,
        k=k,
        vector_ids=tuple(vector.id for vector in vectors),
        vector_set_sha256=canonical_vector_set_sha256(vectors),
        directed_neighbors=directed,
        undirected_edges=union_directed_neighbors(directed),
    )


def canonical_vector_set_sha256(records: Sequence[VectorRecord]) -> str:
    """Bind a canonical topology to numeric IDs and exact source vector values."""

    vectors = validate_records(records)
    payload = {
        "provider_id": EXACT_COSINE_PROVIDER_ID,
        "schema_version": CANONICAL_NEIGHBOR_SCHEMA_VERSION,
        "vectors": [
            {
                "id": vector.id,
                "values_hex": [value.hex() for value in vector.values],
            }
            for vector in vectors
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ExactCosineBackend:
    """Exact all-pairs backend suitable for the current 602-record corpus."""

    def directed_top_k(
        self,
        records: Sequence[VectorRecord],
        *,
        k: int = 5,
    ) -> tuple[LegacyDirectedNeighbor, ...]:
        return directed_top_k(records, k=k)


class AnnBackendPlaceholder:
    """Dependency-free interface marker; ANN is deliberately not implemented."""

    def directed_top_k(
        self,
        records: Sequence[VectorRecord],
        *,
        k: int = 5,
    ) -> tuple[LegacyDirectedNeighbor, ...]:
        _validate_k(k)
        raise AnnBackendUnavailableError(
            "ANN is only a future adapter; use exact all-pairs for 602 cocktails"
        )


def build_union_graph(
    records: Sequence[VectorRecord],
    *,
    backend: NeighborSearchBackend | None = None,
    k: int = 5,
) -> tuple[LegacyUndirectedNeighborEdge, ...]:
    """Build a graph through the neutral backend boundary used by module 3."""

    selected_backend = backend or ExactCosineBackend()
    return union_directed_neighbors(selected_backend.directed_top_k(records, k=k))
