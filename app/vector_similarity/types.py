"""Neutral value types for vector similarity and graph consumers."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

CANONICAL_NEIGHBOR_SCHEMA_VERSION = "cocktail-cosine-directed-union-v1"
EXACT_COSINE_PROVIDER_ID = "vector_similarity_exact_cosine_v1"
GRAPH48_DIMENSION = 48
GRAPH48_K = 5

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _is_canonical_cocktail_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.isascii()
        and value.isdecimal()
        and int(value) > 0
        and value == str(int(value))
    )


def _validate_run_id(value: object) -> None:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise VectorValidationError("run_id is invalid")


class VectorValidationError(ValueError):
    """A vector collection cannot participate in exact similarity."""


class SimilarityConfigurationError(ValueError):
    """A similarity or graph configuration is invalid."""


class AnnBackendUnavailableError(NotImplementedError):
    """ANN was selected even though this baseline intentionally has no ANN."""


@runtime_checkable
class VectorRecord(Protocol):
    """Minimal record shape accepted from an embedding-producing module."""

    @property
    def id(self) -> int: ...

    @property
    def values(self) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class DenseVector:
    """One identifier and one non-empty, dense, finite vector."""

    id: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.id) is not int:
            raise VectorValidationError("vector id must be an integer")
        if not isinstance(self.values, tuple) or not self.values:
            raise VectorValidationError("dense vector values must be a non-empty tuple")
        normalized: list[float] = []
        for value in self.values:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise VectorValidationError(
                    "dense vector values must all be finite numbers"
                )
            normalized.append(float(value))
        object.__setattr__(self, "values", tuple(normalized))

    @classmethod
    def from_values(cls, id: int, values: Sequence[float]) -> DenseVector:
        """Materialize an arbitrary finite sequence as an immutable vector."""

        return cls(id=id, values=tuple(values))


@dataclass(frozen=True, slots=True)
class PairSimilarity:
    """One exact unordered pair; higher score means more similar."""

    left_id: int
    right_id: int
    score: float


@dataclass(frozen=True, slots=True)
class DirectedNeighbor:
    """One canonical graph48 recommendation row."""

    run_id: str
    source_id: str
    target_id: str
    rank: int
    cosine: float

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        if (
            not _is_canonical_cocktail_id(self.source_id)
            or not _is_canonical_cocktail_id(self.target_id)
            or self.source_id == self.target_id
        ):
            raise VectorValidationError(
                "directed endpoints must be distinct canonical cocktail IDs"
            )
        if type(self.rank) is not int or not 1 <= self.rank <= GRAPH48_K:
            raise VectorValidationError("directed rank must be in [1, 5]")
        if (
            not isinstance(self.cosine, (int, float))
            or isinstance(self.cosine, bool)
            or not math.isfinite(float(self.cosine))
            or not -1.0 <= float(self.cosine) <= 1.0
        ):
            raise VectorValidationError("directed cosine must be finite and in [-1, 1]")
        object.__setattr__(self, "cosine", float(self.cosine))

    @property
    def score(self) -> float:
        """Compatibility name for older graph consumers."""

        return self.cosine


@dataclass(frozen=True, slots=True)
class UnionEdge:
    """Canonical either-direction union of graph48 recommendation rows."""

    run_id: str
    a_id: str
    b_id: str
    cosine: float
    a_rank: int | None
    b_rank: int | None

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        if (
            not _is_canonical_cocktail_id(self.a_id)
            or not _is_canonical_cocktail_id(self.b_id)
            or int(self.a_id) >= int(self.b_id)
        ):
            raise VectorValidationError(
                "union endpoints must be numeric-ascending canonical cocktail IDs"
            )
        if (
            not isinstance(self.cosine, (int, float))
            or isinstance(self.cosine, bool)
            or not math.isfinite(float(self.cosine))
            or not -1.0 <= float(self.cosine) <= 1.0
        ):
            raise VectorValidationError("union cosine must be finite and in [-1, 1]")
        object.__setattr__(self, "cosine", float(self.cosine))
        if self.a_rank is None and self.b_rank is None:
            raise VectorValidationError(
                "union edge must be selected by at least one direction"
            )
        for rank in (self.a_rank, self.b_rank):
            if rank is not None and (
                type(rank) is not int or not 1 <= rank <= GRAPH48_K
            ):
                raise VectorValidationError("union rank must be null or in [1, 5]")

    @property
    def left_id(self) -> int:
        """Compatibility endpoint for the existing spherical-graph adapter."""

        return int(self.a_id)

    @property
    def right_id(self) -> int:
        return int(self.b_id)

    @property
    def score(self) -> float:
        return self.cosine

    @property
    def left_rank(self) -> int | None:
        return self.a_rank

    @property
    def right_rank(self) -> int | None:
        return self.b_rank

    @property
    def mutual(self) -> bool:
        return self.a_rank is not None and self.b_rank is not None


@dataclass(frozen=True, slots=True)
class LegacyDirectedNeighbor:
    """Compatibility row for the pre-graph48 generic cosine helpers."""

    source_id: int
    target_id: int
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class LegacyUndirectedNeighborEdge:
    """Compatibility union row for pre-graph48 callers."""

    left_id: int
    right_id: int
    score: float
    left_rank: int | None
    right_rank: int | None
    mutual: bool


# Retained as an import alias for the pre-graph48 public surface.
UndirectedNeighborEdge = LegacyUndirectedNeighborEdge


@dataclass(frozen=True, slots=True)
class CanonicalRun:
    """Identity contract shared by all CSV rows in one graph48 build."""

    run_id: str
    vector_sha256: str
    dimension: int = GRAPH48_DIMENSION
    k: int = GRAPH48_K
    ids_sha256: str = ""

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        if (
            not isinstance(self.vector_sha256, str)
            or _SHA256.fullmatch(self.vector_sha256) is None
        ):
            raise VectorValidationError(
                "vector_sha256 must be a lowercase SHA-256 digest"
            )
        if self.dimension != GRAPH48_DIMENSION:
            raise SimilarityConfigurationError("canonical graph dimension must be 48")
        if self.k != GRAPH48_K:
            raise SimilarityConfigurationError("canonical graph k must be 5")
        if (
            not isinstance(self.ids_sha256, str)
            or _SHA256.fullmatch(self.ids_sha256) is None
        ):
            raise VectorValidationError("ids_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class PreferenceMatch:
    """One masked preference48 MIPS result, separate from graph topology."""

    cocktail_id: str
    rank: int
    score: float

    def __post_init__(self) -> None:
        if not _is_canonical_cocktail_id(self.cocktail_id):
            raise VectorValidationError(
                "preference match cocktail_id must be canonical"
            )
        if type(self.rank) is not int or self.rank <= 0:
            raise VectorValidationError("preference match rank must be positive")
        if (
            not isinstance(self.score, (int, float))
            or isinstance(self.score, bool)
            or not math.isfinite(float(self.score))
        ):
            raise VectorValidationError("preference match score must be finite")
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class InnerProductMatch:
    """One raw-vector query match using pgvector ``<#>`` semantics."""

    id: int
    rank: int
    negative_inner_product: float
    score: float


@dataclass(frozen=True, slots=True)
class CanonicalNeighborArtifact:
    """Single source of exact cocktail recommendations and graph topology."""

    schema_version: str
    provider_id: str
    k: int
    vector_ids: tuple[int, ...]
    vector_set_sha256: str
    directed_neighbors: tuple[LegacyDirectedNeighbor, ...]
    undirected_edges: tuple[LegacyUndirectedNeighborEdge, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CANONICAL_NEIGHBOR_SCHEMA_VERSION:
            raise VectorValidationError("canonical neighbor schema version is invalid")
        if self.provider_id != EXACT_COSINE_PROVIDER_ID:
            raise VectorValidationError("canonical neighbor provider is invalid")
        if type(self.k) is not int or self.k <= 0:
            raise SimilarityConfigurationError(
                "canonical neighbor k must be a positive integer"
            )
        if (
            not self.vector_ids
            or self.vector_ids != tuple(sorted(self.vector_ids))
            or len(set(self.vector_ids)) != len(self.vector_ids)
            or any(
                type(vector_id) is not int or vector_id <= 0
                for vector_id in self.vector_ids
            )
        ):
            raise VectorValidationError(
                "canonical vector IDs must be unique sorted positive integers"
            )
        if (
            len(self.vector_set_sha256) != 64
            or self.vector_set_sha256.lower() != self.vector_set_sha256
            or any(
                character not in "0123456789abcdef"
                for character in self.vector_set_sha256
            )
        ):
            raise VectorValidationError("canonical vector-set SHA-256 is invalid")
        if len(self.vector_ids) <= self.k:
            raise SimilarityConfigurationError(
                "canonical vector count must be greater than k"
            )

        known = set(self.vector_ids)
        expected_row_count = len(self.vector_ids) * self.k
        if len(self.directed_neighbors) != expected_row_count:
            raise VectorValidationError(
                "canonical directed rows must contain exactly k rows per vector"
            )
        if self.directed_neighbors != tuple(
            sorted(
                self.directed_neighbors,
                key=lambda row: (row.source_id, row.rank),
            )
        ):
            raise VectorValidationError(
                "canonical directed rows must be ordered by source ID and rank"
            )

        by_source: dict[int, list[LegacyDirectedNeighbor]] = {
            vector_id: [] for vector_id in self.vector_ids
        }
        directions: dict[tuple[int, int], LegacyDirectedNeighbor] = {}
        pair_scores: dict[tuple[int, int], float] = {}
        for row in self.directed_neighbors:
            if (
                type(row.source_id) is not int
                or type(row.target_id) is not int
                or row.source_id not in known
                or row.target_id not in known
                or row.source_id == row.target_id
            ):
                raise VectorValidationError("canonical directed endpoints are invalid")
            if type(row.rank) is not int or not 1 <= row.rank <= self.k:
                raise VectorValidationError("canonical directed rank is invalid")
            if not math.isfinite(row.score) or not -1.0 <= row.score <= 1.0:
                raise VectorValidationError("canonical cosine score is invalid")
            direction = (row.source_id, row.target_id)
            if direction in directions:
                raise VectorValidationError("canonical directed neighbor was repeated")
            directions[direction] = row
            by_source[row.source_id].append(row)
            pair = (
                min(row.source_id, row.target_id),
                max(row.source_id, row.target_id),
            )
            prior_score = pair_scores.get(pair)
            if prior_score is not None and not math.isclose(
                prior_score,
                row.score,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise VectorValidationError("canonical opposite cosine scores disagree")
            pair_scores[pair] = row.score

        for source_id, rows in by_source.items():
            if [row.rank for row in rows] != list(range(1, self.k + 1)):
                raise VectorValidationError(
                    f"canonical source {source_id} must have ranks 1 through k"
                )
            if len({row.target_id for row in rows}) != self.k:
                raise VectorValidationError(
                    f"canonical source {source_id} repeats a target"
                )
            if rows != sorted(rows, key=lambda row: (-row.score, row.target_id)):
                raise VectorValidationError(
                    "canonical ranks must use score then numeric target-ID order"
                )

        if self.undirected_edges != tuple(
            sorted(
                self.undirected_edges,
                key=lambda edge: (edge.left_id, edge.right_id),
            )
        ):
            raise VectorValidationError(
                "canonical undirected edges must use numeric endpoint order"
            )
        if len(self.undirected_edges) != len(pair_scores):
            raise VectorValidationError(
                "canonical undirected edges must equal the directed-row union"
            )
        seen_pairs: set[tuple[int, int]] = set()
        for edge in self.undirected_edges:
            pair = (edge.left_id, edge.right_id)
            if (
                type(edge.left_id) is not int
                or type(edge.right_id) is not int
                or edge.left_id not in known
                or edge.right_id not in known
                or edge.left_id >= edge.right_id
                or pair in seen_pairs
            ):
                raise VectorValidationError(
                    "canonical undirected endpoints are invalid"
                )
            seen_pairs.add(pair)
            left_row = directions.get(pair)
            right_row = directions.get((edge.right_id, edge.left_id))
            expected_left_rank = left_row.rank if left_row is not None else None
            expected_right_rank = right_row.rank if right_row is not None else None
            if (
                pair not in pair_scores
                or not math.isfinite(edge.score)
                or not math.isclose(
                    edge.score,
                    pair_scores[pair],
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or edge.left_rank != expected_left_rank
                or edge.right_rank != expected_right_rank
                or edge.mutual
                != (expected_left_rank is not None and expected_right_rank is not None)
            ):
                raise VectorValidationError(
                    "canonical undirected edge does not match directed rows"
                )
        if seen_pairs != set(pair_scores):
            raise VectorValidationError(
                "canonical undirected edges must equal the directed-row union"
            )

    def recommendations_for(
        self,
        cocktail_id: int,
    ) -> tuple[LegacyDirectedNeighbor, ...]:
        """Return this cocktail's canonical outgoing rank 1..k rows."""

        if cocktail_id not in self.vector_ids:
            raise VectorValidationError(f"unknown canonical cocktail ID: {cocktail_id}")
        return tuple(
            row for row in self.directed_neighbors if row.source_id == cocktail_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "k": self.k,
            "vector_ids": list(self.vector_ids),
            "vector_set_sha256": self.vector_set_sha256,
            "directed_neighbors": [
                {
                    "source_id": row.source_id,
                    "target_id": row.target_id,
                    "rank": row.rank,
                    "score": row.score,
                }
                for row in self.directed_neighbors
            ],
            "undirected_edges": [
                {
                    "left_id": edge.left_id,
                    "right_id": edge.right_id,
                    "score": edge.score,
                    "left_rank": edge.left_rank,
                    "right_rank": edge.right_rank,
                    "mutual": edge.mutual,
                }
                for edge in self.undirected_edges
            ],
        }


@runtime_checkable
class NeighborSearchBackend(Protocol):
    """Backend boundary shared by the exact baseline and a future ANN adapter."""

    def directed_top_k(
        self,
        records: Sequence[VectorRecord],
        *,
        k: int = 5,
    ) -> tuple[LegacyDirectedNeighbor, ...]: ...
