"""Canonical exact graph48 topology and separate preference48 MIPS."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .types import (
    GRAPH48_DIMENSION,
    GRAPH48_K,
    CanonicalRun,
    DirectedNeighbor,
    PreferenceMatch,
    SimilarityConfigurationError,
    UnionEdge,
    VectorRecord,
    VectorValidationError,
)

UNIT_L2_ABS_TOLERANCE = 1e-6


def _optional_sha256(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise VectorValidationError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class Graph48Pair:
    """One exact unordered graph48 dot product."""

    a_id: str
    b_id: str
    cosine: float


@dataclass(frozen=True, slots=True)
class _Graph48Vector:
    cocktail_id: str
    values: tuple[float, ...]


def canonical_cocktail_id(value: object) -> str:
    """Return a positive canonical decimal cocktail ID."""

    if type(value) is int:
        if value <= 0:
            raise VectorValidationError("cocktail_id must be positive")
        return str(value)
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        try:
            number = int(value)
        except ValueError as error:  # pragma: no cover - guarded by isdecimal
            raise VectorValidationError("cocktail_id is invalid") from error
        if number > 0 and value == str(number):
            return value
    raise VectorValidationError(
        "cocktail_id must be a positive canonical decimal string"
    )


def _record_id(record: object) -> str:
    if hasattr(record, "cocktail_id"):
        return canonical_cocktail_id(getattr(record, "cocktail_id"))
    if hasattr(record, "id"):
        return canonical_cocktail_id(getattr(record, "id"))
    raise VectorValidationError("vector record must expose cocktail_id or id")


def _finite_vector(
    values: object,
    *,
    dimension: int,
    label: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise VectorValidationError(f"{label} must be a numeric vector")
    try:
        received: tuple[object, ...] = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise VectorValidationError(f"{label} must be a numeric vector") from error
    if len(received) != dimension:
        raise VectorValidationError(
            f"{label} must contain exactly {dimension} dimensions"
        )
    result: list[float] = []
    for value in received:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise VectorValidationError(f"{label} must contain finite numbers")
        numeric = float(value)
        result.append(0.0 if numeric == 0.0 else numeric)
    return tuple(result)


def validate_graph48_records(
    records: Sequence[VectorRecord],
) -> tuple[_Graph48Vector, ...]:
    """Validate 48D unit vectors and sort them by numeric cocktail ID."""

    materialized: list[_Graph48Vector] = []
    seen: set[str] = set()
    for record in records:
        cocktail_id = _record_id(record)
        if cocktail_id in seen:
            raise VectorValidationError(f"duplicate cocktail_id: {cocktail_id}")
        seen.add(cocktail_id)
        if not hasattr(record, "values"):
            raise VectorValidationError("vector record must expose values")
        values = _finite_vector(
            getattr(record, "values"),
            dimension=GRAPH48_DIMENSION,
            label=f"graph48 cocktail {cocktail_id}",
        )
        norm = math.sqrt(math.fsum(value * value for value in values))
        if not math.isclose(
            norm,
            1.0,
            rel_tol=0.0,
            abs_tol=UNIT_L2_ABS_TOLERANCE,
        ):
            raise VectorValidationError(
                f"graph48 cocktail {cocktail_id} must have unit L2 norm"
            )
        materialized.append(_Graph48Vector(cocktail_id, values))
    materialized.sort(key=lambda row: int(row.cocktail_id))
    return tuple(materialized)


def graph48_ids_sha256(cocktail_ids: Sequence[str | int]) -> str:
    """Hash the numerically ordered canonical cocktail-ID set."""

    ids = tuple(canonical_cocktail_id(value) for value in cocktail_ids)
    if len(ids) != len(set(ids)):
        raise VectorValidationError("cocktail IDs must be unique")
    ordered = tuple(sorted(ids, key=int))
    encoded = json.dumps(
        {"cocktail_ids": list(ordered)},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def graph48_vector_sha256(records: Sequence[VectorRecord]) -> str:
    """Hash canonical graph48 IDs and exact IEEE-754 vector values."""

    return _validated_vector_sha256(validate_graph48_records(records))


def _validated_vector_sha256(vectors: Sequence[_Graph48Vector]) -> str:
    payload = {
        "dimension": GRAPH48_DIMENSION,
        "space": "graph48",
        "vectors": [
            {
                "cocktail_id": vector.cocktail_id,
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


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    try:
        dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(math.fsum(value * value for value in left))
        right_norm = math.sqrt(math.fsum(value * value for value in right))
        score = dot / (left_norm * right_norm)
    except (OverflowError, ValueError) as error:
        raise VectorValidationError("graph48 cosine is not finite") from error
    if not math.isfinite(score):
        raise VectorValidationError("graph48 cosine is not finite")
    return min(1.0, max(-1.0, score))


def graph48_all_pairs(
    records: Sequence[VectorRecord],
) -> tuple[Graph48Pair, ...]:
    """Compute each unordered exact cosine exactly once."""

    return _validated_all_pairs(validate_graph48_records(records))


def _validated_all_pairs(
    vectors: Sequence[_Graph48Vector],
) -> tuple[Graph48Pair, ...]:
    pairs: list[Graph48Pair] = []
    for a_index, a_vector in enumerate(vectors):
        for b_vector in vectors[a_index + 1 :]:
            pairs.append(
                Graph48Pair(
                    a_id=a_vector.cocktail_id,
                    b_id=b_vector.cocktail_id,
                    cosine=_cosine(a_vector.values, b_vector.values),
                )
            )
    return tuple(pairs)


def _directed_top_five(
    run_id: str,
    cocktail_ids: tuple[str, ...],
    pairs: tuple[Graph48Pair, ...],
) -> tuple[DirectedNeighbor, ...]:
    by_source: dict[str, list[tuple[float, str]]] = {
        cocktail_id: [] for cocktail_id in cocktail_ids
    }
    for pair in pairs:
        by_source[pair.a_id].append((pair.cosine, pair.b_id))
        by_source[pair.b_id].append((pair.cosine, pair.a_id))

    rows: list[DirectedNeighbor] = []
    for source_id in cocktail_ids:
        candidates = sorted(
            by_source[source_id],
            key=lambda item: (-item[0], int(item[1])),
        )
        for rank, (cosine, target_id) in enumerate(
            candidates[:GRAPH48_K],
            start=1,
        ):
            rows.append(
                DirectedNeighbor(
                    run_id=run_id,
                    source_id=source_id,
                    target_id=target_id,
                    rank=rank,
                    cosine=cosine,
                )
            )
    return tuple(rows)


def _union_rows(
    run_id: str,
    neighbors: Sequence[DirectedNeighbor],
) -> tuple[UnionEdge, ...]:
    directions = {(row.source_id, row.target_id): row for row in neighbors}
    pairs = sorted(
        {
            tuple(
                sorted(
                    (row.source_id, row.target_id),
                    key=int,
                )
            )
            for row in neighbors
        },
        key=lambda pair: (int(pair[0]), int(pair[1])),
    )
    edges: list[UnionEdge] = []
    for a_id, b_id in pairs:
        a_row = directions.get((a_id, b_id))
        b_row = directions.get((b_id, a_id))
        selected = a_row if a_row is not None else b_row
        assert selected is not None
        if a_row is not None and b_row is not None and a_row.cosine != b_row.cosine:
            raise VectorValidationError(
                "opposite graph48 directions must have identical cosine"
            )
        edges.append(
            UnionEdge(
                run_id=run_id,
                a_id=a_id,
                b_id=b_id,
                cosine=selected.cosine,
                a_rank=a_row.rank if a_row is not None else None,
                b_rank=b_row.rank if b_row is not None else None,
            )
        )
    return tuple(edges)


@dataclass(frozen=True, slots=True)
class CanonicalGraph48Artifact:
    """The only canonical cocktail recommendation and graph-edge source."""

    run: CanonicalRun
    cocktail_ids: tuple[str, ...]
    directed_neighbors: tuple[DirectedNeighbor, ...]
    union_edges: tuple[UnionEdge, ...]
    source_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run, CanonicalRun):
            raise VectorValidationError("canonical graph48 run is invalid")
        _optional_sha256(
            self.source_artifact_sha256,
            field="source_artifact_sha256",
        )
        ids = tuple(canonical_cocktail_id(value) for value in self.cocktail_ids)
        if ids != self.cocktail_ids:
            raise VectorValidationError(
                "canonical graph48 IDs must already be canonical strings"
            )
        if ids != tuple(sorted(ids, key=int)) or len(ids) != len(set(ids)):
            raise VectorValidationError(
                "canonical graph48 IDs must be unique and numeric-sorted"
            )
        if len(ids) <= GRAPH48_K:
            raise SimilarityConfigurationError(
                "canonical graph48 requires at least six cocktails"
            )
        if graph48_ids_sha256(ids) != self.run.ids_sha256:
            raise VectorValidationError("canonical graph48 ids_sha256 mismatch")

        known = set(ids)
        expected_count = len(ids) * GRAPH48_K
        if len(self.directed_neighbors) != expected_count:
            raise VectorValidationError(
                "canonical graph48 must have exactly 5 outgoing rows per cocktail"
            )
        if self.directed_neighbors != tuple(
            sorted(
                self.directed_neighbors,
                key=lambda row: (int(row.source_id), row.rank),
            )
        ):
            raise VectorValidationError(
                "directed rows must be numeric-source sorted, then rank sorted"
            )

        directions: dict[tuple[str, str], DirectedNeighbor] = {}
        by_source: dict[str, list[DirectedNeighbor]] = {
            cocktail_id: [] for cocktail_id in ids
        }
        for row in self.directed_neighbors:
            if row.run_id != self.run.run_id:
                raise VectorValidationError("directed row run_id mismatch")
            source_id = canonical_cocktail_id(row.source_id)
            target_id = canonical_cocktail_id(row.target_id)
            if (
                source_id != row.source_id
                or target_id != row.target_id
                or source_id not in known
                or target_id not in known
                or source_id == target_id
            ):
                raise VectorValidationError("directed row endpoints are invalid")
            if type(row.rank) is not int or not 1 <= row.rank <= GRAPH48_K:
                raise VectorValidationError("directed row rank is invalid")
            if (
                not isinstance(row.cosine, float)
                or not math.isfinite(row.cosine)
                or not -1.0 <= row.cosine <= 1.0
            ):
                raise VectorValidationError("directed row cosine is invalid")
            direction = (source_id, target_id)
            if direction in directions:
                raise VectorValidationError("directed row is duplicated")
            directions[direction] = row
            by_source[source_id].append(row)

        for source_id, rows in by_source.items():
            if [row.rank for row in rows] != list(range(1, GRAPH48_K + 1)):
                raise VectorValidationError(
                    f"cocktail {source_id} must have directed ranks 1 through 5"
                )
            if len({row.target_id for row in rows}) != GRAPH48_K:
                raise VectorValidationError(
                    f"cocktail {source_id} repeats a directed target"
                )
            if rows != sorted(
                rows,
                key=lambda row: (-row.cosine, int(row.target_id)),
            ):
                raise VectorValidationError(
                    "directed ranks must order cosine DESC then cocktail_id ASC"
                )

        expected_pairs = {tuple(sorted(direction, key=int)) for direction in directions}
        if len(self.union_edges) != len(expected_pairs):
            raise VectorValidationError(
                "union rows must equal the either-direction directed union"
            )
        if self.union_edges != tuple(
            sorted(
                self.union_edges,
                key=lambda row: (int(row.a_id), int(row.b_id)),
            )
        ):
            raise VectorValidationError("union rows must be numeric endpoint sorted")

        seen_pairs: set[tuple[str, str]] = set()
        for edge in self.union_edges:
            if edge.run_id != self.run.run_id:
                raise VectorValidationError("union row run_id mismatch")
            a_id = canonical_cocktail_id(edge.a_id)
            b_id = canonical_cocktail_id(edge.b_id)
            pair = (a_id, b_id)
            if (
                pair in seen_pairs
                or a_id not in known
                or b_id not in known
                or int(a_id) >= int(b_id)
            ):
                raise VectorValidationError("union row endpoints are invalid")
            seen_pairs.add(pair)
            a_row = directions.get(pair)
            b_row = directions.get((b_id, a_id))
            if a_row is None and b_row is None:
                raise VectorValidationError(
                    "union row is absent from both directed selections"
                )
            selected = a_row if a_row is not None else b_row
            assert selected is not None
            if (
                (a_row is not None and a_row.cosine != selected.cosine)
                or (b_row is not None and b_row.cosine != selected.cosine)
                or edge.cosine != selected.cosine
                or edge.a_rank != (a_row.rank if a_row is not None else None)
                or edge.b_rank != (b_row.rank if b_row is not None else None)
            ):
                raise VectorValidationError(
                    "union row must preserve ranks and identical cosine"
                )
            for rank in (edge.a_rank, edge.b_rank):
                if rank is not None and (
                    type(rank) is not int or not 1 <= rank <= GRAPH48_K
                ):
                    raise VectorValidationError("union row rank is invalid")
        if seen_pairs != expected_pairs:
            raise VectorValidationError(
                "union rows must equal the either-direction directed union"
            )

    @property
    def k(self) -> int:
        return self.run.k

    @property
    def vector_sha256(self) -> str:
        return self.run.vector_sha256

    def recommendations_for(
        self,
        cocktail_id: str | int,
    ) -> tuple[DirectedNeighbor, ...]:
        """Read only canonical directed rows for recommendation responses."""

        source_id = canonical_cocktail_id(cocktail_id)
        if source_id not in self.cocktail_ids:
            raise VectorValidationError(f"unknown canonical cocktail_id: {source_id}")
        return tuple(
            row for row in self.directed_neighbors if row.source_id == source_id
        )

    def graph_edges(self) -> tuple[UnionEdge, ...]:
        """Read only the either-direction union for graph construction."""

        return self.union_edges

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": {
                "run_id": self.run.run_id,
                "vector_sha256": self.run.vector_sha256,
                "dimension": self.run.dimension,
                "k": self.run.k,
                "ids_sha256": self.run.ids_sha256,
            },
            "source_artifact_sha256": self.source_artifact_sha256,
            "cocktail_ids": list(self.cocktail_ids),
            "directed_neighbors": [
                {
                    "run_id": row.run_id,
                    "source_id": row.source_id,
                    "target_id": row.target_id,
                    "rank": row.rank,
                    "cosine": row.cosine,
                }
                for row in self.directed_neighbors
            ],
            "union_edges": [
                {
                    "run_id": edge.run_id,
                    "a_id": edge.a_id,
                    "b_id": edge.b_id,
                    "cosine": edge.cosine,
                    "a_rank": edge.a_rank,
                    "b_rank": edge.b_rank,
                }
                for edge in self.union_edges
            ],
        }


def build_graph48_artifact(
    records: Sequence[VectorRecord],
    *,
    run_id: str,
    source_artifact_sha256: str | None = None,
) -> CanonicalGraph48Artifact:
    """Build exact all-pairs directed top-5 rows and their union."""

    vectors = validate_graph48_records(records)
    if len(vectors) <= GRAPH48_K:
        raise SimilarityConfigurationError(
            "canonical graph48 requires at least six cocktails"
        )
    ids = tuple(vector.cocktail_id for vector in vectors)
    computed_vector_sha256 = _validated_vector_sha256(vectors)
    run = CanonicalRun(
        run_id=run_id,
        vector_sha256=computed_vector_sha256,
        ids_sha256=graph48_ids_sha256(ids),
    )
    pairs = _validated_all_pairs(vectors)
    directed = _directed_top_five(run.run_id, ids, pairs)
    return CanonicalGraph48Artifact(
        run=run,
        cocktail_ids=ids,
        directed_neighbors=directed,
        union_edges=_union_rows(run.run_id, directed),
        source_artifact_sha256=_optional_sha256(
            source_artifact_sha256,
            field="source_artifact_sha256",
        ),
    )


def preference48_mips(
    query: Sequence[float] | object,
    records: Sequence[VectorRecord],
    *,
    k: int = GRAPH48_K,
) -> tuple[PreferenceMatch, ...]:
    """Rank preference48 vectors using only coordinates where ``q_j != 0``."""

    if type(k) is not int or k <= 0:
        raise SimilarityConfigurationError("MIPS k must be a positive integer")
    query_values = _finite_vector(
        getattr(query, "values", query),
        dimension=GRAPH48_DIMENSION,
        label="preference48 query",
    )
    selected_positions = tuple(
        index for index, value in enumerate(query_values) if value != 0.0
    )
    if not selected_positions:
        raise VectorValidationError(
            "preference48 query must select at least one nonzero coordinate"
        )

    candidates: list[tuple[float, str]] = []
    seen: set[str] = set()
    for record in records:
        cocktail_id = _record_id(record)
        if cocktail_id in seen:
            raise VectorValidationError(f"duplicate cocktail_id: {cocktail_id}")
        seen.add(cocktail_id)
        if not hasattr(record, "values"):
            raise VectorValidationError("vector record must expose values")
        values = _finite_vector(
            getattr(record, "values"),
            dimension=GRAPH48_DIMENSION,
            label=f"preference48 cocktail {cocktail_id}",
        )
        try:
            score = math.fsum(
                query_values[index] * values[index] for index in selected_positions
            )
        except (OverflowError, ValueError) as error:
            raise VectorValidationError(
                "preference48 inner product is not finite"
            ) from error
        if not math.isfinite(score):
            raise VectorValidationError("preference48 inner product is not finite")
        candidates.append((score, cocktail_id))
    candidates.sort(key=lambda item: (-item[0], int(item[1])))
    return tuple(
        PreferenceMatch(cocktail_id=cocktail_id, rank=rank, score=score)
        for rank, (score, cocktail_id) in enumerate(
            candidates[:k],
            start=1,
        )
    )


rank_preference48_mips = preference48_mips
