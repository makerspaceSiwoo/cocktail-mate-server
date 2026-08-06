"""Strict, create-only CSV persistence for canonical graph48 artifacts."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from os import PathLike
from pathlib import Path

from .graph48 import (
    CanonicalGraph48Artifact,
    canonical_cocktail_id,
)
from .types import (
    GRAPH48_K,
    CanonicalRun,
    DirectedNeighbor,
    UnionEdge,
    VectorValidationError,
)

CANONICAL_RUN_CSV_FIELDS = (
    "run_id",
    "vector_sha256",
    "dimension",
    "k",
    "ids_sha256",
)
DIRECTED_NEIGHBOR_CSV_FIELDS = (
    "run_id",
    "source_id",
    "target_id",
    "rank",
    "cosine",
)
UNION_EDGE_CSV_FIELDS = (
    "run_id",
    "a_id",
    "b_id",
    "cosine",
    "a_rank",
    "b_rank",
)


def _path(value: str | PathLike[str]) -> Path:
    return Path(value)


def _canonical_integer(text: str, *, field: str, positive: bool = True) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError) as error:
        raise VectorValidationError(f"{field} must be a canonical integer") from error
    if text != str(value) or (positive and value <= 0):
        raise VectorValidationError(f"{field} must be a canonical integer")
    return value


def _finite_cosine(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as error:
        raise VectorValidationError("cosine must be a finite number") from error
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise VectorValidationError("cosine must be finite and in [-1, 1]")
    return value


def _read_dict_rows(
    path: str | PathLike[str],
    fields: tuple[str, ...],
) -> list[dict[str, str]]:
    selected_path = _path(path)
    try:
        with selected_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != fields:
                raise VectorValidationError(f"CSV header must be exactly {fields}")
            rows: list[dict[str, str]] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(row.get(field) is None for field in fields):
                    raise VectorValidationError(
                        f"CSV row {line_number} has missing or extra fields"
                    )
                materialized = {field: row[field] for field in fields}
                if all(value == "" for value in materialized.values()):
                    raise VectorValidationError(
                        f"CSV row {line_number} must not be blank"
                    )
                rows.append(materialized)
            return rows
    except OSError as error:
        raise VectorValidationError(f"cannot read CSV: {selected_path}") from error


def _write_rows_create_only(
    path: str | PathLike[str],
    fields: tuple[str, ...],
    rows: Iterable[dict[str, object]],
) -> None:
    selected_path = _path(path)
    try:
        with selected_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fields),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
    except FileExistsError:
        raise
    except OSError as error:
        raise VectorValidationError(f"cannot create CSV: {selected_path}") from error


def write_canonical_run_csv(
    path: str | PathLike[str],
    run: CanonicalRun,
) -> None:
    """Create one canonical-run CSV; an existing path is never overwritten."""

    if not isinstance(run, CanonicalRun):
        raise VectorValidationError("run must be CanonicalRun")
    _write_rows_create_only(
        path,
        CANONICAL_RUN_CSV_FIELDS,
        (
            {
                "run_id": run.run_id,
                "vector_sha256": run.vector_sha256,
                "dimension": run.dimension,
                "k": run.k,
                "ids_sha256": run.ids_sha256,
            },
        ),
    )


def read_canonical_run_csv(path: str | PathLike[str]) -> CanonicalRun:
    rows = _read_dict_rows(path, CANONICAL_RUN_CSV_FIELDS)
    if len(rows) != 1:
        raise VectorValidationError("canonical-run CSV must contain exactly one row")
    row = rows[0]
    return CanonicalRun(
        run_id=row["run_id"],
        vector_sha256=row["vector_sha256"],
        dimension=_canonical_integer(row["dimension"], field="dimension"),
        k=_canonical_integer(row["k"], field="k"),
        ids_sha256=row["ids_sha256"],
    )


def write_directed_neighbors_csv(
    path: str | PathLike[str],
    rows: Sequence[DirectedNeighbor],
) -> None:
    """Create a directed-row CSV using the locked field order."""

    _validate_directed_csv_rows(rows)
    _write_rows_create_only(
        path,
        DIRECTED_NEIGHBOR_CSV_FIELDS,
        (
            {
                "run_id": row.run_id,
                "source_id": row.source_id,
                "target_id": row.target_id,
                "rank": row.rank,
                "cosine": repr(row.cosine),
            }
            for row in rows
        ),
    )


def _validate_directed_csv_rows(rows: Sequence[DirectedNeighbor]) -> None:
    if not rows:
        raise VectorValidationError("directed-row CSV must not be empty")
    run_ids: set[str] = set()
    seen: set[tuple[str, str]] = set()
    by_source: dict[str, list[DirectedNeighbor]] = {}
    for row in rows:
        if not isinstance(row, DirectedNeighbor):
            raise VectorValidationError("directed CSV values must be DirectedNeighbor")
        # Constructing a temporary run applies the exact run-id validator.
        CanonicalRun(
            run_id=row.run_id,
            vector_sha256="0" * 64,
            ids_sha256="0" * 64,
        )
        run_ids.add(row.run_id)
        source_id = canonical_cocktail_id(row.source_id)
        target_id = canonical_cocktail_id(row.target_id)
        if source_id != row.source_id or target_id != row.target_id:
            raise VectorValidationError("directed IDs must be canonical strings")
        if source_id == target_id:
            raise VectorValidationError("directed self-edges are forbidden")
        if type(row.rank) is not int or not 1 <= row.rank <= GRAPH48_K:
            raise VectorValidationError("directed rank must be in [1, 5]")
        if (
            not isinstance(row.cosine, float)
            or not math.isfinite(row.cosine)
            or not -1.0 <= row.cosine <= 1.0
        ):
            raise VectorValidationError("directed cosine is invalid")
        direction = (source_id, target_id)
        if direction in seen:
            raise VectorValidationError("directed row is duplicated")
        seen.add(direction)
        by_source.setdefault(source_id, []).append(row)
    if len(run_ids) != 1:
        raise VectorValidationError("directed CSV must contain exactly one run_id")
    known = set(by_source)
    if any(row.target_id not in known for row in rows):
        raise VectorValidationError(
            "every directed target must also have five outgoing rows"
        )
    for source_id, source_rows in by_source.items():
        if len(source_rows) != GRAPH48_K or [row.rank for row in source_rows] != list(
            range(1, GRAPH48_K + 1)
        ):
            raise VectorValidationError(
                f"directed source {source_id} must have ranks 1 through 5"
            )
        if len({row.target_id for row in source_rows}) != GRAPH48_K:
            raise VectorValidationError(
                f"directed source {source_id} must have five unique targets"
            )
        if source_rows != sorted(
            source_rows,
            key=lambda row: (-row.cosine, int(row.target_id)),
        ):
            raise VectorValidationError(
                "directed rows must order cosine DESC then cocktail_id ASC"
            )
    if tuple(rows) != tuple(
        sorted(rows, key=lambda row: (int(row.source_id), row.rank))
    ):
        raise VectorValidationError(
            "directed rows must be numeric-source sorted, then rank sorted"
        )


def read_directed_neighbors_csv(
    path: str | PathLike[str],
) -> tuple[DirectedNeighbor, ...]:
    received = _read_dict_rows(path, DIRECTED_NEIGHBOR_CSV_FIELDS)
    rows = tuple(
        DirectedNeighbor(
            run_id=row["run_id"],
            source_id=canonical_cocktail_id(row["source_id"]),
            target_id=canonical_cocktail_id(row["target_id"]),
            rank=_canonical_integer(row["rank"], field="rank"),
            cosine=_finite_cosine(row["cosine"]),
        )
        for row in received
    )
    _validate_directed_csv_rows(rows)
    return rows


def write_union_edges_csv(
    path: str | PathLike[str],
    rows: Sequence[UnionEdge],
) -> None:
    """Create an either-direction union CSV using the locked field order."""

    _validate_union_csv_rows(rows)
    _write_rows_create_only(
        path,
        UNION_EDGE_CSV_FIELDS,
        (
            {
                "run_id": row.run_id,
                "a_id": row.a_id,
                "b_id": row.b_id,
                "cosine": repr(row.cosine),
                "a_rank": "" if row.a_rank is None else row.a_rank,
                "b_rank": "" if row.b_rank is None else row.b_rank,
            }
            for row in rows
        ),
    )


def _validate_union_csv_rows(rows: Sequence[UnionEdge]) -> None:
    if not rows:
        raise VectorValidationError("union-edge CSV must not be empty")
    run_ids: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, UnionEdge):
            raise VectorValidationError("union CSV values must be UnionEdge")
        CanonicalRun(
            run_id=row.run_id,
            vector_sha256="0" * 64,
            ids_sha256="0" * 64,
        )
        run_ids.add(row.run_id)
        a_id = canonical_cocktail_id(row.a_id)
        b_id = canonical_cocktail_id(row.b_id)
        if a_id != row.a_id or b_id != row.b_id or int(a_id) >= int(b_id):
            raise VectorValidationError(
                "union IDs must be canonical numeric-ascending strings"
            )
        if (
            not isinstance(row.cosine, float)
            or not math.isfinite(row.cosine)
            or not -1.0 <= row.cosine <= 1.0
        ):
            raise VectorValidationError("union cosine is invalid")
        if row.a_rank is None and row.b_rank is None:
            raise VectorValidationError(
                "union row must be selected by at least one direction"
            )
        for rank in (row.a_rank, row.b_rank):
            if rank is not None and (
                type(rank) is not int or not 1 <= rank <= GRAPH48_K
            ):
                raise VectorValidationError("union rank must be null or in [1, 5]")
        pair = (a_id, b_id)
        if pair in seen:
            raise VectorValidationError("union row is duplicated")
        seen.add(pair)
    if len(run_ids) != 1:
        raise VectorValidationError("union CSV must contain exactly one run_id")
    if tuple(rows) != tuple(
        sorted(rows, key=lambda row: (int(row.a_id), int(row.b_id)))
    ):
        raise VectorValidationError("union rows must be numeric endpoint sorted")


def _nullable_rank(text: str, *, field: str) -> int | None:
    if text == "":
        return None
    return _canonical_integer(text, field=field)


def read_union_edges_csv(
    path: str | PathLike[str],
) -> tuple[UnionEdge, ...]:
    received = _read_dict_rows(path, UNION_EDGE_CSV_FIELDS)
    rows = tuple(
        UnionEdge(
            run_id=row["run_id"],
            a_id=canonical_cocktail_id(row["a_id"]),
            b_id=canonical_cocktail_id(row["b_id"]),
            cosine=_finite_cosine(row["cosine"]),
            a_rank=_nullable_rank(row["a_rank"], field="a_rank"),
            b_rank=_nullable_rank(row["b_rank"], field="b_rank"),
        )
        for row in received
    )
    _validate_union_csv_rows(rows)
    return rows


def write_canonical_graph48_csv(
    artifact: CanonicalGraph48Artifact,
    *,
    run_path: str | PathLike[str],
    directed_path: str | PathLike[str],
    union_path: str | PathLike[str],
    verify_readback: bool = True,
) -> None:
    """Create the three locked CSVs and optionally assert exact readback."""

    if not isinstance(artifact, CanonicalGraph48Artifact):
        raise VectorValidationError("artifact must be CanonicalGraph48Artifact")
    paths = (_path(run_path), _path(directed_path), _path(union_path))
    if len(set(paths)) != len(paths):
        raise VectorValidationError("canonical CSV paths must be distinct")
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "canonical CSV output already exists: " + ", ".join(existing)
        )

    write_canonical_run_csv(paths[0], artifact.run)
    write_directed_neighbors_csv(paths[1], artifact.directed_neighbors)
    write_union_edges_csv(paths[2], artifact.union_edges)
    if verify_readback:
        loaded = read_canonical_graph48_csv(
            run_path=paths[0],
            directed_path=paths[1],
            union_path=paths[2],
            source_artifact_sha256=artifact.source_artifact_sha256,
        )
        if loaded != artifact:
            raise VectorValidationError(
                "canonical graph48 CSV readback is not exactly equal"
            )


def read_canonical_graph48_csv(
    *,
    run_path: str | PathLike[str],
    directed_path: str | PathLike[str],
    union_path: str | PathLike[str],
    expected_run_id: str | None = None,
    expected_vector_sha256: str | None = None,
    source_artifact_sha256: str | None = None,
) -> CanonicalGraph48Artifact:
    """Strictly load and revalidate one canonical graph48 CSV bundle."""

    run = read_canonical_run_csv(run_path)
    if expected_run_id is not None and run.run_id != expected_run_id:
        raise VectorValidationError("canonical CSV run_id does not match expected")
    if (
        expected_vector_sha256 is not None
        and run.vector_sha256 != expected_vector_sha256
    ):
        raise VectorValidationError(
            "canonical CSV vector_sha256 does not match expected"
        )
    directed = read_directed_neighbors_csv(directed_path)
    union = read_union_edges_csv(union_path)
    ids = tuple(
        sorted(
            {row.source_id for row in directed},
            key=int,
        )
    )
    return CanonicalGraph48Artifact(
        run=run,
        cocktail_ids=ids,
        directed_neighbors=directed,
        union_edges=union,
        source_artifact_sha256=source_artifact_sha256,
    )
