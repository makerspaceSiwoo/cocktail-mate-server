"""Load Graph48, Preference48, and S2 sphere coordinates into ``cocktails``.

This loader is deliberately separate from ``scripts/build_cocktail_embeddings.py``.
That script hard-asserts 32D artifacts and takes its session from the server
runtime engine (``app.core.database.SessionLocal``), so it can only ever write
whatever database the ambient process environment happens to point at. This
script takes an explicit ``--dsn`` and refuses to fall back to anything else.

The write is a single transaction that touches exactly four columns of
``cocktails`` -- ``embedding``, ``preference_embedding``, ``embedding_3d`` and
``embedding_updated_at``. It never writes ``ingredients`` (still 32D) and never
writes any legacy backup table, because those are the only rollback path once
the in-place 48D cutover has destroyed the old 32D values.

Every gate runs before the commit; a dry run (no ``--commit``) performs the
whole sequence, including the write and the post-write verification, and then
rolls back.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, make_url

from app.sensory_embedding.vertex_batch import id_set_sha256
from app.vector_similarity.graph48 import canonical_cocktail_id, graph48_ids_sha256

EXPECTED_ROW_COUNT = 602
EXPECTED_DB_IDS_SHA256 = (
    "56e77646b60ad9b45cbdcd43f4807dde994ef40b1d5e4461dbfa41ca2d59c05f"
)
GRAPH48_DIMENSION = 48
PREFERENCE48_DIMENSION = 48
COORDINATE_DIMENSION = 3
UNIT_NORM_TOLERANCE = 1e-6
PREFERENCE_MIN = 0.0
PREFERENCE_MAX = 1.0
TOP5_K = 5
MANIFEST_NAME = "manifest.json"
GRAPH48_FILE = "graph48.csv"
PREFERENCE48_FILE = "preference48.csv"
TOP5_FILE = "graph48-directed-top5.csv"
CHUNK_SIZE = 1024

# The 10 leading provenance columns of graph48.csv / preference48.csv. The
# remaining columns are the vector itself, one per sensory axis.
VECTOR_CSV_METADATA_COLUMNS = (
    "cocktail_id",
    "registry_version",
    "registry_sha256",
    "source_sha256",
    "projection_ready_record_sha256",
    "teacher_lineage_sha256",
    "projection_provenance_sha256",
    "contract_version",
    "contract_sha256",
    "vector_sha256",
)

# The only columns this loader is ever allowed to assign.
WRITABLE_COLUMNS = (
    "embedding",
    "preference_embedding",
    "embedding_3d",
    "embedding_updated_at",
)


class SensoryLoadError(RuntimeError):
    """A gate failed; the load must not proceed or must be rolled back."""


# --- artifact reading -------------------------------------------------------


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 hex digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE * CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_integrity(artifact_dir: Path) -> dict[str, str]:
    """Re-hash every artifact file and compare it with ``manifest.json``."""

    manifest_path = artifact_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise SensoryLoadError(f"manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SensoryLoadError(f"cannot read manifest {manifest_path}: {error}")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise SensoryLoadError("manifest has no files section")

    present = sorted(
        item.name
        for item in artifact_dir.iterdir()
        if item.is_file() and item.name != MANIFEST_NAME
    )
    missing = sorted(set(declared) - set(present))
    if missing:
        raise SensoryLoadError(f"artifact files declared but absent: {missing}")
    undeclared = sorted(set(present) - set(declared))
    if undeclared:
        raise SensoryLoadError(f"artifact files present but undeclared: {undeclared}")

    digests: dict[str, str] = {}
    for name in present:
        expected = declared[name].get("sha256")
        actual = file_sha256(artifact_dir / name)
        if actual != expected:
            raise SensoryLoadError(
                f"artifact SHA-256 mismatch for {name}: "
                f"expected {expected}, recomputed {actual}"
            )
        digests[name] = actual
    return digests


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            header = next(reader, None)
            if header is None:
                raise SensoryLoadError(f"{path} is empty")
            return list(header), [row for row in reader if row]
    except (OSError, UnicodeError, csv.Error) as error:
        raise SensoryLoadError(f"cannot read {path}: {error}")


def _parse_floats(values: Sequence[str], *, label: str) -> tuple[float, ...]:
    parsed: list[float] = []
    for raw in values:
        try:
            number = float(raw)
        except ValueError:
            raise SensoryLoadError(f"{label} contains a non-numeric value: {raw!r}")
        if not math.isfinite(number):
            raise SensoryLoadError(f"{label} contains a non-finite value: {raw!r}")
        parsed.append(number)
    return tuple(parsed)


def read_vector_csv(path: Path, *, dimension: int) -> dict[str, tuple[float, ...]]:
    """Read a provenance-prefixed vector CSV keyed by canonical cocktail ID."""

    header, rows = _read_csv_rows(path)
    metadata_width = len(VECTOR_CSV_METADATA_COLUMNS)
    if tuple(header[:metadata_width]) != VECTOR_CSV_METADATA_COLUMNS:
        raise SensoryLoadError(f"{path.name} has an unexpected metadata header")
    if len(header) != metadata_width + dimension:
        raise SensoryLoadError(
            f"{path.name} must have {metadata_width + dimension} columns, "
            f"found {len(header)}"
        )
    vectors: dict[str, tuple[float, ...]] = {}
    for index, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise SensoryLoadError(f"{path.name} line {index} has a ragged row")
        cocktail_id = canonical_cocktail_id(row[0])
        if cocktail_id in vectors:
            raise SensoryLoadError(f"{path.name} repeats cocktail_id {cocktail_id}")
        vectors[cocktail_id] = _parse_floats(
            row[metadata_width:], label=f"{path.name} line {index}"
        )
    if not vectors:
        raise SensoryLoadError(f"{path.name} has no data rows")
    return vectors


def read_coordinates_csv(path: Path) -> dict[str, tuple[float, ...]]:
    """Read the ``cocktail_id,x,y,z`` sphere coordinate CSV."""

    header, rows = _read_csv_rows(path)
    if tuple(header) != ("cocktail_id", "x", "y", "z"):
        raise SensoryLoadError(f"{path.name} must have columns cocktail_id,x,y,z")
    coordinates: dict[str, tuple[float, ...]] = {}
    for index, row in enumerate(rows, start=2):
        if len(row) != 4:
            raise SensoryLoadError(f"{path.name} line {index} has a ragged row")
        cocktail_id = canonical_cocktail_id(row[0])
        if cocktail_id in coordinates:
            raise SensoryLoadError(f"{path.name} repeats cocktail_id {cocktail_id}")
        coordinates[cocktail_id] = _parse_floats(
            row[1:], label=f"{path.name} line {index}"
        )
    if not coordinates:
        raise SensoryLoadError(f"{path.name} has no data rows")
    return coordinates


def read_expected_top5(path: Path) -> dict[str, frozenset[str]]:
    """Read the offline directed top-5 neighbour sets keyed by source ID."""

    header, rows = _read_csv_rows(path)
    if tuple(header) != ("run_id", "source_id", "target_id", "rank", "cosine"):
        raise SensoryLoadError(f"{path.name} has an unexpected header")
    collected: dict[str, set[str]] = {}
    for index, row in enumerate(rows, start=2):
        if len(row) != 5:
            raise SensoryLoadError(f"{path.name} line {index} has a ragged row")
        source = canonical_cocktail_id(row[1])
        target = canonical_cocktail_id(row[2])
        if source == target:
            raise SensoryLoadError(f"{path.name} line {index} is a self edge")
        neighbours = collected.setdefault(source, set())
        if target in neighbours:
            raise SensoryLoadError(f"{path.name} repeats edge {source}->{target}")
        neighbours.add(target)
    for source, neighbours in collected.items():
        if len(neighbours) != TOP5_K:
            raise SensoryLoadError(
                f"{path.name} source {source} has {len(neighbours)} neighbours, "
                f"expected {TOP5_K}"
            )
    return {source: frozenset(targets) for source, targets in collected.items()}


# --- value gates ------------------------------------------------------------


def l2_norm(values: Sequence[float]) -> float:
    """Return the Euclidean norm of one vector."""

    return math.sqrt(math.fsum(value * value for value in values))


def assert_dimension(
    vectors: Mapping[str, Sequence[float]], *, dimension: int, label: str
) -> None:
    """Reject any vector whose width is not exactly ``dimension``."""

    wrong = sorted(
        (key for key, values in vectors.items() if len(values) != dimension), key=int
    )
    if wrong:
        raise SensoryLoadError(
            f"{label} has {len(wrong)} rows that are not {dimension}D "
            f"(first: {wrong[0]})"
        )


def assert_unit_norm(
    vectors: Mapping[str, Sequence[float]],
    *,
    label: str,
    tolerance: float = UNIT_NORM_TOLERANCE,
) -> float:
    """Reject any vector whose L2 norm is not 1 within ``tolerance``."""

    worst_id: str | None = None
    worst_error = 0.0
    for key, values in vectors.items():
        error = abs(l2_norm(values) - 1.0)
        if error > worst_error:
            worst_error, worst_id = error, key
    if worst_error > tolerance:
        raise SensoryLoadError(
            f"{label} is not unit-L2: cocktail {worst_id} deviates by {worst_error:.3e} "
            f"(tolerance {tolerance:.1e})"
        )
    return worst_error


def assert_unit_interval(
    vectors: Mapping[str, Sequence[float]], *, label: str
) -> tuple[float, float]:
    """Reject any value outside [0, 1]; ``Preference48`` is not normalized."""

    lowest = math.inf
    highest = -math.inf
    for key, values in vectors.items():
        for value in values:
            lowest = min(lowest, value)
            highest = max(highest, value)
            if value < PREFERENCE_MIN or value > PREFERENCE_MAX:
                raise SensoryLoadError(
                    f"{label} cocktail {key} has value {value!r} outside "
                    f"[{PREFERENCE_MIN}, {PREFERENCE_MAX}]"
                )
    return lowest, highest


def assert_identical_id_sets(named_sets: Mapping[str, Iterable[str]]) -> set[str]:
    """Require every named ID set to be identical and return it."""

    materialized = {name: set(values) for name, values in named_sets.items()}
    reference_name, reference = next(iter(materialized.items()))
    for name, values in materialized.items():
        if values == reference:
            continue
        only_here = sorted(values - reference, key=int)[:5]
        only_there = sorted(reference - values, key=int)[:5]
        raise SensoryLoadError(
            f"ID sets disagree: {name} vs {reference_name}; "
            f"only in {name}: {only_here}; only in {reference_name}: {only_there}"
        )
    return reference


# --- SQL --------------------------------------------------------------------


def cocktails_table() -> sa.TableClause:
    """Return the minimal ``cocktails`` projection this loader may touch."""

    return sa.table(
        "cocktails",
        sa.column("id"),
        sa.column("embedding"),
        sa.column("preference_embedding"),
        sa.column("embedding_3d"),
        sa.column("embedding_updated_at"),
    )


def build_update_statement() -> sa.Update:
    """Build the one UPDATE this loader issues, keyed by cocktail ID.

    The statement assigns exactly ``WRITABLE_COLUMNS`` and nothing else. Tests
    assert this against the compiled SQL so that no future edit can quietly add
    a fifth column, an ``ingredients`` write, or a backup-table write.
    """

    table = cocktails_table()
    return (
        sa.update(table)
        .where(table.c.id == sa.bindparam("target_id"))
        .values(
            embedding=sa.bindparam("embedding"),
            preference_embedding=sa.bindparam("preference_embedding"),
            embedding_3d=sa.bindparam("embedding_3d"),
            embedding_updated_at=sa.func.now(),
        )
    )


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


def _parse_vector_text(text: str) -> tuple[float, ...]:
    stripped = text.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        raise SensoryLoadError(f"unparseable vector from database: {text!r}")
    body = stripped[1:-1].strip()
    if not body:
        return ()
    return tuple(float(part) for part in body.split(","))


# --- database phases --------------------------------------------------------


def run_preflight(
    connection: Connection, *, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Read-only gates. Nothing is written before every one of these passes."""

    row_count = connection.execute(
        sa.text("SELECT count(*) FROM cocktails")
    ).scalar_one()
    if row_count != EXPECTED_ROW_COUNT:
        raise SensoryLoadError(
            f"cocktails has {row_count} rows, expected {EXPECTED_ROW_COUNT}"
        )

    db_ids = [
        int(value)
        for value in connection.execute(
            sa.text("SELECT id FROM cocktails ORDER BY id")
        ).scalars()
    ]

    # Two digests over the same IDs. `ids_sha256` encodes them as JSON integers;
    # `graph48_ids_sha256` encodes them as canonical decimal strings. They are
    # different values over identical input and are NOT interchangeable, so each
    # is checked against its own source of truth.
    db_ids_sha256 = id_set_sha256(db_ids)
    if db_ids_sha256 != EXPECTED_DB_IDS_SHA256:
        raise SensoryLoadError(
            f"cocktails ID-set digest is {db_ids_sha256}, "
            f"expected {EXPECTED_DB_IDS_SHA256}"
        )
    cohort = manifest.get("cohort")
    if not isinstance(cohort, dict):
        raise SensoryLoadError("manifest has no cohort section")
    if cohort.get("ids_sha256") != EXPECTED_DB_IDS_SHA256:
        raise SensoryLoadError(
            "manifest cohort.ids_sha256 does not match the pinned cohort digest"
        )
    expected_graph48_ids = cohort.get("graph48_ids_sha256")
    db_graph48_ids_sha256 = graph48_ids_sha256(db_ids)
    if db_graph48_ids_sha256 != expected_graph48_ids:
        raise SensoryLoadError(
            f"cocktails graph48 ID-set digest is {db_graph48_ids_sha256}, "
            f"expected {expected_graph48_ids}"
        )
    if db_ids_sha256 == db_graph48_ids_sha256:
        raise SensoryLoadError("the two cohort digests must not coincide")

    return {
        "cocktail_row_count": row_count,
        "db_ids_sha256": db_ids_sha256,
        "db_graph48_ids_sha256": db_graph48_ids_sha256,
        "db_ids": {canonical_cocktail_id(value) for value in db_ids},
    }


def write_vectors(
    connection: Connection,
    *,
    graph48: Mapping[str, Sequence[float]],
    preference48: Mapping[str, Sequence[float]],
    coordinates: Mapping[str, Sequence[float]],
) -> int:
    """Apply the four-column UPDATE for every cocktail in ID order."""

    statement = build_update_statement()
    parameters = [
        {
            "target_id": int(cocktail_id),
            "embedding": _vector_literal(graph48[cocktail_id]),
            "preference_embedding": _vector_literal(preference48[cocktail_id]),
            "embedding_3d": _vector_literal(coordinates[cocktail_id]),
        }
        for cocktail_id in sorted(graph48, key=int)
    ]
    result = connection.execute(statement, parameters)
    updated = result.rowcount
    if updated != len(parameters):
        raise SensoryLoadError(
            f"UPDATE reported {updated} rows, expected {len(parameters)}"
        )
    return updated


def verify_after_write(
    connection: Connection,
    *,
    graph48: Mapping[str, Sequence[float]],
    preference48: Mapping[str, Sequence[float]],
    coordinates: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Re-read every row inside the same transaction and re-check every gate."""

    rows = connection.execute(
        sa.text(
            "SELECT id, "
            "vector_dims(embedding) AS embedding_dims, "
            "vector_dims(preference_embedding) AS preference_dims, "
            "vector_dims(embedding_3d) AS coordinate_dims, "
            "embedding::text AS embedding_text, "
            "preference_embedding::text AS preference_text, "
            "embedding_3d::text AS coordinate_text, "
            "embedding_updated_at "
            "FROM cocktails ORDER BY id"
        )
    ).all()
    if len(rows) != EXPECTED_ROW_COUNT:
        raise SensoryLoadError(
            f"post-write cocktails has {len(rows)} rows, expected {EXPECTED_ROW_COUNT}"
        )

    read_back_graph48: dict[str, tuple[float, ...]] = {}
    read_back_coordinates: dict[str, tuple[float, ...]] = {}
    read_back_preference: dict[str, tuple[float, ...]] = {}
    for row in rows:
        cocktail_id = canonical_cocktail_id(int(row.id))
        for column, value in (
            ("embedding", row.embedding_text),
            ("preference_embedding", row.preference_text),
            ("embedding_3d", row.coordinate_text),
            ("embedding_updated_at", row.embedding_updated_at),
        ):
            if value is None:
                raise SensoryLoadError(
                    f"post-write cocktail {cocktail_id} has NULL {column}"
                )
        expected_dims = (
            ("embedding", row.embedding_dims, GRAPH48_DIMENSION),
            ("preference_embedding", row.preference_dims, PREFERENCE48_DIMENSION),
            ("embedding_3d", row.coordinate_dims, COORDINATE_DIMENSION),
        )
        for column, actual, expected in expected_dims:
            if actual != expected:
                raise SensoryLoadError(
                    f"post-write cocktail {cocktail_id} has {column} of "
                    f"{actual} dimensions, expected {expected}"
                )
        read_back_graph48[cocktail_id] = _parse_vector_text(row.embedding_text)
        read_back_preference[cocktail_id] = _parse_vector_text(row.preference_text)
        read_back_coordinates[cocktail_id] = _parse_vector_text(row.coordinate_text)

    expected_ids = set(graph48)
    if set(read_back_graph48) != expected_ids:
        raise SensoryLoadError("post-write cocktail IDs do not match the artifacts")

    graph48_error = assert_unit_norm(read_back_graph48, label="post-write embedding")
    coordinate_error = assert_unit_norm(
        read_back_coordinates, label="post-write embedding_3d"
    )
    preference_bounds = assert_unit_interval(
        read_back_preference, label="post-write preference_embedding"
    )
    # `vector` stores float4, so the read-back is the float32 rounding of the
    # artifact rather than a byte-identical copy. Bound that drift explicitly.
    drift = 0.0
    for cocktail_id in expected_ids:
        for source, target in (
            (graph48[cocktail_id], read_back_graph48[cocktail_id]),
            (preference48[cocktail_id], read_back_preference[cocktail_id]),
            (coordinates[cocktail_id], read_back_coordinates[cocktail_id]),
        ):
            drift = max(
                drift, max(abs(a - b) for a, b in zip(source, target, strict=True))
            )
    return {
        "rows": len(rows),
        "null_columns": 0,
        "dimensions": {
            "embedding": GRAPH48_DIMENSION,
            "preference_embedding": PREFERENCE48_DIMENSION,
            "embedding_3d": COORDINATE_DIMENSION,
        },
        "embedding_max_norm_error": graph48_error,
        "embedding_3d_max_norm_error": coordinate_error,
        "preference_embedding_min": preference_bounds[0],
        "preference_embedding_max": preference_bounds[1],
        "max_float4_roundtrip_drift": drift,
        "read_back_graph48": read_back_graph48,
    }


def check_top5_parity(
    connection: Connection,
    *,
    graph48: Mapping[str, Sequence[float]],
    expected_top5: Mapping[str, frozenset[str]],
) -> dict[str, Any]:
    """Compare exact in-database top-5 neighbours with the offline artifact.

    Index scans are disabled for the duration of the transaction so that the
    comparison is against exact cosine distance rather than an HNSW
    approximation. This is the check that actually proves the vectors landed in
    the right rows: a transposed or truncated load changes the neighbour sets.
    """

    connection.execute(sa.text("SET LOCAL enable_indexscan = off"))
    connection.execute(sa.text("SET LOCAL enable_bitmapscan = off"))
    neighbour_query = sa.text(
        "SELECT id FROM cocktails "
        "ORDER BY embedding <=> CAST(:query AS vector) "
        "LIMIT :limit"
    )
    sources = sorted(graph48, key=int)
    missing_sources = sorted(set(sources) - set(expected_top5), key=int)
    if missing_sources:
        raise SensoryLoadError(
            f"offline top-5 artifact is missing {len(missing_sources)} sources "
            f"(first: {missing_sources[0]})"
        )
    matched = 0
    mismatches: list[dict[str, Any]] = []
    for cocktail_id in sources:
        returned = [
            canonical_cocktail_id(int(value))
            for value in connection.execute(
                neighbour_query,
                {
                    "query": _vector_literal(graph48[cocktail_id]),
                    "limit": TOP5_K + 1,
                },
            ).scalars()
        ]
        neighbours = [value for value in returned if value != cocktail_id][:TOP5_K]
        if frozenset(neighbours) == expected_top5[cocktail_id]:
            matched += 1
            continue
        mismatches.append(
            {
                "source_id": cocktail_id,
                "database": sorted(neighbours, key=int),
                "artifact": sorted(expected_top5[cocktail_id], key=int),
            }
        )
    return {
        "sources": len(sources),
        "matched": matched,
        "mismatches": mismatches[:10],
        "mismatch_count": len(mismatches),
    }


# --- ledger and CLI ---------------------------------------------------------


def build_ledger(
    *,
    run_id: str,
    dsn_role: str,
    database: str,
    artifact_dir: Path,
    coordinates_path: Path,
    artifact_digests: Mapping[str, str],
    coordinates_sha256: str,
    manifest: Mapping[str, Any],
    preflight: Mapping[str, Any],
    row_counts: Mapping[str, int],
    verification: Mapping[str, Any],
    parity: Mapping[str, Any],
    committed: bool,
) -> dict[str, Any]:
    """Assemble the JSON ledger. It records the role name, never the DSN."""

    cohort = manifest["cohort"]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "recorded_at": datetime.now(UTC).isoformat(),
        "committed": committed,
        "database": {"role": dsn_role, "database": database},
        "artifact_dir": str(artifact_dir),
        "artifact_sha256": dict(sorted(artifact_digests.items())),
        "coordinates": {
            "path": str(coordinates_path),
            "sha256": coordinates_sha256,
        },
        "cohort_digests": {
            "ids_sha256": cohort["ids_sha256"],
            "graph48_ids_sha256": cohort["graph48_ids_sha256"],
            "db_ids_sha256": preflight["db_ids_sha256"],
            "db_graph48_ids_sha256": preflight["db_graph48_ids_sha256"],
        },
        "row_counts": dict(row_counts),
        "columns_written": list(WRITABLE_COLUMNS),
        "verification": {
            key: value
            for key, value in verification.items()
            if key != "read_back_graph48"
        },
        "top5_parity": {
            "sources": parity["sources"],
            "matched": parity["matched"],
            "mismatch_count": parity["mismatch_count"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI. ``--dsn`` is required and has no ambient fallback."""

    parser = argparse.ArgumentParser(
        description=(
            "Load 48D Graph48/Preference48 vectors and S2 sphere coordinates "
            "into cocktails. The target database is always explicit."
        )
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--dsn",
        required=True,
        help="Full SQLAlchemy/libpq DSN. Never read from the environment.",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit. Without it every gate still runs and the write is rolled back.",
    )
    parser.add_argument(
        "--confirm-write",
        default=None,
        help="Must repeat --run-id exactly. Required with --commit.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Ledger path. Defaults to a file beside --coordinates.",
    )
    return parser


def _resolve_confirmation(arguments: argparse.Namespace) -> None:
    if arguments.confirm_write is None:
        if arguments.commit:
            raise SensoryLoadError("--commit requires --confirm-write <run-id>")
        return
    if arguments.confirm_write != arguments.run_id:
        raise SensoryLoadError("--confirm-write does not equal --run-id")


def load(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run every gate, write, verify, and either commit or roll back."""

    _resolve_confirmation(arguments)
    artifact_dir = arguments.artifact_dir.expanduser().resolve()
    coordinates_path = arguments.coordinates.expanduser().resolve()

    artifact_digests = verify_artifact_integrity(artifact_dir)
    manifest = json.loads((artifact_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    print(f"artifact integrity   : {len(artifact_digests)} files re-hashed, all match")

    graph48 = read_vector_csv(artifact_dir / GRAPH48_FILE, dimension=GRAPH48_DIMENSION)
    preference48 = read_vector_csv(
        artifact_dir / PREFERENCE48_FILE, dimension=PREFERENCE48_DIMENSION
    )
    coordinates = read_coordinates_csv(coordinates_path)
    expected_top5 = read_expected_top5(artifact_dir / TOP5_FILE)
    coordinates_sha256 = file_sha256(coordinates_path)

    assert_dimension(graph48, dimension=GRAPH48_DIMENSION, label="graph48")
    assert_dimension(
        preference48, dimension=PREFERENCE48_DIMENSION, label="preference48"
    )
    assert_dimension(coordinates, dimension=COORDINATE_DIMENSION, label="coordinates")
    graph48_norm_error = assert_unit_norm(graph48, label="graph48")
    coordinate_norm_error = assert_unit_norm(coordinates, label="coordinates")
    preference_bounds = assert_unit_interval(preference48, label="preference48")
    print(
        f"graph48              : {len(graph48)} rows, "
        f"max |norm-1| = {graph48_norm_error:.3e}"
    )
    print(
        f"coordinates          : {len(coordinates)} rows, "
        f"max |norm-1| = {coordinate_norm_error:.3e}"
    )
    print(
        f"preference48         : {len(preference48)} rows, "
        f"values in [{preference_bounds[0]:.6g}, {preference_bounds[1]:.6g}] "
        f"(unnormalized by design)"
    )

    url = make_url(arguments.dsn)
    engine = sa.create_engine(url, future=True)
    committed = False
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                preflight = run_preflight(connection, manifest=manifest)
                print(
                    f"preflight rows       : {preflight['cocktail_row_count']} "
                    f"(expected {EXPECTED_ROW_COUNT})"
                )
                print(f"preflight ids_sha256 : {preflight['db_ids_sha256']}")
                print(f"preflight graph48_ids: {preflight['db_graph48_ids_sha256']}")

                shared_ids = assert_identical_id_sets(
                    {
                        "graph48": set(graph48),
                        "preference48": set(preference48),
                        "coordinates": set(coordinates),
                        "database": preflight["db_ids"],
                    }
                )
                print(f"cross-file id sets   : identical, {len(shared_ids)} ids")

                updated = write_vectors(
                    connection,
                    graph48=graph48,
                    preference48=preference48,
                    coordinates=coordinates,
                )
                print(
                    f"update               : {updated} rows, columns "
                    f"{', '.join(WRITABLE_COLUMNS)}"
                )

                verification = verify_after_write(
                    connection,
                    graph48=graph48,
                    preference48=preference48,
                    coordinates=coordinates,
                )
                print(
                    f"post-write rows      : {verification['rows']}, "
                    f"nulls {verification['null_columns']}, dims "
                    f"{verification['dimensions']['embedding']}/"
                    f"{verification['dimensions']['preference_embedding']}/"
                    f"{verification['dimensions']['embedding_3d']}"
                )
                print(
                    f"post-write norms     : embedding max |norm-1| = "
                    f"{verification['embedding_max_norm_error']:.3e}, "
                    f"embedding_3d max |norm-1| = "
                    f"{verification['embedding_3d_max_norm_error']:.3e}"
                )
                print(
                    f"post-write preference: "
                    f"[{verification['preference_embedding_min']:.6g}, "
                    f"{verification['preference_embedding_max']:.6g}]"
                )

                parity = check_top5_parity(
                    connection, graph48=graph48, expected_top5=expected_top5
                )
                print(f"top-5 parity         : {parity['matched']}/{parity['sources']}")
                if parity["matched"] != parity["sources"]:
                    for mismatch in parity["mismatches"]:
                        print(
                            f"  MISMATCH source {mismatch['source_id']}: "
                            f"db {mismatch['database']} != "
                            f"artifact {mismatch['artifact']}",
                            file=sys.stderr,
                        )
                    raise SensoryLoadError(
                        f"top-5 parity failed for {parity['mismatch_count']} of "
                        f"{parity['sources']} sources"
                    )

                if arguments.commit:
                    transaction.commit()
                    committed = True
                else:
                    transaction.rollback()
            except BaseException:
                if transaction.is_active:
                    transaction.rollback()
                raise
    finally:
        engine.dispose()

    ledger = build_ledger(
        run_id=arguments.run_id,
        dsn_role=url.username or "",
        database=url.database or "",
        artifact_dir=artifact_dir,
        coordinates_path=coordinates_path,
        artifact_digests=artifact_digests,
        coordinates_sha256=coordinates_sha256,
        manifest=manifest,
        preflight=preflight,
        row_counts={
            "graph48": len(graph48),
            "preference48": len(preference48),
            "coordinates": len(coordinates),
            "cocktails": preflight["cocktail_row_count"],
            "updated": updated,
        },
        verification=verification,
        parity=parity,
        committed=committed,
    )
    if committed:
        ledger_path = arguments.ledger or coordinates_path.with_name(
            f"load-sensory-48d-ledger-{arguments.run_id}.json"
        )
        ledger_path.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"ledger               : {ledger_path}")
    else:
        print("ledger               : not written (dry run rolled back)")
    print(f"result               : {'COMMITTED' if committed else 'ROLLED BACK'}")
    return ledger


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success and 2 on any failed gate."""

    arguments = build_parser().parse_args(argv)
    try:
        load(arguments)
    except SensoryLoadError as error:
        print(f"load_sensory_48d: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
