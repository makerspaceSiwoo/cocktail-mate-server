"""Gate tests for the 48D sensory loader.

These cover the failure modes that would silently corrupt production: an ID
normalized the wrong way, the two cohort digests confused with each other, a
non-unit Graph48 vector, a Preference48 value outside [0, 1], and an UPDATE
that grew a fifth column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.dialects import postgresql

from app.sensory_embedding.vertex_batch import id_set_sha256
from app.vector_similarity.graph48 import canonical_cocktail_id, graph48_ids_sha256
from scripts.load_sensory_48d import (
    EXPECTED_DB_IDS_SHA256,
    EXPECTED_ROW_COUNT,
    GRAPH48_DIMENSION,
    VECTOR_CSV_METADATA_COLUMNS,
    WRITABLE_COLUMNS,
    SensoryLoadError,
    _resolve_confirmation,
    assert_identical_id_sets,
    assert_unit_interval,
    assert_unit_norm,
    build_update_statement,
    read_coordinates_csv,
    read_expected_top5,
    read_vector_csv,
    run_preflight,
    verify_artifact_integrity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "sensory-batch" / "run-20260806-full602-v1" / "artifacts"

EXPECTED_UPDATE_SQL = (
    "UPDATE cocktails SET "
    "embedding=%(embedding)s, "
    "preference_embedding=%(preference_embedding)s, "
    "embedding_3d=%(embedding_3d)s, "
    "embedding_updated_at=now() "
    "WHERE cocktails.id = %(target_id)s"
)


# --- helpers ----------------------------------------------------------------


def _unit_vector(dimension: int = GRAPH48_DIMENSION) -> tuple[float, ...]:
    return (1.0,) + (0.0,) * (dimension - 1)


def _write_vector_csv(
    path: Path, rows: dict[str, tuple[float, ...]], *, dimension: int
) -> None:
    header = list(VECTOR_CSV_METADATA_COLUMNS) + [
        f"axis_{index}" for index in range(dimension)
    ]
    lines = [",".join(header)]
    for cocktail_id, values in rows.items():
        metadata = [cocktail_id] + ["x"] * (len(VECTOR_CSV_METADATA_COLUMNS) - 1)
        lines.append(",".join(metadata + [repr(value) for value in values]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _StubResult:
    def __init__(self, values: list[Any]) -> None:
        self._values = values

    def scalar_one(self) -> Any:
        return self._values[0]

    def scalars(self) -> list[Any]:
        return list(self._values)


class _StubConnection:
    """The read-only surface ``run_preflight`` uses, and nothing more."""

    def __init__(self, *, row_count: int, ids: list[int]) -> None:
        self.row_count = row_count
        self.ids = ids
        self.statements: list[str] = []

    def execute(self, statement: Any, parameters: Any = None) -> _StubResult:
        sql = str(statement)
        self.statements.append(sql)
        if "count(*)" in sql:
            return _StubResult([self.row_count])
        if "SELECT id FROM cocktails" in sql:
            return _StubResult(sorted(self.ids))
        raise AssertionError(f"unexpected preflight statement: {sql}")


def _cohort_ids() -> list[int]:
    coordinates = ARTIFACT_DIR / "graph48.csv"
    if not coordinates.is_file():
        pytest.skip("graph48 artifact is not present in this checkout")
    return [int(value) for value in read_vector_csv(coordinates, dimension=48)]


# --- ID normalization -------------------------------------------------------


def test_canonical_ids_agree_for_integers_and_decimal_strings() -> None:
    assert canonical_cocktail_id(619) == canonical_cocktail_id("619") == "619"


@pytest.mark.parametrize("value", ["01", " 1", "0", "-3", "1.0", "", "abc"])
def test_canonical_ids_reject_non_canonical_spellings(value: str) -> None:
    with pytest.raises(Exception):
        canonical_cocktail_id(value)


def test_vector_csv_keys_are_canonical_regardless_of_source_spelling(
    tmp_path: Path,
) -> None:
    path = tmp_path / "graph48.csv"
    _write_vector_csv(
        path,
        {"7": _unit_vector(), "42": _unit_vector()},
        dimension=GRAPH48_DIMENSION,
    )

    vectors = read_vector_csv(path, dimension=GRAPH48_DIMENSION)

    assert set(vectors) == {"7", "42"}
    assert all(isinstance(key, str) for key in vectors)
    # An integer key from another source normalizes onto the same bucket.
    assert canonical_cocktail_id(7) in vectors


def test_vector_csv_rejects_a_non_canonical_id(tmp_path: Path) -> None:
    path = tmp_path / "graph48.csv"
    _write_vector_csv(path, {"007": _unit_vector()}, dimension=GRAPH48_DIMENSION)

    with pytest.raises(Exception):
        read_vector_csv(path, dimension=GRAPH48_DIMENSION)


def test_coordinates_csv_keys_are_canonical(tmp_path: Path) -> None:
    path = tmp_path / "coordinates.csv"
    path.write_text("cocktail_id,x,y,z\n7,1.0,0.0,0.0\n", encoding="utf-8")

    coordinates = read_coordinates_csv(path)

    assert coordinates == {"7": (1.0, 0.0, 0.0)}


# --- the two cohort digests -------------------------------------------------


def test_the_two_cohort_digests_are_different_functions_of_the_same_ids() -> None:
    ids = [3, 1, 2]

    json_integer_digest = id_set_sha256(ids)
    decimal_string_digest = graph48_ids_sha256(ids)

    assert json_integer_digest != decimal_string_digest
    # Both are order-independent, so the difference is the encoding, not sorting.
    assert id_set_sha256([1, 2, 3]) == json_integer_digest
    assert graph48_ids_sha256(["1", "2", "3"]) == decimal_string_digest


def test_pinned_cohort_digests_are_reproduced_from_the_artifact_ids() -> None:
    ids = _cohort_ids()
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert len(ids) == EXPECTED_ROW_COUNT
    assert id_set_sha256(ids) == EXPECTED_DB_IDS_SHA256
    assert id_set_sha256(ids) == manifest["cohort"]["ids_sha256"]
    assert graph48_ids_sha256(ids) == manifest["cohort"]["graph48_ids_sha256"]
    assert manifest["cohort"]["ids_sha256"] != manifest["cohort"]["graph48_ids_sha256"]


def test_preflight_accepts_the_pinned_cohort() -> None:
    ids = _cohort_ids()
    manifest = {
        "cohort": {
            "ids_sha256": EXPECTED_DB_IDS_SHA256,
            "graph48_ids_sha256": graph48_ids_sha256(ids),
        }
    }

    report = run_preflight(
        _StubConnection(row_count=EXPECTED_ROW_COUNT, ids=ids), manifest=manifest
    )

    assert report["cocktail_row_count"] == EXPECTED_ROW_COUNT
    assert report["db_ids_sha256"] == EXPECTED_DB_IDS_SHA256
    assert report["db_graph48_ids_sha256"] == graph48_ids_sha256(ids)
    assert report["db_ids_sha256"] != report["db_graph48_ids_sha256"]


def test_preflight_rejects_the_digests_used_interchangeably() -> None:
    ids = _cohort_ids()
    swapped = {
        "cohort": {
            "ids_sha256": EXPECTED_DB_IDS_SHA256,
            # The JSON-integer digest pasted where the decimal-string one belongs.
            "graph48_ids_sha256": EXPECTED_DB_IDS_SHA256,
        }
    }

    with pytest.raises(SensoryLoadError, match="graph48 ID-set digest"):
        run_preflight(
            _StubConnection(row_count=EXPECTED_ROW_COUNT, ids=ids), manifest=swapped
        )


def test_preflight_rejects_a_cohort_that_is_not_the_pinned_one() -> None:
    ids = list(range(1, EXPECTED_ROW_COUNT + 1))
    manifest = {
        "cohort": {
            "ids_sha256": EXPECTED_DB_IDS_SHA256,
            "graph48_ids_sha256": graph48_ids_sha256(ids),
        }
    }

    with pytest.raises(SensoryLoadError, match="ID-set digest"):
        run_preflight(
            _StubConnection(row_count=EXPECTED_ROW_COUNT, ids=ids), manifest=manifest
        )


def test_preflight_rejects_a_wrong_row_count() -> None:
    with pytest.raises(SensoryLoadError, match="expected 602"):
        run_preflight(
            _StubConnection(row_count=601, ids=[1]),
            manifest={"cohort": {"ids_sha256": EXPECTED_DB_IDS_SHA256}},
        )


# --- value gates ------------------------------------------------------------


def test_unit_norm_gate_accepts_deviation_inside_the_tolerance() -> None:
    inside = (1.0 - 5e-7,) + (0.0,) * (GRAPH48_DIMENSION - 1)

    error = assert_unit_norm({"1": inside}, label="graph48")

    assert error <= 1e-6


def test_unit_norm_gate_rejects_deviation_outside_the_tolerance() -> None:
    outside = (1.0 - 1e-5,) + (0.0,) * (GRAPH48_DIMENSION - 1)

    with pytest.raises(SensoryLoadError, match="not unit-L2"):
        assert_unit_norm({"9": outside}, label="graph48")


def test_unit_norm_gate_names_the_offending_cocktail() -> None:
    with pytest.raises(SensoryLoadError, match="cocktail 42"):
        assert_unit_norm(
            {"1": _unit_vector(), "42": (2.0,) + (0.0,) * (GRAPH48_DIMENSION - 1)},
            label="coordinates",
        )


@pytest.mark.parametrize("bad", [-1e-9, 1.0000001, 5.0])
def test_unit_interval_gate_rejects_values_outside_zero_one(bad: float) -> None:
    with pytest.raises(SensoryLoadError, match="outside"):
        assert_unit_interval({"3": (0.5, bad)}, label="preference48")


def test_unit_interval_gate_accepts_the_closed_interval() -> None:
    assert assert_unit_interval({"3": (0.0, 1.0, 0.25)}, label="preference48") == (
        0.0,
        1.0,
    )


def test_preference_vectors_are_never_required_to_be_unit_norm() -> None:
    # Preference48 is compared by inner product and is unnormalized by design.
    preference = {"3": (1.0,) * GRAPH48_DIMENSION}

    assert assert_unit_interval(preference, label="preference48") == (1.0, 1.0)
    with pytest.raises(SensoryLoadError):
        assert_unit_norm(preference, label="if this were graph48")


# --- cross-file agreement ---------------------------------------------------


def test_identical_id_sets_pass_and_return_the_shared_set() -> None:
    assert assert_identical_id_sets(
        {"graph48": {"1", "2"}, "database": {"2", "1"}}
    ) == {"1", "2"}


def test_disagreeing_id_sets_are_rejected_with_both_differences() -> None:
    with pytest.raises(SensoryLoadError, match="ID sets disagree"):
        assert_identical_id_sets(
            {"graph48": {"1", "2"}, "coordinates": {"1", "3"}},
        )


# --- artifact integrity -----------------------------------------------------


def test_artifact_integrity_detects_a_tampered_file(tmp_path: Path) -> None:
    payload = tmp_path / "graph48.csv"
    payload.write_text("original\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": {"graph48.csv": {"sha256": "0" * 64}}}),
        encoding="utf-8",
    )

    with pytest.raises(SensoryLoadError, match="SHA-256 mismatch"):
        verify_artifact_integrity(tmp_path)


def test_artifact_integrity_detects_an_undeclared_extra_file(tmp_path: Path) -> None:
    payload = tmp_path / "graph48.csv"
    payload.write_text("original\n", encoding="utf-8")
    (tmp_path / "stowaway.csv").write_text("surprise\n", encoding="utf-8")
    digest = "3a4f2c40e2f0f2a2eea9c74e0f2f0a1b" + "0" * 32
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": {"graph48.csv": {"sha256": digest}}}),
        encoding="utf-8",
    )

    with pytest.raises(SensoryLoadError, match="present but undeclared"):
        verify_artifact_integrity(tmp_path)


def test_expected_top5_requires_exactly_five_neighbours(tmp_path: Path) -> None:
    path = tmp_path / "graph48-directed-top5.csv"
    path.write_text(
        "run_id,source_id,target_id,rank,cosine\nr,1,2,1,0.9\n", encoding="utf-8"
    )

    with pytest.raises(SensoryLoadError, match="expected 5"):
        read_expected_top5(path)


# --- the write statement ----------------------------------------------------


def test_update_statement_touches_only_the_four_intended_columns() -> None:
    statement = build_update_statement()

    sql = str(statement.compile(dialect=postgresql.dialect()))
    assignments = sql.split(" SET ", 1)[1].split(" WHERE ", 1)[0]
    assigned = [part.split("=", 1)[0].strip() for part in assignments.split(", ")]

    assert sql == EXPECTED_UPDATE_SQL
    assert statement.table.name == "cocktails"
    assert tuple(assigned) == WRITABLE_COLUMNS
    assert len(assigned) == 4


def test_update_statement_never_mentions_ingredients_or_a_backup_table() -> None:
    sql = str(build_update_statement().compile(dialect=postgresql.dialect()))

    assert "ingredients" not in sql
    assert "backup" not in sql
    assert "legacy" not in sql
    assert "taste_descriptors" not in sql
    assert sql.count("UPDATE") == 1


def test_update_statement_is_keyed_by_a_bound_cocktail_id() -> None:
    sql = str(build_update_statement().compile(dialect=postgresql.dialect()))

    assert "WHERE cocktails.id = %(target_id)s" in sql


# --- confirmation -----------------------------------------------------------


def test_commit_requires_a_confirmation_that_repeats_the_run_id() -> None:
    arguments = argparse.Namespace(
        commit=True, confirm_write="other-run", run_id="run-a"
    )

    with pytest.raises(SensoryLoadError, match="does not equal"):
        _resolve_confirmation(arguments)


def test_commit_without_any_confirmation_is_refused() -> None:
    arguments = argparse.Namespace(commit=True, confirm_write=None, run_id="run-a")

    with pytest.raises(SensoryLoadError, match="requires --confirm-write"):
        _resolve_confirmation(arguments)


def test_dry_run_needs_no_confirmation() -> None:
    arguments = argparse.Namespace(commit=False, confirm_write=None, run_id="run-a")

    assert _resolve_confirmation(arguments) is None
