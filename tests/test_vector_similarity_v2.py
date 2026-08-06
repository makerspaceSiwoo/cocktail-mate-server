from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from app.vector_similarity import (
    CANONICAL_RUN_CSV_FIELDS,
    DIRECTED_NEIGHBOR_CSV_FIELDS,
    GRAPH48_DIMENSION,
    GRAPH48_K,
    UNION_EDGE_CSV_FIELDS,
    AnnBackendPlaceholder,
    AnnBackendUnavailableError,
    CanonicalGraph48Artifact,
    SimilarityConfigurationError,
    VectorValidationError,
    build_graph48_artifact,
    graph48_all_pairs,
    graph48_ids_sha256,
    graph48_vector_sha256,
    preference48_mips,
    read_canonical_graph48_csv,
    write_canonical_graph48_csv,
)


@dataclass(frozen=True)
class _Record:
    id: int | str
    values: Sequence[float]


def _unit(cocktail_id: int | str, *head: float) -> _Record:
    values = tuple(head) + (0.0,) * (GRAPH48_DIMENSION - len(head))
    return _Record(cocktail_id, values)


class Graph48SimilarityTests(unittest.TestCase):
    def test_exact_all_pairs_are_numeric_sorted_unit_vector_dots(self) -> None:
        inverse_root_two = 1.0 / math.sqrt(2.0)
        records = (
            _unit("10", 0.0, 1.0),
            _unit(2, inverse_root_two, inverse_root_two),
            _unit("1", 1.0, 0.0),
        )

        pairs = graph48_all_pairs(records)

        self.assertEqual(
            [(row.a_id, row.b_id) for row in pairs],
            [("1", "2"), ("1", "10"), ("2", "10")],
        )
        self.assertAlmostEqual(pairs[0].cosine, inverse_root_two, places=15)
        self.assertEqual(pairs[1].cosine, 0.0)
        self.assertAlmostEqual(pairs[2].cosine, inverse_root_two, places=15)

    def test_graph48_rejects_non_unit_wrong_dimension_and_noncanonical_id(
        self,
    ) -> None:
        for record, message in (
            (_Record("01", (1.0,) + (0.0,) * 47), "canonical decimal"),
            (_Record(1, (1.0,) + (0.0,) * 46), "48 dimensions"),
            (_Record(1, (0.5,) + (0.0,) * 47), "unit L2"),
        ):
            with self.subTest(record=record):
                with self.assertRaisesRegex(VectorValidationError, message):
                    graph48_all_pairs((record,))

    def test_canonical_top_five_and_either_direction_union_are_one_source(
        self,
    ) -> None:
        records = tuple(
            _unit(cocktail_id, 1.0) for cocktail_id in (10, 6, 5, 4, 3, 2, 1)
        )
        artifact = build_graph48_artifact(records, run_id="teacher-run-20260806")

        self.assertEqual(artifact.run.dimension, 48)
        self.assertEqual(artifact.run.k, 5)
        self.assertEqual(artifact.cocktail_ids, ("1", "2", "3", "4", "5", "6", "10"))
        self.assertEqual(
            artifact.run.ids_sha256,
            graph48_ids_sha256(artifact.cocktail_ids),
        )
        self.assertEqual(
            artifact.run.vector_sha256,
            graph48_vector_sha256(tuple(reversed(records))),
        )
        self.assertEqual(len(artifact.directed_neighbors), len(records) * 5)
        self.assertEqual(
            [row.target_id for row in artifact.recommendations_for("1")],
            ["2", "3", "4", "5", "6"],
        )
        self.assertEqual(
            [row.target_id for row in artifact.recommendations_for(10)],
            ["1", "2", "3", "4", "5"],
        )

        edge = next(
            row for row in artifact.graph_edges() if (row.a_id, row.b_id) == ("1", "10")
        )
        self.assertIsNone(edge.a_rank)
        self.assertEqual(edge.b_rank, 1)
        self.assertEqual(edge.cosine, 1.0)
        self.assertNotIn(edge, artifact.recommendations_for(1))

    def test_vector_hash_is_local_and_source_artifact_hash_is_separate(self) -> None:
        records = tuple(_unit(cocktail_id, 1.0) for cocktail_id in range(1, 8))
        source_sha256 = "f" * 64

        artifact = build_graph48_artifact(
            records,
            run_id="local-hash-v1",
            source_artifact_sha256=source_sha256,
        )

        self.assertEqual(
            artifact.run.vector_sha256,
            graph48_vector_sha256(records),
        )
        self.assertNotEqual(artifact.run.vector_sha256, source_sha256)
        self.assertEqual(artifact.source_artifact_sha256, source_sha256)
        with self.assertRaisesRegex(
            VectorValidationError,
            "source_artifact_sha256",
        ):
            build_graph48_artifact(
                records,
                run_id="bad-source-hash-v1",
                source_artifact_sha256="not-a-hash",
            )

    def test_artifact_rejects_rank_union_and_run_tampering(self) -> None:
        artifact = build_graph48_artifact(
            tuple(_unit(cocktail_id, 1.0) for cocktail_id in range(1, 8)),
            run_id="run-v1",
        )
        rows = list(artifact.directed_neighbors)
        rows[0] = replace(rows[0], run_id="other-run")
        with self.assertRaisesRegex(VectorValidationError, "run_id"):
            CanonicalGraph48Artifact(
                run=artifact.run,
                cocktail_ids=artifact.cocktail_ids,
                directed_neighbors=tuple(rows),
                union_edges=artifact.union_edges,
            )

        edge = artifact.union_edges[0]
        edges = (replace(edge, cosine=0.5),) + artifact.union_edges[1:]
        with self.assertRaisesRegex(VectorValidationError, "identical cosine"):
            CanonicalGraph48Artifact(
                run=artifact.run,
                cocktail_ids=artifact.cocktail_ids,
                directed_neighbors=artifact.directed_neighbors,
                union_edges=edges,
            )

    def test_preference48_mips_masks_zero_query_axes_and_never_builds_edges(
        self,
    ) -> None:
        query = (0.5, 0.0, 0.5) + (0.0,) * 45
        records = (
            _Record(10, (0.5, 1000.0, 0.5) + (0.0,) * 45),
            _Record(2, (0.8, -1000.0, 0.1) + (0.0,) * 45),
            _Record(1, (0.9, 0.0, 0.9) + (0.0,) * 45),
        )

        matches = preference48_mips(query, records, k=3)

        self.assertEqual(
            [(row.cocktail_id, row.rank, row.score) for row in matches],
            [("1", 1, 0.9), ("10", 2, 0.5), ("2", 3, 0.45)],
        )
        self.assertFalse(any(hasattr(row, "cosine") for row in matches))
        self.assertFalse(any(hasattr(row, "a_rank") for row in matches))

    def test_csv_bundle_is_locked_create_only_and_exact_on_readback(self) -> None:
        artifact = build_graph48_artifact(
            tuple(_unit(cocktail_id, 1.0) for cocktail_id in range(1, 8)),
            run_id="run-csv-v1",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_path = root / "run.csv"
            directed_path = root / "directed.csv"
            union_path = root / "union.csv"

            write_canonical_graph48_csv(
                artifact,
                run_path=run_path,
                directed_path=directed_path,
                union_path=union_path,
            )

            self.assertEqual(
                run_path.read_text(encoding="utf-8").splitlines()[0],
                ",".join(CANONICAL_RUN_CSV_FIELDS),
            )
            self.assertEqual(
                directed_path.read_text(encoding="utf-8").splitlines()[0],
                ",".join(DIRECTED_NEIGHBOR_CSV_FIELDS),
            )
            self.assertEqual(
                union_path.read_text(encoding="utf-8").splitlines()[0],
                ",".join(UNION_EDGE_CSV_FIELDS),
            )
            loaded = read_canonical_graph48_csv(
                run_path=run_path,
                directed_path=directed_path,
                union_path=union_path,
                expected_run_id=artifact.run.run_id,
                expected_vector_sha256=artifact.run.vector_sha256,
            )
            self.assertEqual(loaded, artifact)

            with self.assertRaises(FileExistsError):
                write_canonical_graph48_csv(
                    artifact,
                    run_path=run_path,
                    directed_path=directed_path,
                    union_path=union_path,
                )

    def test_csv_reader_rejects_mixed_run_ids_and_vector_hash_mismatch(self) -> None:
        artifact = build_graph48_artifact(
            tuple(_unit(cocktail_id, 1.0) for cocktail_id in range(1, 8)),
            run_id="run-csv-v2",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_path = root / "run.csv"
            directed_path = root / "directed.csv"
            union_path = root / "union.csv"
            write_canonical_graph48_csv(
                artifact,
                run_path=run_path,
                directed_path=directed_path,
                union_path=union_path,
            )
            text = directed_path.read_text(encoding="utf-8")
            directed_path.write_text(
                text.replace("run-csv-v2", "other-run", 1),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(VectorValidationError, "one run_id"):
                read_canonical_graph48_csv(
                    run_path=run_path,
                    directed_path=directed_path,
                    union_path=union_path,
                )
            with self.assertRaisesRegex(VectorValidationError, "vector_sha256"):
                read_canonical_graph48_csv(
                    run_path=run_path,
                    directed_path=directed_path,
                    union_path=union_path,
                    expected_vector_sha256="f" * 64,
                )

    def test_exact_baseline_handles_602_and_ann_never_runs(self) -> None:
        records = tuple(_unit(cocktail_id, 1.0) for cocktail_id in range(1, 603))

        artifact = build_graph48_artifact(records, run_id="all-602-v1")

        self.assertEqual(len(artifact.directed_neighbors), 602 * GRAPH48_K)
        self.assertEqual(
            [row.target_id for row in artifact.recommendations_for(602)],
            ["1", "2", "3", "4", "5"],
        )
        with self.assertRaisesRegex(AnnBackendUnavailableError, "602"):
            AnnBackendPlaceholder().directed_top_k(records)

    def test_fixed_top_five_requires_at_least_six_records(self) -> None:
        with self.assertRaises(SimilarityConfigurationError):
            build_graph48_artifact(
                tuple(_unit(cocktail_id, 1.0) for cocktail_id in range(1, 6)),
                run_id="too-small-v1",
            )


if __name__ == "__main__":
    unittest.main()
