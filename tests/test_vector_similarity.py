from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from typing import Sequence

from app.vector_similarity import (
    AnnBackendPlaceholder,
    AnnBackendUnavailableError,
    CANONICAL_NEIGHBOR_SCHEMA_VERSION,
    EXACT_COSINE_PROVIDER_ID,
    CanonicalNeighborArtifact,
    DenseVector,
    ExactCosineBackend,
    NeighborSearchBackend,
    SimilarityConfigurationError,
    VectorValidationError,
    all_pair_similarities,
    build_canonical_neighbor_artifact,
    build_union_graph,
    cosine_similarity,
    directed_top_k,
    exact_top_k_union_graph,
    negative_inner_product,
    positive_score_from_distance,
    rank_by_inner_product,
    validate_records,
)


def _vector(id: int, *values: float) -> DenseVector:
    return DenseVector.from_values(id, values)


@dataclass(frozen=True)
class _NeutralRecord:
    id: int
    values: Sequence[float]


class VectorSimilarityTests(unittest.TestCase):
    def test_dense_vector_accepts_neutral_sequences_and_rejects_nonfinite(self) -> None:
        vector = DenseVector.from_values(7, [1, 2.5, -3])
        self.assertEqual(vector.values, (1.0, 2.5, -3.0))

        for values in ([], [math.nan], [math.inf], [True]):
            with self.subTest(values=values):
                with self.assertRaises(VectorValidationError):
                    DenseVector.from_values(1, values)
        with self.assertRaises(VectorValidationError):
            DenseVector.from_values(True, [1.0])

    def test_cosine_zero_dimension_and_nonfinite_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(VectorValidationError, "zero vector"):
            cosine_similarity((0.0, 0.0), (1.0, 0.0))
        with self.assertRaisesRegex(VectorValidationError, "dimensions"):
            cosine_similarity((1.0,), (1.0, 2.0))
        with self.assertRaisesRegex(VectorValidationError, "finite"):
            cosine_similarity((math.nan,), (1.0,))

        with self.assertRaisesRegex(VectorValidationError, "zero vector"):
            all_pair_similarities([_vector(1, 0.0), _vector(2, 1.0)])

    def test_cosine_is_exact_and_stable_for_extreme_finite_values(self) -> None:
        self.assertEqual(cosine_similarity((3.0, 4.0), (3.0, 4.0)), 1.0)
        self.assertEqual(cosine_similarity((1.0, 0.0), (-1.0, 0.0)), -1.0)
        self.assertEqual(cosine_similarity((1.0, 0.0), (0.0, 2.0)), 0.0)
        self.assertTrue(
            math.isclose(
                cosine_similarity((1e308, 1e308), (1e308, -1e308)),
                0.0,
                abs_tol=1e-15,
            )
        )
        self.assertEqual(
            cosine_similarity((5e-324, 0.0), (5e-324, 0.0)),
            1.0,
        )

    def test_negative_inner_product_converts_to_higher_is_better_score(self) -> None:
        distance = negative_inner_product((1.0, 0.0), (0.6, 0.8))

        self.assertTrue(math.isclose(distance, -0.6))
        self.assertTrue(math.isclose(positive_score_from_distance(distance), 0.6))
        with self.assertRaises(VectorValidationError):
            positive_score_from_distance(math.inf)

    def test_collection_validation_rejects_duplicate_ids_and_dimension_mismatch(
        self,
    ) -> None:
        with self.assertRaisesRegex(VectorValidationError, "duplicate"):
            validate_records([_vector(1, 1.0), _vector(1, 2.0)])
        with self.assertRaisesRegex(VectorValidationError, "dimensions"):
            validate_records([_vector(1, 1.0), _vector(2, 1.0, 2.0)])

        records = [
            _NeutralRecord(id=2, values=[0.0, 1.0]),
            _NeutralRecord(id=1, values=[1.0, 0.0]),
        ]
        self.assertEqual(
            [record.id for record in validate_records(records)],
            [1, 2],
        )

    def test_all_pairs_are_complete_and_deterministically_ordered(self) -> None:
        pairs = all_pair_similarities(
            [
                _vector(3, -1.0, 0.0),
                _vector(1, 1.0, 0.0),
                _vector(2, 0.0, 1.0),
            ]
        )

        self.assertEqual(
            [(pair.left_id, pair.right_id) for pair in pairs],
            [(1, 2), (1, 3), (2, 3)],
        )
        self.assertEqual([pair.score for pair in pairs], [0.0, -1.0, 0.0])
        for pair in pairs:
            self.assertFalse(hasattr(pair, "negative_inner_product"))

    def test_raw_inner_product_query_matches_pgvector_distance_semantics(
        self,
    ) -> None:
        matches = rank_by_inner_product(
            (2.0, 0.0),
            [
                _vector(4, 1.0, 0.0),
                _vector(2, -1.0, 0.0),
                _vector(3, 3.0, 0.0),
                _vector(1, 1.0, 0.0),
            ],
            k=4,
        )

        self.assertEqual([match.id for match in matches], [3, 1, 4, 2])
        self.assertEqual(
            [match.negative_inner_product for match in matches],
            [-6.0, -2.0, -2.0, 2.0],
        )
        self.assertEqual([match.score for match in matches], [6.0, 2.0, 2.0, -2.0])
        self.assertEqual([match.rank for match in matches], [1, 2, 3, 4])

    def test_top_k_ties_are_broken_by_target_id_not_input_order(self) -> None:
        neighbors = directed_top_k(
            [
                _vector(4, 1.0, 0.0),
                _vector(2, 1.0, 0.0),
                _vector(3, 1.0, 0.0),
                _vector(1, 1.0, 0.0),
            ],
            k=2,
        )
        targets = {
            source_id: [
                neighbor.target_id
                for neighbor in neighbors
                if neighbor.source_id == source_id
            ]
            for source_id in range(1, 5)
        }

        self.assertEqual(targets[1], [2, 3])
        self.assertEqual(targets[2], [1, 3])
        self.assertEqual(targets[3], [1, 2])
        self.assertEqual(targets[4], [1, 2])
        with self.assertRaises(SimilarityConfigurationError):
            directed_top_k([_vector(1, 1.0)], k=0)

    def test_asymmetric_directed_selection_is_preserved_by_union(self) -> None:
        edges = exact_top_k_union_graph(
            [
                _vector(3, 1.0),
                _vector(1, 1.0),
                _vector(2, 1.0),
            ],
            k=1,
        )

        self.assertEqual(
            [(edge.left_id, edge.right_id) for edge in edges],
            [(1, 2), (1, 3)],
        )
        mutual = edges[0]
        self.assertEqual((mutual.left_rank, mutual.right_rank), (1, 1))
        self.assertTrue(mutual.mutual)
        asymmetric = edges[1]
        self.assertIsNone(asymmetric.left_rank)
        self.assertEqual(asymmetric.right_rank, 1)
        self.assertFalse(asymmetric.mutual)

    def test_top_five_union_can_have_undirected_degree_above_five(self) -> None:
        records = [_vector(id, 1.0, 0.0) for id in range(1, 8)]
        edges = exact_top_k_union_graph(records)
        degree = {id: 0 for id in range(1, 8)}
        for edge in edges:
            degree[edge.left_id] += 1
            degree[edge.right_id] += 1

        self.assertEqual(degree[1], 6)
        self.assertGreater(max(degree.values()), 5)
        edge_1_7 = next(
            edge for edge in edges if (edge.left_id, edge.right_id) == (1, 7)
        )
        self.assertIsNone(edge_1_7.left_rank)
        self.assertEqual(edge_1_7.right_rank, 1)

    def test_canonical_artifact_is_numeric_tie_broken_single_source(self) -> None:
        artifact = build_canonical_neighbor_artifact(
            [_vector(id, 1.0, 0.0) for id in (10, 2, 1, 3, 4, 5, 6)]
        )

        self.assertEqual(
            artifact.schema_version,
            CANONICAL_NEIGHBOR_SCHEMA_VERSION,
        )
        self.assertEqual(artifact.provider_id, EXACT_COSINE_PROVIDER_ID)
        self.assertEqual(artifact.k, 5)
        self.assertEqual(artifact.vector_ids, (1, 2, 3, 4, 5, 6, 10))
        self.assertEqual(
            [row.target_id for row in artifact.recommendations_for(1)],
            [2, 3, 4, 5, 6],
        )
        self.assertEqual(
            [row.target_id for row in artifact.recommendations_for(10)],
            [1, 2, 3, 4, 5],
        )
        expected_union = exact_top_k_union_graph(
            [_vector(id, 1.0, 0.0) for id in (10, 2, 1, 3, 4, 5, 6)]
        )
        self.assertEqual(artifact.undirected_edges, expected_union)
        degree = {id: 0 for id in artifact.vector_ids}
        for edge in artifact.undirected_edges:
            degree[edge.left_id] += 1
            degree[edge.right_id] += 1
        self.assertGreater(degree[1], 5)
        asymmetric = next(
            edge
            for edge in artifact.undirected_edges
            if (edge.left_id, edge.right_id) == (1, 10)
        )
        self.assertIsNone(asymmetric.left_rank)
        self.assertEqual(asymmetric.right_rank, 1)
        self.assertFalse(asymmetric.mutual)

    def test_canonical_artifact_rejects_rank_or_union_tampering(self) -> None:
        artifact = build_canonical_neighbor_artifact(
            [_vector(id, 1.0, 0.0) for id in range(1, 8)]
        )
        rows = list(artifact.directed_neighbors)
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(VectorValidationError, "ordered"):
            CanonicalNeighborArtifact(
                schema_version=artifact.schema_version,
                provider_id=artifact.provider_id,
                k=artifact.k,
                vector_ids=artifact.vector_ids,
                vector_set_sha256=artifact.vector_set_sha256,
                directed_neighbors=tuple(rows),
                undirected_edges=artifact.undirected_edges,
            )
        with self.assertRaisesRegex(VectorValidationError, "union"):
            CanonicalNeighborArtifact(
                schema_version=artifact.schema_version,
                provider_id=artifact.provider_id,
                k=artifact.k,
                vector_ids=artifact.vector_ids,
                vector_set_sha256=artifact.vector_set_sha256,
                directed_neighbors=artifact.directed_neighbors,
                undirected_edges=artifact.undirected_edges[:-1],
            )

    def test_exact_backend_is_the_dependency_free_602_record_baseline(self) -> None:
        records = [_vector(id, 1.0, 0.0) for id in range(1, 603)]
        backend = ExactCosineBackend()

        self.assertIsInstance(backend, NeighborSearchBackend)
        neighbors = backend.directed_top_k(records)
        self.assertEqual(len(neighbors), 602 * 5)
        last_targets = [
            neighbor.target_id for neighbor in neighbors if neighbor.source_id == 602
        ]
        self.assertEqual(last_targets, [1, 2, 3, 4, 5])

    def test_ann_is_an_interface_placeholder_without_dependency_or_fallback(
        self,
    ) -> None:
        placeholder = AnnBackendPlaceholder()

        self.assertIsInstance(placeholder, NeighborSearchBackend)
        with self.assertRaisesRegex(AnnBackendUnavailableError, "future adapter"):
            build_union_graph(
                [_vector(1, 1.0), _vector(2, 1.0)],
                backend=placeholder,
            )


if __name__ == "__main__":
    unittest.main()
