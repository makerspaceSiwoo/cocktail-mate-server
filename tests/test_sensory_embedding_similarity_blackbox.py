from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from typing import Sequence

from app.sensory_embedding import build_user_query
from app.vector_similarity import (
    build_graph48_artifact,
    cosine_similarity,
    graph48_all_pairs,
    graph48_vector_sha256,
)


GRAPH48_DIMENSION = 48


def _basis(position: int, *, scale: float = 1.0) -> tuple[float, ...]:
    values = [0.0] * GRAPH48_DIMENSION
    values[position] = scale
    return tuple(values)


@dataclass(frozen=True)
class _StaticRecord:
    id: int
    values: Sequence[float]


class _StatefulRecord:
    """Expose different valid snapshots to detect repeated source reads."""

    def __init__(
        self,
        cocktail_id: int,
        snapshots: Sequence[tuple[float, ...]],
    ) -> None:
        self.id = cocktail_id
        self._snapshots = tuple(snapshots)
        self.read_count = 0

    @property
    def values(self) -> tuple[float, ...]:
        index = min(self.read_count, len(self._snapshots) - 1)
        self.read_count += 1
        return self._snapshots[index]


class SensoryEmbeddingSimilarityBlackBoxTests(unittest.TestCase):
    def test_extreme_finite_query_weights_preserve_equal_category_mass(self) -> None:
        query = build_user_query(
            {
                "sweetness": 1e308,
                "saltiness": 1e308,
                "citrus_fruit": 1.0,
            }
        )

        taste_mass = math.fsum(query.values[:8])
        fruit_mass = math.fsum(query.values[8:15])
        self.assertEqual(taste_mass, 0.5)
        self.assertEqual(fruit_mass, 0.5)
        self.assertEqual(query.values[0], 0.25)
        self.assertEqual(query.values[1], 0.25)
        self.assertEqual(query.values[8], 0.5)
        self.assertTrue(
            all(
                value == 0.0
                for index, value in enumerate(query.values)
                if index not in {0, 1, 8}
            )
        )

    def test_graph48_pair_score_is_exact_cosine_for_accepted_vectors(self) -> None:
        almost_unit = _StaticRecord(1, _basis(0, scale=0.9999995))
        unit = _StaticRecord(2, _basis(0))

        pair = graph48_all_pairs((almost_unit, unit))[0]

        self.assertEqual(
            pair.cosine,
            cosine_similarity(almost_unit.values, unit.values),
        )
        self.assertEqual(pair.cosine, 1.0)

    def test_graph_artifact_hash_binds_the_snapshot_used_for_topology(self) -> None:
        first_snapshot = _basis(0)
        substituted_hash_snapshot = _basis(1)
        changing = _StatefulRecord(
            1,
            (
                first_snapshot,
                first_snapshot,
                first_snapshot,
                substituted_hash_snapshot,
                substituted_hash_snapshot,
                first_snapshot,
            ),
        )
        stable_records = tuple(
            _StaticRecord(cocktail_id, _basis(cocktail_id - 1))
            for cocktail_id in range(2, 7)
        )

        artifact = build_graph48_artifact(
            (changing, *stable_records),
            run_id="blackbox-mutable-source-v1",
        )
        topology_snapshot = (
            _StaticRecord(1, first_snapshot),
            *stable_records,
        )

        self.assertTrue(
            all(row.cosine == 0.0 for row in artifact.recommendations_for(1))
        )
        self.assertEqual(
            artifact.run.vector_sha256,
            graph48_vector_sha256(topology_snapshot),
        )


if __name__ == "__main__":
    unittest.main()
