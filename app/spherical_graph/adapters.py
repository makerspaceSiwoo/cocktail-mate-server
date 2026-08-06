"""Optional bridges from embedding and similarity modules into module 3.

This module intentionally owns the imports of ``sensory_embedding`` and
``vector_similarity``.  The core ``app.spherical_graph`` package remains usable
without either optional producer module being installed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.sensory_embedding import CocktailEmbedding
from app.spherical_graph.clustering import HighDimensionalClusterer
from app.spherical_graph.models import (
    DirectedNeighbor as SphericalDirectedNeighbor,
)
from app.spherical_graph.models import GraphEdge, SphericalGraph
from app.spherical_graph.models import VectorRecord as SphericalVectorRecord
from app.spherical_graph.pipeline import (
    SphericalGraphConfig,
    build_spherical_graph_from_topology,
    prepare_spherical_graph_topology,
    similarity_to_target_distance,
)
from app.spherical_graph.similarity import SimilarityProvider
from app.vector_similarity import (
    CanonicalNeighborArtifact,
    DenseVector,
    canonical_vector_set_sha256,
    cosine_similarity,
)


def sensory_embeddings_to_records(
    embeddings: Sequence[CocktailEmbedding],
) -> tuple[SphericalVectorRecord, ...]:
    """Adapt one embedding space into deterministically ordered graph records."""

    ordered = _validated_embeddings(embeddings)
    return tuple(
        SphericalVectorRecord(
            node_id=str(embedding.cocktail_id),
            vector=embedding.values,
        )
        for embedding in ordered
    )


def sensory_embeddings_to_mapping(
    embeddings: Sequence[CocktailEmbedding],
) -> dict[str, tuple[float, ...]]:
    """Adapt embeddings to module 3's equivalent mapping input form."""

    return {
        record.node_id: record.vector
        for record in sensory_embeddings_to_records(embeddings)
    }


def _validated_embeddings(
    embeddings: Sequence[CocktailEmbedding],
) -> tuple[CocktailEmbedding, ...]:
    if not embeddings:
        raise ValueError("at least one cocktail embedding is required")
    if not all(isinstance(embedding, CocktailEmbedding) for embedding in embeddings):
        raise TypeError("all values must be CocktailEmbedding instances")

    first_contract = embeddings[0].contract
    if any(embedding.contract != first_contract for embedding in embeddings):
        raise ValueError("cocktail embeddings must share one embedding contract")
    ids = [embedding.cocktail_id for embedding in embeddings]
    if len(set(ids)) != len(ids):
        raise ValueError("cocktail embedding IDs must be unique")
    return tuple(sorted(embeddings, key=lambda embedding: embedding.cocktail_id))


class VectorSimilarityCosineProvider:
    """Module 3 provider backed by module 2's exact cosine primitive."""

    provider_id = "vector_similarity_exact_cosine_v1"

    def pairwise_cosine(
        self,
        node_ids: Sequence[str],
        vectors: np.ndarray,
    ) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a two-dimensional matrix")
        if len(node_ids) != len(matrix):
            raise ValueError("node_ids and vectors must have equal row counts")
        if matrix.shape[1] == 0:
            raise ValueError("vectors must have a positive dimension")

        similarities = np.eye(len(node_ids), dtype=np.float64)
        rows = tuple(tuple(float(value) for value in row) for row in matrix)
        for left_index, left in enumerate(rows):
            for right_index in range(left_index + 1, len(rows)):
                score = cosine_similarity(left, rows[right_index])
                similarities[left_index, right_index] = score
                similarities[right_index, left_index] = score
        return similarities


def build_spherical_graph_from_canonical(
    records: Sequence[SphericalVectorRecord],
    artifact: CanonicalNeighborArtifact,
    *,
    similarity_provider: SimilarityProvider | None = None,
    clusterer: HighDimensionalClusterer | None = None,
    config: SphericalGraphConfig | None = None,
) -> SphericalGraph:
    """Consume module 2 ranks and union edges without deriving either again."""

    selected_config = config or SphericalGraphConfig(k=artifact.k)
    if selected_config.k != artifact.k:
        raise ValueError("graph config k must match canonical artifact k")

    numeric_records: list[DenseVector] = []
    for record in records:
        try:
            cocktail_id = int(record.node_id)
        except ValueError as error:
            raise ValueError("canonical graph node IDs must be numeric") from error
        if str(cocktail_id) != record.node_id or cocktail_id <= 0:
            raise ValueError(
                "canonical graph node IDs must be canonical positive integers"
            )
        numeric_records.append(DenseVector.from_values(cocktail_id, record.vector))
    if tuple(sorted(record.id for record in numeric_records)) != artifact.vector_ids:
        raise ValueError("graph records do not match canonical artifact IDs")
    if canonical_vector_set_sha256(numeric_records) != artifact.vector_set_sha256:
        raise ValueError("graph records do not match canonical vector-set SHA-256")

    directed = tuple(
        SphericalDirectedNeighbor(
            source_id=str(row.source_id),
            target_id=str(row.target_id),
            rank=row.rank,
            similarity=row.score,
            target_distance=similarity_to_target_distance(row.score),
        )
        for row in artifact.directed_neighbors
    )
    edges = tuple(
        GraphEdge(
            source_id=str(edge.left_id),
            target_id=str(edge.right_id),
            edge_kind="cocktail_knn",
            similarity=edge.score,
            target_distance=similarity_to_target_distance(edge.score),
            source_rank=edge.left_rank,
            target_rank=edge.right_rank,
            is_mutual=edge.mutual,
            is_bridge=False,
            visible=True,
            recommendable=True,
        )
        for edge in artifact.undirected_edges
    )
    (
        node_ids,
        components,
        private_hub_edges,
        clusterer_id,
        audit_similarities,
    ) = prepare_spherical_graph_topology(
        records,
        directed_neighbors=directed,
        cocktail_edges=edges,
        similarity_provider=(similarity_provider or VectorSimilarityCosineProvider()),
        clusterer=clusterer,
    )
    return build_spherical_graph_from_topology(
        node_ids,
        directed_neighbors=directed,
        cocktail_edges=edges,
        components=components,
        private_hub_edges=private_hub_edges,
        topology_provider_id=artifact.provider_id,
        clustering_policy=clusterer_id,
        audit_similarities=audit_similarities,
        audit_node_ids=node_ids,
        config=selected_config,
    )
