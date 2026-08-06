"""Similarity provider protocol and an offline exact-cosine fallback."""

from __future__ import annotations

from typing import Protocol, Sequence, cast, runtime_checkable

import numpy as np


@runtime_checkable
class SimilarityProvider(Protocol):
    """Structural boundary for a future module-owned similarity implementation."""

    provider_id: str

    def pairwise_cosine(
        self,
        node_ids: Sequence[str],
        vectors: np.ndarray,
    ) -> np.ndarray:
        """Return a finite symmetric ``[n, n]`` cosine-similarity matrix."""


class ExactCosineSimilarity:
    """Deterministic in-memory cosine similarity with no external I/O."""

    provider_id = "exact_numpy_cosine_v1"

    def pairwise_cosine(
        self,
        node_ids: Sequence[str],
        vectors: np.ndarray,
    ) -> np.ndarray:
        if len(node_ids) != len(vectors):
            raise ValueError("node_ids and vectors must have equal row counts")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
            raise ValueError("source vectors must be finite and non-zero")
        normalized = vectors / norms[:, None]
        similarities = normalized @ normalized.T
        return cast(np.ndarray, np.clip(similarities, -1.0, 1.0))


def validate_similarity_matrix(
    value: np.ndarray,
    *,
    row_count: int,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Copy and validate a provider result before graph construction."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (row_count, row_count):
        raise ValueError(
            "similarity provider must return a square matrix matching inputs"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("similarity matrix contains NaN or infinity")
    if np.any(matrix < -1.0 - tolerance) or np.any(matrix > 1.0 + tolerance):
        raise ValueError("cosine similarity must be within [-1, 1]")
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0.0):
        raise ValueError("cosine similarity matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("cosine similarity matrix diagonal must equal one")
    result = np.clip((matrix + matrix.T) * 0.5, -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result
