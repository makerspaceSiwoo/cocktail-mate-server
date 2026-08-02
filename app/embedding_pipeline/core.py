"""Pure data, artifact, and neighbor-preservation helpers."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

TASTE_COLUMNS = (
    "cocktail_id",
    "cocktail_name_ko",
    "cocktail_name_en",
    "recipe",
    "embedding_text",
)


@dataclass(frozen=True, slots=True)
class TasteDataset:
    cocktail_ids: np.ndarray
    cocktail_names: tuple[str, ...]
    embedding_texts: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.cocktail_names)


@dataclass(frozen=True, slots=True)
class VectorArtifact:
    cocktail_ids: np.ndarray
    cocktail_names: tuple[str, ...]
    vectors: np.ndarray
    metadata: dict[str, Any]

    def __len__(self) -> int:
        return len(self.cocktail_names)


def read_taste_dataset(path: Path) -> TasteDataset:
    """Read the Gemini-produced CSV without making any Gemini/API call."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != TASTE_COLUMNS:
                raise ValueError(
                    f"{path}: expected columns {TASTE_COLUMNS}, got "
                    f"{tuple(reader.fieldnames or ())}"
                )
            rows = list(reader)
    except FileNotFoundError as error:
        raise ValueError(f"taste CSV does not exist: {path}") from error

    if len(rows) < 2:
        raise ValueError(f"{path}: at least two cocktail rows are required")

    parsed: list[tuple[int, str, str]] = []
    seen_ids: set[int] = set()
    for line_number, row in enumerate(rows, start=2):
        try:
            cocktail_id = int(row["cocktail_id"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:{line_number}: cocktail_id must be an integer"
            ) from error
        if cocktail_id <= 0:
            raise ValueError(f"{path}:{line_number}: cocktail_id must be positive")
        if cocktail_id in seen_ids:
            raise ValueError(
                f"{path}:{line_number}: duplicate cocktail_id {cocktail_id}"
            )
        name = row["cocktail_name_ko"].strip()
        text = row["embedding_text"].strip()
        if not name:
            raise ValueError(f"{path}:{line_number}: cocktail_name_ko is empty")
        if not text:
            raise ValueError(f"{path}:{line_number}: embedding_text is empty")
        seen_ids.add(cocktail_id)
        parsed.append((cocktail_id, name, text))

    parsed.sort(key=lambda item: item[0])
    return TasteDataset(
        cocktail_ids=np.asarray([item[0] for item in parsed], dtype=np.int64),
        cocktail_names=tuple(item[1] for item in parsed),
        embedding_texts=tuple(item[2] for item in parsed),
    )


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(np.isfinite(matrix)) or np.any(norms <= 1e-12):
        raise ValueError("vectors must be finite and have non-zero norm")
    return np.asarray(matrix / norms, dtype=np.float32)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    normalized = l2_normalize(vectors)
    return np.asarray(normalized @ normalized.T, dtype=np.float32)


def knn_indices(
    vectors: np.ndarray,
    k: int,
    *,
    metric: Literal["cosine", "euclidean"] = "cosine",
) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {matrix.shape}")
    if not 1 <= k < len(matrix):
        raise ValueError(f"k must be between 1 and n-1; got k={k}, n={len(matrix)}")

    if metric == "cosine":
        scores = cosine_similarity_matrix(matrix)
        np.fill_diagonal(scores, -np.inf)
        candidates = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1, kind="stable")
    elif metric == "euclidean":
        squared = np.sum(matrix * matrix, axis=1, keepdims=True)
        distances = squared + squared.T - 2.0 * (matrix @ matrix.T)
        distances = np.maximum(distances, 0.0)
        np.fill_diagonal(distances, np.inf)
        candidates = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        candidate_scores = np.take_along_axis(distances, candidates, axis=1)
        order = np.argsort(candidate_scores, axis=1, kind="stable")
    else:
        raise ValueError(f"unsupported metric: {metric}")
    return np.take_along_axis(candidates, order, axis=1)


def neighbor_recall(
    reference_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    k: int,
    *,
    candidate_metric: Literal["cosine", "euclidean"] = "cosine",
) -> float:
    if len(reference_vectors) != len(candidate_vectors):
        raise ValueError("reference and candidate row counts must match")
    reference = knn_indices(reference_vectors, k, metric="cosine")
    candidate = knn_indices(candidate_vectors, k, metric=candidate_metric)
    overlap = [
        len(set(expected).intersection(actual)) / k
        for expected, actual in zip(reference, candidate, strict=True)
    ]
    return float(np.mean(overlap))


def neighbor_metrics(
    reference_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    *,
    candidate_metric: Literal["cosine", "euclidean"] = "cosine",
    ks: tuple[int, ...] = (5, 10, 15),
) -> dict[str, float]:
    valid_ks = tuple(k for k in ks if k < len(reference_vectors))
    if not valid_ks:
        raise ValueError("the dataset is too small for the requested k values")
    return {
        f"recall@{k}": neighbor_recall(
            reference_vectors,
            candidate_vectors,
            k,
            candidate_metric=candidate_metric,
        )
        for k in valid_ks
    }


def normalized_stress(
    reference_vectors: np.ndarray,
    coordinates: np.ndarray,
) -> float:
    """Scale-invariant Kruskal stress against cosine distance."""
    reference = 1.0 - cosine_similarity_matrix(reference_vectors)
    points = np.asarray(coordinates, dtype=np.float64)
    differences = points[:, None, :] - points[None, :, :]
    visual = np.linalg.norm(differences, axis=2)
    upper = np.triu_indices(len(points), k=1)
    source_distances = reference[upper].astype(np.float64)
    visual_distances = visual[upper]
    denominator = float(visual_distances @ visual_distances)
    if denominator <= 1e-15:
        return float("inf")
    scale = float((source_distances @ visual_distances) / denominator)
    residual = source_distances - scale * visual_distances
    source_energy = float(source_distances @ source_distances)
    return float(np.sqrt((residual @ residual) / source_energy))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_vector_artifact(
    path: Path,
    *,
    cocktail_ids: np.ndarray,
    cocktail_names: tuple[str, ...],
    vectors: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape[0] != len(cocktail_ids) or len(cocktail_ids) != len(cocktail_names):
        raise ValueError("IDs, names, and vectors must have the same row count")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary,
            cocktail_ids=np.asarray(cocktail_ids, dtype=np.int64),
            cocktail_names=np.asarray(cocktail_names, dtype=np.str_),
            vectors=matrix,
            metadata=np.asarray(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            ),
        )
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_vector_artifact(path: Path) -> VectorArtifact:
    try:
        with np.load(path, allow_pickle=False) as archive:
            ids = np.asarray(archive["cocktail_ids"], dtype=np.int64)
            names = tuple(str(value) for value in archive["cocktail_names"].tolist())
            vectors = np.asarray(archive["vectors"], dtype=np.float32)
            metadata = json.loads(str(archive["metadata"].item()))
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        raise ValueError(f"invalid vector artifact {path}: {error}") from error
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or len(ids) != len(names):
        raise ValueError(f"invalid vector artifact shapes in {path}")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"duplicate cocktail IDs in {path}")
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"non-finite vectors in {path}")
    return VectorArtifact(ids, names, vectors, metadata)


def write_vector_csv(
    path: Path,
    artifact: VectorArtifact,
    *,
    vector_column: str,
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("cocktail_id", "cocktail_name_ko", vector_column))
    for cocktail_id, name, vector in zip(
        artifact.cocktail_ids,
        artifact.cocktail_names,
        artifact.vectors,
        strict=True,
    ):
        writer.writerow(
            (
                int(cocktail_id),
                name,
                json.dumps([round(float(value), 8) for value in vector]),
            )
        )
    _atomic_write_text(path, buffer.getvalue())


def write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
