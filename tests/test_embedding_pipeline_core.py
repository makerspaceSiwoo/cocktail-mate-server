from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from app.embedding_pipeline.core import (
    TasteDataset,
    knn_indices,
    load_vector_artifact,
    neighbor_metrics,
    normalized_stress,
    read_taste_dataset,
    save_vector_artifact,
    write_vector_csv,
)
from app.embedding_pipeline.local_embedding import encode_taste_dataset


def _write_taste_csv(path: Path, rows: list[tuple[int, str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "cocktail_id",
                "cocktail_name_ko",
                "cocktail_name_en",
                "recipe",
                "embedding_text",
            )
        )
        for cocktail_id, name, text in rows:
            writer.writerow((cocktail_id, name, "", "[]", text))


def test_read_taste_dataset_handles_bom_and_sorts_ids(tmp_path: Path) -> None:
    path = tmp_path / "taste.csv"
    _write_taste_csv(path, [(7, "칠", "맛 칠"), (2, "이", "맛 이")])

    dataset = read_taste_dataset(path)

    assert dataset.cocktail_ids.tolist() == [2, 7]
    assert dataset.cocktail_names == ("이", "칠")
    assert dataset.embedding_texts == ("맛 이", "맛 칠")


def test_read_taste_dataset_rejects_duplicate_id(tmp_path: Path) -> None:
    path = tmp_path / "taste.csv"
    _write_taste_csv(path, [(2, "이", "맛 이"), (2, "또 이", "다른 맛")])

    with pytest.raises(ValueError, match="duplicate cocktail_id 2"):
        read_taste_dataset(path)


def test_neighbor_metrics_are_exact_for_orthogonal_rotation() -> None:
    generator = np.random.default_rng(7)
    source = generator.normal(size=(40, 12)).astype(np.float32)
    q, _ = np.linalg.qr(generator.normal(size=(12, 12)))
    rotated = source @ q.astype(np.float32)

    metrics = neighbor_metrics(source, rotated, ks=(5, 10))

    assert metrics == {"recall@5": 1.0, "recall@10": 1.0}
    assert normalized_stress(source, rotated) == pytest.approx(
        normalized_stress(source, source), abs=1e-6
    )


def test_knn_indices_excludes_self() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]], dtype=np.float32)

    neighbors = knn_indices(vectors, 1)

    assert neighbors[:, 0].tolist() == [1, 0, 1]


def test_vector_artifact_round_trip_and_csv(tmp_path: Path) -> None:
    artifact_path = tmp_path / "vectors.npz"
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    save_vector_artifact(
        artifact_path,
        cocktail_ids=np.asarray([1, 2]),
        cocktail_names=("하나", "둘"),
        vectors=vectors,
        metadata={"model": "local"},
    )

    artifact = load_vector_artifact(artifact_path)
    csv_path = tmp_path / "vectors.csv"
    write_vector_csv(csv_path, artifact, vector_column="embedding")

    assert artifact.metadata == {"model": "local"}
    assert np.array_equal(artifact.vectors, vectors)
    with csv_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert json.loads(rows[1]["embedding"]) == [0.0, 1.0]


class _FakeSentenceTransformer:
    def __init__(self, *_args, **kwargs) -> None:
        assert kwargs["local_files_only"] is True

    @staticmethod
    def get_sentence_embedding_dimension() -> int:
        return 512

    @staticmethod
    def encode(texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        vectors = np.zeros((len(texts), 512), dtype=np.float32)
        vectors[:, 0] = 2.0
        return vectors


def test_encode_taste_dataset_requires_local_model_and_returns_unit_vectors(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    dataset = TasteDataset(
        cocktail_ids=np.asarray([1, 2]),
        cocktail_names=("하나", "둘"),
        embedding_texts=("맛 하나", "맛 둘"),
    )

    vectors = encode_taste_dataset(
        dataset,
        model_dir,
        progress=False,
        model_factory=_FakeSentenceTransformer,
    )

    assert vectors.shape == (2, 512)
    assert np.linalg.norm(vectors, axis=1).tolist() == [1.0, 1.0]
