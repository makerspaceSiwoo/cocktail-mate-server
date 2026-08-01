"""Pinned Hugging Face download and fully local sentence encoding."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from app.embedding_pipeline.core import TasteDataset, l2_normalize
from app.embedding_pipeline.core import write_json

MODEL_ID = "sentence-transformers/distiluse-base-multilingual-cased-v2"
MODEL_REVISION = "bfe45d0732ca50787611c0fe107ba278c7f3f889"
MODEL_DIMENSION = 512
MODEL_MAX_SEQUENCE_LENGTH = 128
MODEL_ALLOW_PATTERNS = (
    "*.json",
    "*.txt",
    "*.safetensors",
    "README.md",
)
MODEL_MARKER = ".cocktail-mate-model.json"


def download_model(
    model_dir: Path,
    *,
    repo_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
) -> Path:
    """Download only native safetensors/tokenizer files into a local directory."""
    required_files = (
        model_dir / "modules.json",
        model_dir / "model.safetensors",
        model_dir / "2_Dense" / "model.safetensors",
    )
    marker_path = model_dir / MODEL_MARKER
    if all(path.is_file() for path in required_files):
        write_json(
            marker_path,
            {"model_id": repo_id, "model_revision": revision},
        )
        return model_dir
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface-hub is required; install requirements-embedding.txt"
        ) from error

    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=model_dir,
            allow_patterns=list(MODEL_ALLOW_PATTERNS),
        )
    except Exception as error:
        raise RuntimeError(
            f"failed to download {repo_id}@{revision}: {error}"
        ) from error
    write_json(
        marker_path,
        {"model_id": repo_id, "model_revision": revision},
    )
    return model_dir


def encode_taste_dataset(
    dataset: TasteDataset,
    model_dir: Path,
    *,
    batch_size: int = 32,
    device: str = "cpu",
    progress: bool = True,
    model_factory: Callable[..., Any] | None = None,
) -> np.ndarray:
    """Load the model from disk only and create normalized 512D vectors."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not model_dir.is_dir():
        raise ValueError(
            f"local model is missing at {model_dir}; run download-model first"
        )
    if model_factory is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required; install requirements-embedding.txt"
            ) from error
        model_factory = SentenceTransformer

    model = model_factory(
        str(model_dir),
        device=device,
        local_files_only=True,
    )
    dimension_getter = getattr(model, "get_embedding_dimension", None)
    if dimension_getter is None:
        dimension_getter = model.get_sentence_embedding_dimension
    dimension = int(dimension_getter())
    if dimension != MODEL_DIMENSION:
        raise ValueError(
            f"local model output must be {MODEL_DIMENSION}D, got {dimension}D"
        )
    chunks: list[str] = []
    chunk_owners: list[int] = []
    for owner, text in enumerate(dataset.embedding_texts):
        sentences = tuple(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ) or (text,)
        chunks.extend(sentences)
        chunk_owners.extend([owner] * len(sentences))

    chunk_vectors = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    encoded = np.asarray(chunk_vectors, dtype=np.float32)
    expected_chunk_shape = (len(chunks), MODEL_DIMENSION)
    if encoded.shape != expected_chunk_shape:
        raise ValueError(
            f"embedding model returned shape {encoded.shape}, "
            f"expected {expected_chunk_shape}"
        )
    matrix = np.zeros((len(dataset), MODEL_DIMENSION), dtype=np.float32)
    counts = np.zeros((len(dataset), 1), dtype=np.float32)
    np.add.at(matrix, np.asarray(chunk_owners), encoded)
    np.add.at(counts, np.asarray(chunk_owners), 1.0)
    matrix /= counts
    return l2_normalize(matrix)
