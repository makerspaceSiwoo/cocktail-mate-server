"""NumPy inference for the offline-trained taste graph decoder."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "artifacts" / "taste-query-gnn.npz"
)
MINIMUM_DESCRIPTOR_SUPPORT = 5


@dataclass(frozen=True, slots=True)
class TasteQueryModel:
    descriptor_codes: tuple[str, ...]
    descriptor_support: np.ndarray
    node_embeddings: np.ndarray
    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray

    def encode(self, codes: list[str] | tuple[str, ...]) -> np.ndarray:
        if not codes:
            raise ValueError("at least one taste descriptor is required")
        index = {code: position for position, code in enumerate(self.descriptor_codes)}
        unknown = sorted(set(codes) - set(index))
        if unknown:
            raise ValueError(f"model does not support taste descriptors: {unknown}")
        unique_codes = list(dict.fromkeys(codes))
        positions = [index[code] for code in unique_codes]
        unsupported = [
            code
            for code, position in zip(unique_codes, positions, strict=True)
            if int(self.descriptor_support[position]) < MINIMUM_DESCRIPTOR_SUPPORT
        ]
        if unsupported:
            raise ValueError(
                "taste descriptors need at least "
                f"{MINIMUM_DESCRIPTOR_SUPPORT} training cocktails: {unsupported}"
            )
        message = self.node_embeddings[positions].mean(axis=0)
        hidden = np.tanh(message @ self.hidden_weight + self.hidden_bias)
        output = hidden @ self.output_weight + self.output_bias
        norm = float(np.linalg.norm(output))
        if norm <= 1e-12:
            raise ValueError("taste query decoder produced a zero vector")
        return np.asarray(output / norm, dtype=np.float32)

    @property
    def supported_codes(self) -> frozenset[str]:
        return frozenset(
            code
            for code, support in zip(
                self.descriptor_codes,
                self.descriptor_support,
                strict=True,
            )
            if int(support) >= MINIMUM_DESCRIPTOR_SUPPORT
        )


def load_taste_query_model(path: Path = DEFAULT_MODEL_PATH) -> TasteQueryModel:
    if not path.exists():
        raise RuntimeError(f"taste query model is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return TasteQueryModel(
            descriptor_codes=tuple(str(value) for value in archive["descriptor_codes"]),
            descriptor_support=np.asarray(
                archive["descriptor_support"],
                dtype=np.int32,
            ),
            node_embeddings=np.asarray(archive["node_embeddings"], dtype=np.float32),
            hidden_weight=np.asarray(archive["hidden_weight"], dtype=np.float32),
            hidden_bias=np.asarray(archive["hidden_bias"], dtype=np.float32),
            output_weight=np.asarray(archive["output_weight"], dtype=np.float32),
            output_bias=np.asarray(archive["output_bias"], dtype=np.float32),
        )


@lru_cache(maxsize=1)
def get_taste_query_model() -> TasteQueryModel:
    return load_taste_query_model()
