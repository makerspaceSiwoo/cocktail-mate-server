"""Versioned, side-effect-free embedding contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Iterable, Literal, Protocol, runtime_checkable

Metric = Literal["cosine", "inner_product"]

_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase versioned identifier")
    return value


def validate_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def coerce_vector(values: Iterable[float], dimension: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("embedding vector must be numeric")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError("embedding vector must be numeric") from error
    if len(result) != dimension:
        raise ValueError(f"expected {dimension} dimensions, got {len(result)}")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("embedding vector values must be finite")
    return tuple(0.0 if value == 0.0 else value for value in result)


@dataclass(frozen=True, slots=True)
class EmbeddingContract:
    """Identity and search semantics shared by query and cocktail vectors."""

    space: str
    version: str
    dimension: int
    metric: Metric
    registry_sha256: str
    space_sha256: str

    def __post_init__(self) -> None:
        validate_identifier(self.space, field="space")
        validate_identifier(self.version, field="version")
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if self.metric not in {"cosine", "inner_product"}:
            raise ValueError("metric must be cosine or inner_product")
        validate_sha256(self.registry_sha256, field="registry_sha256")
        validate_sha256(self.space_sha256, field="space_sha256")

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(
            {
                "dimension": self.dimension,
                "metric": self.metric,
                "registry_sha256": self.registry_sha256,
                "space": self.space,
                "space_sha256": self.space_sha256,
                "version": self.version,
            }
        )


def vector_sha256(
    contract: EmbeddingContract,
    values: tuple[float, ...],
    *,
    identity: str,
) -> str:
    return canonical_sha256(
        {
            "contract_sha256": contract.contract_sha256,
            "identity": identity,
            "values_hex": [value.hex() for value in values],
        }
    )


@dataclass(frozen=True, slots=True)
class CocktailEmbedding:
    cocktail_id: int
    values: tuple[float, ...]
    contract: EmbeddingContract
    vector_sha256: str

    def __post_init__(self) -> None:
        if type(self.cocktail_id) is not int or self.cocktail_id <= 0:
            raise ValueError("cocktail_id must be a positive integer")
        values = coerce_vector(self.values, self.contract.dimension)
        if values != self.values:
            raise ValueError("values must already be a canonical float tuple")
        expected = vector_sha256(
            self.contract,
            values,
            identity=f"cocktail:{self.cocktail_id}",
        )
        if self.vector_sha256 != expected:
            raise ValueError("cocktail vector SHA-256 does not match its contents")

    @property
    def id(self) -> int:
        """Structural compatibility with vector_similarity.VectorRecord."""
        return self.cocktail_id


@dataclass(frozen=True, slots=True)
class QueryEmbedding:
    values: tuple[float, ...]
    contract: EmbeddingContract
    selected_ids: tuple[str, ...]
    vector_sha256: str

    def __post_init__(self) -> None:
        values = coerce_vector(self.values, self.contract.dimension)
        if values != self.values:
            raise ValueError("values must already be a canonical float tuple")
        if not self.selected_ids or len(set(self.selected_ids)) != len(
            self.selected_ids
        ):
            raise ValueError("selected_ids must be non-empty and unique")
        expected = vector_sha256(
            self.contract,
            values,
            identity="query:" + ",".join(self.selected_ids),
        )
        if self.vector_sha256 != expected:
            raise ValueError("query vector SHA-256 does not match its contents")


@runtime_checkable
class CocktailEmbeddingAdapter(Protocol):
    @property
    def contract(self) -> EmbeddingContract: ...

    def adapt(
        self,
        cocktail_id: int,
        values: Iterable[float],
    ) -> CocktailEmbedding: ...


@runtime_checkable
class QueryEmbeddingEncoder(Protocol):
    @property
    def contract(self) -> EmbeddingContract: ...

    def encode(self, selections: Iterable[object]) -> QueryEmbedding: ...
