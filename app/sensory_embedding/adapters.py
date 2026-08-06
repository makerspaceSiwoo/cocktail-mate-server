"""Adapters for legacy 32D and sensory-v2 48D embedding spaces."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Protocol

from app.sensory_embedding.contracts import (
    CocktailEmbedding,
    EmbeddingContract,
    QueryEmbedding,
    canonical_sha256,
    coerce_vector,
    vector_sha256,
)
from app.sensory_embedding.query import PositiveSelection
from app.sensory_embedding.registry import SensoryRegistry

LEGACY_32_DIMENSION = 32
LEGACY_NO_REGISTRY_SHA256 = hashlib.sha256(
    b"cocktail-mate:legacy-32:no-axis-registry"
).hexdigest()
GRAPH48_VERSION = "sensory-graph-48-v2"
PREFERENCE48_VERSION = "sensory-preference-48-v2"
GRAPH48_SPACE_SHA256 = canonical_sha256(
    {
        "balance": "multiply-axis-by-sqrt-one-over-category-size",
        "expected_intensity": [0.0, 0.25, 0.5, 0.75, 1.0],
        "normalization": "l2",
        "schema": GRAPH48_VERSION,
        "zero_vector": "quarantine",
    }
)
PREFERENCE48_SPACE_SHA256 = canonical_sha256(
    {
        "basic_taste_axes": [
            "sweetness",
            "saltiness",
            "sourness",
            "bitterness",
            "umami",
        ],
        "basic_taste_utility": [0.0, 0.2, 0.65, 1.0, 0.8],
        "default_utility": [0.0, 0.25, 0.5, 0.75, 1.0],
        "normalization": "none",
        "schema": PREFERENCE48_VERSION,
    }
)


class SupportsLegacyTasteQuery(Protocol):
    """Structural subset implemented by app.taste_query.model.TasteQueryModel."""

    def encode(self, codes: list[str] | tuple[str, ...]) -> Iterable[float]: ...


def legacy_32_contract(
    *,
    space_sha256: str,
    version: str = "legacy-cocktail-32-v1",
) -> EmbeddingContract:
    return EmbeddingContract(
        space="legacy-cocktail-32",
        version=version,
        dimension=LEGACY_32_DIMENSION,
        metric="cosine",
        registry_sha256=LEGACY_NO_REGISTRY_SHA256,
        space_sha256=space_sha256,
    )


def graph48_contract(
    *,
    registry: SensoryRegistry,
    space_sha256: str = GRAPH48_SPACE_SHA256,
    version: str = GRAPH48_VERSION,
) -> EmbeddingContract:
    if registry.dimension != 48:
        raise ValueError("sensory graph registry must have 48 axes")
    return EmbeddingContract(
        space="sensory-graph-48",
        version=version,
        dimension=registry.dimension,
        metric="cosine",
        registry_sha256=registry.registry_sha256,
        space_sha256=space_sha256,
    )


def preference48_contract(
    *,
    registry: SensoryRegistry,
    space_sha256: str = PREFERENCE48_SPACE_SHA256,
    version: str = PREFERENCE48_VERSION,
) -> EmbeddingContract:
    if registry.dimension != 48:
        raise ValueError("sensory preference registry must have 48 axes")
    return EmbeddingContract(
        space="sensory-preference-48",
        version=version,
        dimension=registry.dimension,
        metric="inner_product",
        registry_sha256=registry.registry_sha256,
        space_sha256=space_sha256,
    )


def sensory_48_contract(
    *,
    registry: SensoryRegistry,
    space_sha256: str,
    version: str = PREFERENCE48_VERSION,
) -> EmbeddingContract:
    """Backward-compatible spelling for the preference48 MIPS contract."""

    return preference48_contract(
        registry=registry,
        space_sha256=space_sha256,
        version=version,
    )


@dataclass(frozen=True, slots=True)
class FixedDimensionCocktailAdapter:
    """Validate an already-computed vector without DB or file access."""

    embedding_contract: EmbeddingContract
    minimum: float | None = None
    maximum: float | None = None
    require_unit_norm: bool = False

    def __post_init__(self) -> None:
        if self.minimum is not None and not math.isfinite(self.minimum):
            raise ValueError("minimum must be finite")
        if self.maximum is not None and not math.isfinite(self.maximum):
            raise ValueError("maximum must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum cannot exceed maximum")

    @property
    def contract(self) -> EmbeddingContract:
        return self.embedding_contract

    def adapt(
        self,
        cocktail_id: int,
        values: Iterable[float],
    ) -> CocktailEmbedding:
        vector = coerce_vector(values, self.embedding_contract.dimension)
        if self.minimum is not None and any(value < self.minimum for value in vector):
            raise ValueError("cocktail vector is below its allowed range")
        if self.maximum is not None and any(value > self.maximum for value in vector):
            raise ValueError("cocktail vector is above its allowed range")
        if self.require_unit_norm and not math.isclose(
            math.sqrt(math.fsum(value * value for value in vector)),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("cocktail vector must have unit L2 norm")
        return CocktailEmbedding(
            cocktail_id=cocktail_id,
            values=vector,
            contract=self.embedding_contract,
            vector_sha256=vector_sha256(
                self.embedding_contract,
                vector,
                identity=f"cocktail:{cocktail_id}",
            ),
        )


def legacy_32_cocktail_adapter(
    contract: EmbeddingContract,
) -> FixedDimensionCocktailAdapter:
    if contract.dimension != LEGACY_32_DIMENSION or contract.metric != "cosine":
        raise ValueError("legacy cocktail adapter requires a 32D cosine contract")
    return FixedDimensionCocktailAdapter(
        embedding_contract=contract,
        require_unit_norm=True,
    )


def graph48_cocktail_adapter(
    contract: EmbeddingContract,
) -> FixedDimensionCocktailAdapter:
    if (
        contract.space != "sensory-graph-48"
        or contract.dimension != 48
        or contract.metric != "cosine"
    ):
        raise ValueError("graph48 adapter requires its 48D cosine contract")
    return FixedDimensionCocktailAdapter(
        embedding_contract=contract,
        minimum=0.0,
        maximum=1.0,
        require_unit_norm=True,
    )


def preference48_cocktail_adapter(
    contract: EmbeddingContract,
) -> FixedDimensionCocktailAdapter:
    if (
        contract.space != "sensory-preference-48"
        or contract.dimension != 48
        or contract.metric != "inner_product"
    ):
        raise ValueError("preference48 adapter requires its 48D inner-product contract")
    return FixedDimensionCocktailAdapter(
        embedding_contract=contract,
        minimum=0.0,
        maximum=1.0,
    )


def sensory_48_cocktail_adapter(
    contract: EmbeddingContract,
) -> FixedDimensionCocktailAdapter:
    """Backward-compatible spelling for the preference48 cocktail adapter."""

    return preference48_cocktail_adapter(contract)


@dataclass(frozen=True, slots=True)
class LegacyTasteQueryAdapter:
    """Use the current taste-query decoder through the new query protocol."""

    model: SupportsLegacyTasteQuery
    embedding_contract: EmbeddingContract

    def __post_init__(self) -> None:
        if (
            self.embedding_contract.dimension != LEGACY_32_DIMENSION
            or self.embedding_contract.metric != "cosine"
        ):
            raise ValueError("legacy taste query requires a 32D cosine contract")

    @property
    def contract(self) -> EmbeddingContract:
        return self.embedding_contract

    def encode(
        self,
        selections: Iterable[PositiveSelection],
    ) -> QueryEmbedding:
        received = tuple(selections)
        if not received:
            raise ValueError("at least one positive selection is required")
        if not all(isinstance(item, PositiveSelection) for item in received):
            raise ValueError("all selections must be PositiveSelection values")
        if len({item.axis_id for item in received}) != len(received):
            raise ValueError("duplicate legacy selections are forbidden")
        if any(not math.isclose(float(item.weight), 1.0) for item in received):
            raise ValueError(
                "legacy taste-query model supports unweighted selections only"
            )

        codes = tuple(sorted(item.axis_id for item in received))
        vector = coerce_vector(
            self.model.encode(codes),
            self.embedding_contract.dimension,
        )
        if not math.isclose(
            math.sqrt(math.fsum(value * value for value in vector)),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("legacy taste-query model must return a unit vector")
        return QueryEmbedding(
            values=vector,
            contract=self.embedding_contract,
            selected_ids=codes,
            vector_sha256=vector_sha256(
                self.embedding_contract,
                vector,
                identity="query:" + ",".join(codes),
            ),
        )
