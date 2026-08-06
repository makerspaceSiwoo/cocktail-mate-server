"""Positive-only sensory selections to a category-balanced 48D query."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable

from app.sensory_embedding.contracts import (
    CocktailEmbedding,
    EmbeddingContract,
    QueryEmbedding,
    validate_identifier,
    vector_sha256,
)
from app.sensory_embedding.registry import (
    SENSORY_V2_DIMENSION,
    SensoryRegistry,
)


@dataclass(frozen=True, slots=True)
class PositiveSelection:
    axis_id: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        validate_identifier(self.axis_id, field="selection axis_id")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(float(self.weight))
            or float(self.weight) <= 0.0
        ):
            raise ValueError("selection weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class SensoryPositiveQueryEncoder:
    """Encode q_a=(1/|C|)*r_a/sum_category(r) with ||q||_1=1."""

    registry: SensoryRegistry
    embedding_contract: EmbeddingContract

    def __post_init__(self) -> None:
        if self.registry.dimension != SENSORY_V2_DIMENSION:
            raise ValueError("sensory v2 query registry must have 48 axes")
        if self.embedding_contract.dimension != self.registry.dimension:
            raise ValueError("query contract dimension does not match registry")
        if (
            self.embedding_contract.space != "sensory-preference-48"
            or self.embedding_contract.metric != "inner_product"
        ):
            raise ValueError("sensory preference queries require inner_product")
        if self.embedding_contract.registry_sha256 != self.registry.registry_sha256:
            raise ValueError("query contract registry SHA-256 does not match")

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

        by_id = {axis.axis_id: axis for axis in self.registry.axes}
        unknown = sorted(
            selection.axis_id
            for selection in received
            if selection.axis_id not in by_id
        )
        if unknown:
            raise ValueError(f"unknown sensory axes: {unknown}")
        selected_ids = [selection.axis_id for selection in received]
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("duplicate sensory selections are forbidden")

        ordered = sorted(
            received,
            key=lambda selection: by_id[selection.axis_id].axis_order,
        )
        category_selections: dict[str, list[PositiveSelection]] = {}
        for selection in ordered:
            category = by_id[selection.axis_id].category
            category_selections.setdefault(category, []).append(selection)

        category_mass = 1.0 / len(category_selections)
        values = [0.0] * self.registry.dimension
        for selections in category_selections.values():
            maximum = max(float(selection.weight) for selection in selections)
            scaled = tuple(
                float(selection.weight) / maximum for selection in selections
            )
            scaled_total = math.fsum(scaled)
            for selection, scaled_weight in zip(selections, scaled, strict=True):
                axis = by_id[selection.axis_id]
                values[axis.axis_order] = category_mass * scaled_weight / scaled_total

        # Make the persisted L1 invariant exact despite floating-point division.
        selected_positions = [by_id[item.axis_id].axis_order for item in ordered]
        values[selected_positions[-1]] += 1.0 - math.fsum(values)
        if any(value < 0.0 for value in values) or not math.isclose(
            math.fsum(values),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("category-balanced query failed its L1 invariant")

        vector = tuple(values)
        ids = tuple(item.axis_id for item in ordered)
        return QueryEmbedding(
            values=vector,
            contract=self.embedding_contract,
            selected_ids=ids,
            vector_sha256=vector_sha256(
                self.embedding_contract,
                vector,
                identity="query:" + ",".join(ids),
            ),
        )


def build_user_query(
    selected_axis_weights: (
        Mapping[str, float] | Iterable[PositiveSelection] | Iterable[tuple[str, float]]
    ),
    *,
    registry: SensoryRegistry | None = None,
    contract: EmbeddingContract | None = None,
) -> QueryEmbedding:
    """Build a sparse, category-balanced preference48 query.

    Every unselected coordinate remains exactly zero. Active categories receive
    equal mass; weights within each category are L1-normalized.
    """

    from app.sensory_embedding.adapters import preference48_contract
    from app.sensory_embedding.registry import SENSORY_V2_REGISTRY

    active_registry = registry or SENSORY_V2_REGISTRY
    if isinstance(selected_axis_weights, Mapping):
        selections = tuple(
            PositiveSelection(axis_id, weight)
            for axis_id, weight in selected_axis_weights.items()
        )
    else:
        received = tuple(selected_axis_weights)
        selections_list: list[PositiveSelection] = []
        for item in received:
            if isinstance(item, PositiveSelection):
                selections_list.append(item)
                continue
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
            ):
                raise ValueError(
                    "selected_axis_weights must map axis IDs to weights or "
                    "contain (axis_id, weight) pairs"
                )
            selections_list.append(PositiveSelection(item[0], item[1]))
        selections = tuple(selections_list)

    active_contract = contract or preference48_contract(registry=active_registry)
    return SensoryPositiveQueryEncoder(active_registry, active_contract).encode(
        selections
    )


@dataclass(frozen=True, slots=True)
class PreferenceMIPSMatch:
    cocktail_id: int
    score: float

    @property
    def id(self) -> int:
        return self.cocktail_id

    @property
    def negative_inner_product(self) -> float:
        """pgvector ``<#>`` distance corresponding to this exact MIPS score."""

        return -self.score


def score_preference_mips(
    query: QueryEmbedding,
    cocktails: Iterable[object],
    k: int,
) -> tuple[PreferenceMIPSMatch, ...]:
    """Rank preference48 cocktails by exact inner product, then numeric ID."""

    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if (
        query.contract.space != "sensory-preference-48"
        or query.contract.dimension != SENSORY_V2_DIMENSION
        or query.contract.metric != "inner_product"
    ):
        raise ValueError("MIPS query must use the preference48 contract")
    if any(value < 0.0 for value in query.values) or not math.isclose(
        math.fsum(query.values),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("MIPS query must be non-negative with L1 norm one")

    embeddings: list[CocktailEmbedding] = []
    for item in cocktails:
        candidate = getattr(item, "embedding", item)
        if not isinstance(candidate, CocktailEmbedding):
            raise ValueError(
                "cocktails must be Preference48 or CocktailEmbedding values"
            )
        if candidate.contract != query.contract:
            raise ValueError(
                "MIPS query and cocktails must use the exact same preference48 contract"
            )
        if any(not 0.0 <= value <= 1.0 for value in candidate.values):
            raise ValueError("preference48 cocktail values must be in [0, 1]")
        embeddings.append(candidate)
    if len({embedding.cocktail_id for embedding in embeddings}) != len(embeddings):
        raise ValueError("cocktail IDs must be unique")

    matches = tuple(
        PreferenceMIPSMatch(
            cocktail_id=embedding.cocktail_id,
            score=math.fsum(
                query_value * cocktail_value
                for query_value, cocktail_value in zip(
                    query.values,
                    embedding.values,
                    strict=True,
                )
            ),
        )
        for embedding in embeddings
    )
    return tuple(
        sorted(
            matches,
            key=lambda match: (-match.score, match.cocktail_id),
        )[:k]
    )
