from __future__ import annotations

import hashlib
import math

import pytest

from app.sensory_embedding import (
    CocktailEmbeddingAdapter,
    EmbeddingContract,
    LegacyTasteQueryAdapter,
    PositiveSelection,
    SENSORY_V2_REGISTRY,
    SensoryAxis,
    SensoryPositiveQueryEncoder,
    SensoryRegistry,
    legacy_32_cocktail_adapter,
    legacy_32_contract,
    sensory_48_cocktail_adapter,
    sensory_48_contract,
)
from app.sensory_embedding.contracts import CocktailEmbedding


SPACE_SHA256 = hashlib.sha256(b"approved-space-manifest").hexdigest()


def _sensory_contract() -> EmbeddingContract:
    return sensory_48_contract(
        registry=SENSORY_V2_REGISTRY,
        space_sha256=SPACE_SHA256,
    )


def test_exact_sensory_v2_registry_order_categories_and_hash() -> None:
    registry = SENSORY_V2_REGISTRY

    assert registry.dimension == 48
    assert [axis.axis_order for axis in registry.axes] == list(range(48))
    assert registry.axes[0].axis_id == "sweetness"
    assert registry.axes[6].axis_id == "astringency"
    assert registry.axes[43].axis_id == "coldness"
    assert registry.axes[-1].axis_id == "alcohol_intensity"
    assert registry.category_counts == {
        "taste_chemosensory": 8,
        "fruit": 7,
        "aroma": 10,
        "mouthfeel": 5,
        "finish": 9,
        "body": 4,
        "temperature": 1,
        "alcohol": 4,
    }
    assert len(registry.registry_sha256) == 64
    assert (
        SensoryRegistry(
            registry.version,
            registry.axes,
            registry.source_sha256,
        ).registry_sha256
        == registry.registry_sha256
    )


def test_registry_rejects_non_contiguous_and_duplicate_axes() -> None:
    source_sha = "0" * 64
    with pytest.raises(ValueError, match="contiguously"):
        SensoryRegistry(
            "test-registry-v1",
            (SensoryAxis(1, "one", "taste"),),
            source_sha,
        )
    with pytest.raises(ValueError, match="unique"):
        SensoryRegistry(
            "test-registry-v1",
            (
                SensoryAxis(0, "same", "taste"),
                SensoryAxis(1, "same", "taste"),
            ),
            source_sha,
        )


def test_positive_query_balances_categories_and_has_l1_one() -> None:
    encoder = SensoryPositiveQueryEncoder(
        SENSORY_V2_REGISTRY,
        _sensory_contract(),
    )

    query = encoder.encode(
        (
            PositiveSelection("citrus_fruit", 1.0),
            PositiveSelection("saltiness", 1.0),
            PositiveSelection("sweetness", 2.0),
        )
    )

    assert len(query.values) == 48
    assert math.fsum(query.values) == pytest.approx(1.0, abs=1e-15)
    assert query.values[0] == pytest.approx(1.0 / 3.0)
    assert query.values[1] == pytest.approx(1.0 / 6.0)
    assert query.values[8] == pytest.approx(1.0 / 2.0)
    assert all(value >= 0.0 for value in query.values)
    assert query.selected_ids == ("sweetness", "saltiness", "citrus_fruit")
    assert len(query.vector_sha256) == 64

    same = encoder.encode(
        (
            PositiveSelection("sweetness", 2.0),
            PositiveSelection("saltiness", 1.0),
            PositiveSelection("citrus_fruit", 1.0),
        )
    )
    assert same.vector_sha256 == query.vector_sha256


@pytest.mark.parametrize(
    ("selections", "message"),
    [
        ((), "at least one"),
        ((PositiveSelection("unknown_axis"),), "unknown"),
        (
            (
                PositiveSelection("sweetness"),
                PositiveSelection("sweetness"),
            ),
            "duplicate",
        ),
    ],
)
def test_positive_query_rejects_invalid_selection_sets(
    selections: tuple[PositiveSelection, ...],
    message: str,
) -> None:
    encoder = SensoryPositiveQueryEncoder(
        SENSORY_V2_REGISTRY,
        _sensory_contract(),
    )

    with pytest.raises(ValueError, match=message):
        encoder.encode(selections)


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_positive_selection_requires_positive_finite_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        PositiveSelection("sweetness", weight)


def test_sensory_query_rejects_contract_registry_mismatch() -> None:
    other = EmbeddingContract(
        space="sensory-preference-48",
        version="sensory-preference-48-v2",
        dimension=48,
        metric="inner_product",
        registry_sha256="0" * 64,
        space_sha256=SPACE_SHA256,
    )

    with pytest.raises(ValueError, match="registry SHA-256"):
        SensoryPositiveQueryEncoder(SENSORY_V2_REGISTRY, other)


def test_sensory_48_cocktail_adapter_validates_range_dimension_and_hash() -> None:
    contract = _sensory_contract()
    adapter = sensory_48_cocktail_adapter(contract)

    embedding = adapter.adapt(123, [0.5] * 48)

    assert isinstance(adapter, CocktailEmbeddingAdapter)
    assert embedding.id == embedding.cocktail_id == 123
    assert embedding.contract == contract
    assert len(embedding.values) == 48
    assert len(embedding.vector_sha256) == 64
    assert adapter.adapt(123, [0.5] * 48).vector_sha256 == embedding.vector_sha256

    with pytest.raises(ValueError, match="expected 48"):
        adapter.adapt(123, [0.5] * 47)
    with pytest.raises(ValueError, match="above"):
        adapter.adapt(123, [0.5] * 47 + [1.01])
    with pytest.raises(ValueError, match="finite"):
        adapter.adapt(123, [0.5] * 47 + [float("nan")])


def test_vector_dataclass_rejects_tampered_hash() -> None:
    contract = _sensory_contract()

    with pytest.raises(ValueError, match="does not match"):
        CocktailEmbedding(
            cocktail_id=1,
            values=(0.0,) * 48,
            contract=contract,
            vector_sha256="0" * 64,
        )


def test_legacy_32_cocktail_adapter_preserves_current_space() -> None:
    contract = legacy_32_contract(space_sha256=SPACE_SHA256)
    adapter = legacy_32_cocktail_adapter(contract)

    embedding = adapter.adapt(7, [1.0] + [0.0] * 31)

    assert embedding.contract.dimension == 32
    assert embedding.contract.metric == "cosine"
    assert embedding.values == (1.0,) + (0.0,) * 31
    with pytest.raises(ValueError, match="unit L2"):
        adapter.adapt(7, [1.0] * 32)


class _FakeTasteQueryModel:
    received: tuple[str, ...] | None = None

    def encode(self, codes: list[str] | tuple[str, ...]) -> tuple[float, ...]:
        self.received = tuple(codes)
        return (1.0,) + (0.0,) * 31


def test_legacy_taste_query_adapter_uses_existing_model_protocol() -> None:
    model = _FakeTasteQueryModel()
    adapter = LegacyTasteQueryAdapter(
        model=model,
        embedding_contract=legacy_32_contract(space_sha256=SPACE_SHA256),
    )

    query = adapter.encode(
        (
            PositiveSelection("fruit.citrus"),
            PositiveSelection("aroma.spicy"),
        )
    )

    assert model.received == ("aroma.spicy", "fruit.citrus")
    assert query.values == (1.0,) + (0.0,) * 31
    assert query.contract.dimension == 32
    with pytest.raises(ValueError, match="unweighted"):
        adapter.encode((PositiveSelection("fruit.citrus", 2.0),))


def test_contract_rejects_invalid_dimension_version_and_hash() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        EmbeddingContract(
            space="test-space",
            version="test-v1",
            dimension=0,
            metric="inner_product",
            registry_sha256="0" * 64,
            space_sha256="1" * 64,
        )
    with pytest.raises(ValueError, match="versioned identifier"):
        EmbeddingContract(
            space="test-space",
            version="INVALID VERSION",
            dimension=48,
            metric="inner_product",
            registry_sha256="0" * 64,
            space_sha256="1" * 64,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        EmbeddingContract(
            space="test-space",
            version="test-v1",
            dimension=48,
            metric="inner_product",
            registry_sha256="0" * 64,
            space_sha256="short",
        )
