from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from app.sensory_embedding.adapters import (
    graph48_contract,
    preference48_contract,
)
from app.sensory_embedding.query import (
    build_user_query,
    score_preference_mips,
)
from app.sensory_embedding.registry import (
    RAW240_COORDINATES,
    SENSORY_V2_REGISTRY,
)
from app.sensory_embedding.teacher_projection import (
    AxisSoftLabels,
    Graph48,
    Preference48,
    Raw240,
    TeacherEmbeddingBundle,
    ZeroGraph48VectorError,
    project_teacher_soft_labels,
)


def _axes(
    distributions: dict[str, tuple[float, ...]] | None = None,
) -> tuple[AxisSoftLabels, ...]:
    by_id = distributions or {}
    return tuple(
        AxisSoftLabels.from_values(
            axis.axis_id,
            by_id.get(axis.axis_id, (1.0, 0.0, 0.0, 0.0, 0.0)),
        )
        for axis in SENSORY_V2_REGISTRY.axes
    )


def test_registry_and_raw240_pin_axis_then_pa_through_pe_order() -> None:
    axes = tuple(
        AxisSoftLabels.from_values(
            axis.axis_id,
            tuple(1.0 if level == axis.axis_order % 5 else 0.0 for level in range(5)),
        )
        for axis in SENSORY_V2_REGISTRY.axes
    )
    bundle = project_teacher_soft_labels(17, axes)

    assert isinstance(bundle, TeacherEmbeddingBundle)
    assert isinstance(bundle.raw240, Raw240)
    assert len(bundle.raw240.values) == 240
    assert bundle.raw240.coordinates == RAW240_COORDINATES
    assert RAW240_COORDINATES[:6] == (
        ("sweetness", "p_A"),
        ("sweetness", "p_B"),
        ("sweetness", "p_C"),
        ("sweetness", "p_D"),
        ("sweetness", "p_E"),
        ("saltiness", "p_A"),
    )
    for axis in SENSORY_V2_REGISTRY.axes:
        start = axis.axis_order * 5
        assert (
            bundle.raw240.probabilities_for(axis.axis_id)
            == bundle.raw240.values[start : start + 5]
            == axes[axis.axis_order].probabilities
        )
    assert bundle.source_sha256 == bundle.raw240.source_sha256
    assert bundle.graph48.source_sha256 == bundle.source_sha256
    assert bundle.preference48.source_sha256 == bundle.source_sha256
    assert len(bundle.provenance_sha256) == 64

    with pytest.raises(FrozenInstanceError):
        SENSORY_V2_REGISTRY.version = "drift"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        bundle.raw240.values = (0.0,) * 240  # type: ignore[misc]


def test_graph48_and_preference48_are_distinct_projection_contracts() -> None:
    bundle = project_teacher_soft_labels(
        1,
        _axes(
            {
                "sweetness": (0.0, 0.0, 0.0, 0.0, 1.0),
                "pungency": (0.0, 0.0, 0.0, 0.0, 1.0),
                "astringency": (0.0, 0.0, 0.0, 0.0, 1.0),
                "fattiness": (0.0, 0.0, 0.0, 0.0, 1.0),
                "citrus_fruit": (0.0, 0.0, 0.0, 0.0, 1.0),
            }
        ),
    )

    assert isinstance(bundle.graph48, Graph48)
    assert bundle.graph48.contract == graph48_contract(registry=SENSORY_V2_REGISTRY)
    assert bundle.graph48.contract.metric == "cosine"
    assert math.sqrt(math.fsum(x * x for x in bundle.graph48.values)) == pytest.approx(
        1.0,
        abs=1e-15,
    )
    assert bundle.graph48.values[0] / bundle.graph48.values[8] == pytest.approx(
        math.sqrt(7.0 / 8.0)
    )

    assert isinstance(bundle.preference48, Preference48)
    assert bundle.preference48.contract == preference48_contract(
        registry=SENSORY_V2_REGISTRY
    )
    assert bundle.preference48.contract.metric == "inner_product"
    assert bundle.preference48.values[0] == 0.8
    assert bundle.preference48.values[4] == 0.0
    assert bundle.preference48.values[5] == 1.0
    assert bundle.preference48.values[6] == 1.0
    assert bundle.preference48.values[7] == 1.0
    assert bundle.preference48.values[8] == 1.0
    assert math.sqrt(
        math.fsum(x * x for x in bundle.preference48.values)
    ) != pytest.approx(1.0)


def test_graph48_quarantines_zero_vector() -> None:
    with pytest.raises(ZeroGraph48VectorError, match="quarantined"):
        project_teacher_soft_labels(99, _axes())


def test_build_user_query_is_sparse_and_equal_mass_per_active_category() -> None:
    query = build_user_query(
        {
            "citrus_fruit": 1.0,
            "saltiness": 1.0,
            "sweetness": 2.0,
        }
    )

    assert query.values[0] == pytest.approx(1.0 / 3.0)
    assert query.values[1] == pytest.approx(1.0 / 6.0)
    assert query.values[8] == pytest.approx(1.0 / 2.0)
    assert math.fsum(query.values[:8]) == pytest.approx(0.5)
    assert math.fsum(query.values[8:15]) == pytest.approx(0.5)
    assert all(
        value == 0.0
        for index, value in enumerate(query.values)
        if index not in {0, 1, 8}
    )
    assert query.contract.metric == "inner_product"

    with pytest.raises(ValueError, match="at least one"):
        build_user_query({})


def test_score_preference_mips_uses_exact_contract_and_numeric_tie_break() -> None:
    query = build_user_query({"sweetness": 1.0})
    contract = query.contract
    first = project_teacher_soft_labels(
        10,
        _axes({"sweetness": (0.0, 0.0, 0.0, 1.0, 0.0)}),
        preference_contract=contract,
    )
    second = project_teacher_soft_labels(
        2,
        _axes({"sweetness": (0.0, 0.0, 0.0, 0.0, 1.0)}),
        preference_contract=contract,
    )
    tied = project_teacher_soft_labels(
        1,
        _axes({"sweetness": (0.0, 0.0, 0.0, 0.0, 1.0)}),
        preference_contract=contract,
    )

    matches = score_preference_mips(
        query,
        (first.preference48, second.preference48, tied.preference48),
        k=3,
    )

    assert [(match.cocktail_id, match.score) for match in matches] == [
        (10, 1.0),
        (1, 0.8),
        (2, 0.8),
    ]
    assert all(match.negative_inner_product == -match.score for match in matches)

    with pytest.raises(ValueError, match="exact same preference48"):
        score_preference_mips(query, (first.graph48,), k=1)
