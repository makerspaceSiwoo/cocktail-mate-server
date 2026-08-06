from __future__ import annotations

import hashlib
import math

import pytest

from app.sensory_embedding import (
    ASSOCIATION_UTILITY,
    SENSORY_V2_REGISTRY,
    TASTE_CHEMOSENSORY_UTILITY,
    TEACHER_LABELS,
    TEACHER_RAW_DIMENSION,
    TEACHER_SOURCE_SCHEMA,
    AxisSoftLabels,
    TeacherProjection,
    TeacherSoftLabelProjector,
    sensory_48_contract,
    teacher_source_sha256,
)
from app.sensory_embedding.contracts import (
    CocktailEmbedding,
    EmbeddingContract,
    canonical_sha256,
    vector_sha256,
)

SPACE_SHA256 = hashlib.sha256(b"teacher-projection-space-v1").hexdigest()
PROBABILITIES = (0.0, 0.1, 0.2, 0.3, 0.4)


def _contract() -> EmbeddingContract:
    return sensory_48_contract(
        registry=SENSORY_V2_REGISTRY,
        space_sha256=SPACE_SHA256,
    )


def _axes(
    probabilities: tuple[float, ...] = PROBABILITIES,
) -> tuple[AxisSoftLabels, ...]:
    return tuple(
        AxisSoftLabels.from_values(axis.axis_id, probabilities)
        for axis in SENSORY_V2_REGISTRY.axes
    )


def test_projection_preserves_raw_registry_order_and_builds_embedding() -> None:
    projector = TeacherSoftLabelProjector(_contract())

    result = projector.project(17, _axes())

    expected_taste = math.fsum(
        probability * utility
        for probability, utility in zip(
            PROBABILITIES,
            TASTE_CHEMOSENSORY_UTILITY,
            strict=True,
        )
    )
    expected_association = math.fsum(
        probability * utility
        for probability, utility in zip(
            PROBABILITIES,
            ASSOCIATION_UTILITY,
            strict=True,
        )
    )
    assert len(result.raw_probabilities) == TEACHER_RAW_DIMENSION == 240
    assert result.raw_probabilities[:5] == PROBABILITIES
    assert result.raw_probabilities[-5:] == PROBABILITIES
    assert result.projected_values[:5] == pytest.approx((expected_taste,) * 5)
    assert result.projected_values[5:8] == pytest.approx((expected_association,) * 3)
    assert result.projected_values[8:] == pytest.approx((expected_association,) * 40)
    assert result.cocktail_id == result.embedding.cocktail_id == 17
    assert result.embedding.values == result.projected_values
    assert result.embedding.contract == _contract()
    assert len(result.embedding.vector_sha256) == 64


def test_only_five_basic_tastes_use_nonmonotonic_e() -> None:
    axes = list(_axes((0.0, 0.0, 0.0, 0.0, 1.0)))

    result = TeacherSoftLabelProjector(_contract()).project(1, tuple(axes))

    assert result.projected_values[0] == 0.8
    assert result.projected_values[4] == 0.8
    assert result.projected_values[5] == 1.0
    assert result.projected_values[6] == 1.0
    assert result.projected_values[7] == 1.0
    assert result.projected_values[8] == 1.0
    assert result.projected_values[-1] == 1.0


def test_source_sha_is_canonical_raw_content_not_cocktail_identity() -> None:
    projector = TeacherSoftLabelProjector(_contract())

    first = projector.project(1, _axes())
    second = projector.project(2, _axes())
    expected = canonical_sha256(
        {
            "labels": list(TEACHER_LABELS),
            "probabilities_hex": [value.hex() for value in first.raw_probabilities],
            "registry_sha256": SENSORY_V2_REGISTRY.registry_sha256,
            "schema": TEACHER_SOURCE_SCHEMA,
        }
    )

    assert first.source_sha256 == second.source_sha256 == expected
    assert TEACHER_SOURCE_SCHEMA == "sensory-teacher-soft-labels-240-v1"
    assert first.embedding.vector_sha256 != second.embedding.vector_sha256
    assert (
        teacher_source_sha256(
            SENSORY_V2_REGISTRY.registry_sha256,
            first.raw_probabilities,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((0.0, 0.0, 0.0, 1.0), "exactly 5"),
        ((0.0, 0.0, 0.0, 0.0, 0.5), "sum to 1"),
        ((0.0, 0.0, 0.0, -0.1, 1.1), r"\[0, 1\]"),
        ((0.0, 0.0, 0.0, float("nan"), 1.0), "finite"),
        ((0.0, 0.0, 0.0, float("inf"), 1.0), "finite"),
        ((0.0, 0.0, 0.0, False, 1.0), "finite"),
    ],
)
def test_axis_distribution_rejects_invalid_probabilities(
    values: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AxisSoftLabels.from_values("sweetness", values)


def test_projector_rejects_missing_reordered_and_wrong_axis_ids() -> None:
    projector = TeacherSoftLabelProjector(_contract())
    axes = _axes()

    with pytest.raises(ValueError, match="expected 48"):
        projector.project(1, axes[:-1])
    with pytest.raises(ValueError, match="registry order"):
        projector.project(1, (axes[1], axes[0], *axes[2:]))
    wrong = (
        AxisSoftLabels.from_values("wrong_axis", PROBABILITIES),
        *axes[1:],
    )
    with pytest.raises(ValueError, match="registry order"):
        projector.project(1, wrong)


def test_projector_rejects_non_sensory_contract() -> None:
    wrong = EmbeddingContract(
        space="other-space",
        version="other-space-v1",
        dimension=48,
        metric="inner_product",
        registry_sha256=SENSORY_V2_REGISTRY.registry_sha256,
        space_sha256=SPACE_SHA256,
    )

    with pytest.raises(ValueError, match="sensory_48 contract"):
        TeacherSoftLabelProjector(wrong)


def test_projection_dataclass_rejects_tampered_source_or_projection() -> None:
    result = TeacherSoftLabelProjector(_contract()).project(4, _axes())

    with pytest.raises(ValueError, match="source SHA-256"):
        TeacherProjection(
            raw_probabilities=result.raw_probabilities,
            source_sha256="0" * 64,
            projected_values=result.projected_values,
            embedding=result.embedding,
        )
    with pytest.raises(ValueError, match="utility projection"):
        TeacherProjection(
            raw_probabilities=result.raw_probabilities,
            source_sha256=result.source_sha256,
            projected_values=(0.0,) * 48,
            embedding=result.embedding,
        )

    wrong_contract = EmbeddingContract(
        space="other-space",
        version="other-space-v1",
        dimension=48,
        metric="inner_product",
        registry_sha256=SENSORY_V2_REGISTRY.registry_sha256,
        space_sha256=SPACE_SHA256,
    )
    wrong_embedding = CocktailEmbedding(
        cocktail_id=result.cocktail_id,
        values=result.projected_values,
        contract=wrong_contract,
        vector_sha256=vector_sha256(
            wrong_contract,
            result.projected_values,
            identity=f"cocktail:{result.cocktail_id}",
        ),
    )
    with pytest.raises(ValueError, match="sensory_48 contract"):
        TeacherProjection(
            raw_probabilities=result.raw_probabilities,
            source_sha256=result.source_sha256,
            projected_values=result.projected_values,
            embedding=wrong_embedding,
        )


def test_source_hash_rejects_noncanonical_raw_shape_and_axis_sum() -> None:
    registry_sha = SENSORY_V2_REGISTRY.registry_sha256

    with pytest.raises(ValueError, match="expected 240"):
        teacher_source_sha256(registry_sha, (0.0,) * 239)
    invalid = list(_axes()[0].probabilities) * 48
    invalid[0] = 0.1
    with pytest.raises(ValueError, match="sum to 1"):
        teacher_source_sha256(registry_sha, invalid)
