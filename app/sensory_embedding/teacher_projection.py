"""Pure raw240, graph48, and preference48 teacher projections."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.sensory_embedding.adapters import (
    graph48_cocktail_adapter,
    graph48_contract,
    preference48_cocktail_adapter,
    preference48_contract,
)
from app.sensory_embedding.contracts import (
    CocktailEmbedding,
    EmbeddingContract,
    canonical_sha256,
    coerce_vector,
    validate_identifier,
    validate_sha256,
)
from app.sensory_embedding.registry import (
    RAW240_COORDINATES,
    SENSORY_V2_DIMENSION,
    SENSORY_V2_LEVELS,
    SENSORY_V2_REGISTRY,
    SENSORY_V2_VERSION,
    SensoryRegistry,
)

TEACHER_LABELS = SENSORY_V2_LEVELS
TEACHER_LEVEL_COUNT = len(TEACHER_LABELS)
TEACHER_RAW_DIMENSION = SENSORY_V2_DIMENSION * TEACHER_LEVEL_COUNT
TEACHER_SOURCE_SCHEMA = "sensory-teacher-soft-labels-240-v1"
TEACHER_PROJECTION_SCHEMA = "sensory-teacher-projection-bundle-v2"
TASTE_CHEMOSENSORY_UTILITY = (0.0, 0.2, 0.65, 1.0, 0.8)
ASSOCIATION_UTILITY = (0.0, 0.25, 0.5, 0.75, 1.0)
GRAPH48_UTILITY = ASSOCIATION_UTILITY
PREFERENCE48_NONMONOTONIC_UTILITY = TASTE_CHEMOSENSORY_UTILITY
PREFERENCE48_MONOTONIC_UTILITY = ASSOCIATION_UTILITY
PREFERENCE48_NONMONOTONIC_AXES = frozenset(
    {"sweetness", "saltiness", "sourness", "bitterness", "umami"}
)
PROBABILITY_SUM_ABS_TOLERANCE = 1e-9


class ZeroGraph48VectorError(ValueError):
    """Raised when a graph48 projection must be quarantined."""


def _probability_tuple(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != TEACHER_LEVEL_COUNT:
        raise ValueError("teacher axis probabilities must contain exactly 5 values")
    probabilities: list[float] = []
    for value in values:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError("teacher axis probabilities must be finite numbers")
        probability = float(value)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("teacher axis probabilities must be in [0, 1]")
        probabilities.append(0.0 if probability == 0.0 else probability)
    if not math.isclose(
        math.fsum(probabilities),
        1.0,
        rel_tol=0.0,
        abs_tol=PROBABILITY_SUM_ABS_TOLERANCE,
    ):
        raise ValueError("teacher axis probabilities must sum to 1")
    return tuple(probabilities)


@dataclass(frozen=True, slots=True)
class AxisSoftLabels:
    """One ordered sensory axis and its raw A-E teacher probabilities."""

    axis_id: str
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.axis_id, field="axis_id")
        canonical = _probability_tuple(self.probabilities)
        object.__setattr__(self, "probabilities", canonical)

    @classmethod
    def from_values(
        cls,
        axis_id: str,
        probabilities: Sequence[float],
    ) -> AxisSoftLabels:
        return cls(axis_id=axis_id, probabilities=tuple(probabilities))


def teacher_source_sha256(
    registry_sha256: str,
    raw_probabilities: Sequence[float],
) -> str:
    """Hash canonical raw 240D content, not a file or model-run artifact.

    The digest commits to the sensory registry, A-E label order, source schema,
    and every validated probability encoded with ``float.hex``. Cocktail
    identity and the derived 48D projection are deliberately excluded.
    """

    validate_sha256(registry_sha256, field="registry_sha256")
    if isinstance(raw_probabilities, (str, bytes)):
        raise ValueError("raw teacher probabilities must be numeric")
    received = tuple(raw_probabilities)
    if len(received) != TEACHER_RAW_DIMENSION:
        raise ValueError(
            f"expected {TEACHER_RAW_DIMENSION} dimensions, got {len(received)}"
        )
    groups = tuple(
        _probability_tuple(received[offset : offset + TEACHER_LEVEL_COUNT])
        for offset in range(0, TEACHER_RAW_DIMENSION, TEACHER_LEVEL_COUNT)
    )
    raw = tuple(value for group in groups for value in group)
    return canonical_sha256(
        {
            "labels": list(TEACHER_LABELS),
            "probabilities_hex": [value.hex() for value in raw],
            "registry_sha256": registry_sha256,
            "schema": TEACHER_SOURCE_SCHEMA,
        }
    )


def _validate_v2_registry(registry: SensoryRegistry) -> None:
    if (
        registry.version != SENSORY_V2_VERSION
        or registry.registry_sha256 != SENSORY_V2_REGISTRY.registry_sha256
        or registry.raw240_coordinates != RAW240_COORDINATES
    ):
        raise ValueError("projection requires the exact immutable sensory v2 registry")


def _raw_from_axes(
    axes: Sequence[AxisSoftLabels],
    registry: SensoryRegistry,
) -> tuple[float, ...]:
    if len(axes) != registry.dimension:
        raise ValueError(f"expected {registry.dimension} teacher axes, got {len(axes)}")
    if not all(isinstance(axis, AxisSoftLabels) for axis in axes):
        raise ValueError("teacher axes must all be AxisSoftLabels values")
    expected_ids = tuple(axis.axis_id for axis in registry.axes)
    received_ids = tuple(axis.axis_id for axis in axes)
    if received_ids != expected_ids:
        raise ValueError(
            "teacher axes must exactly follow sensory registry order "
            "(SENSORY_V2_REGISTRY)"
        )
    return tuple(
        probability for axis_labels in axes for probability in axis_labels.probabilities
    )


def _expected_intensities(
    raw: tuple[float, ...],
    registry: SensoryRegistry,
    *,
    preference: bool,
) -> tuple[float, ...]:
    return tuple(
        math.fsum(
            probability * utility
            for probability, utility in zip(
                raw[
                    axis.axis_order * TEACHER_LEVEL_COUNT : (axis.axis_order + 1)
                    * TEACHER_LEVEL_COUNT
                ],
                (
                    PREFERENCE48_NONMONOTONIC_UTILITY
                    if preference and axis.axis_id in PREFERENCE48_NONMONOTONIC_AXES
                    else PREFERENCE48_MONOTONIC_UTILITY
                ),
                strict=True,
            )
        )
        for axis in registry.axes
    )


def _project_preference48(
    raw: tuple[float, ...],
    registry: SensoryRegistry,
) -> tuple[float, ...]:
    # Preference vectors intentionally remain unnormalized for MIPS.
    return _expected_intensities(raw, registry, preference=True)


def _project_graph48(
    raw: tuple[float, ...],
    registry: SensoryRegistry,
    *,
    cocktail_id: int,
) -> tuple[float, ...]:
    expected = _expected_intensities(raw, registry, preference=False)
    category_counts = registry.category_counts
    balanced = tuple(
        expected[axis.axis_order] * math.sqrt(1.0 / category_counts[axis.category])
        for axis in registry.axes
    )
    norm = math.sqrt(math.fsum(value * value for value in balanced))
    if norm == 0.0:
        raise ZeroGraph48VectorError(
            f"cocktail {cocktail_id} graph48 projection is a zero vector; quarantined"
        )
    return tuple(value / norm for value in balanced)


@dataclass(frozen=True, slots=True)
class Raw240:
    """Canonical registry-order p_A..p_E probabilities and their content hash."""

    values: tuple[float, ...]
    registry_version: str
    registry_sha256: str
    source_sha256: str

    def __post_init__(self) -> None:
        if (
            self.registry_version != SENSORY_V2_VERSION
            or self.registry_sha256 != SENSORY_V2_REGISTRY.registry_sha256
        ):
            raise ValueError("raw240 requires the exact sensory v2 registry contract")
        received = tuple(self.values)
        if len(received) != TEACHER_RAW_DIMENSION:
            raise ValueError(
                f"expected {TEACHER_RAW_DIMENSION} dimensions, got {len(received)}"
            )
        canonical = tuple(
            value
            for offset in range(0, TEACHER_RAW_DIMENSION, TEACHER_LEVEL_COUNT)
            for value in _probability_tuple(
                received[offset : offset + TEACHER_LEVEL_COUNT]
            )
        )
        object.__setattr__(self, "values", canonical)
        expected_source = teacher_source_sha256(self.registry_sha256, canonical)
        if self.source_sha256 != expected_source:
            raise ValueError("raw240 source SHA-256 does not match its probabilities")

    @property
    def coordinates(self) -> tuple[tuple[str, str], ...]:
        return RAW240_COORDINATES

    def probabilities_for(self, axis_id: str) -> tuple[float, ...]:
        by_id = {axis.axis_id: axis for axis in SENSORY_V2_REGISTRY.axes}
        try:
            axis = by_id[axis_id]
        except KeyError as error:
            raise ValueError(f"unknown sensory axis: {axis_id}") from error
        start = axis.axis_order * TEACHER_LEVEL_COUNT
        return self.values[start : start + TEACHER_LEVEL_COUNT]


@dataclass(frozen=True, slots=True)
class Graph48:
    """Category-balanced, unit-L2 cocktail vector for cosine graph topology."""

    embedding: CocktailEmbedding
    source_sha256: str

    def __post_init__(self) -> None:
        validate_sha256(self.source_sha256, field="source_sha256")
        contract = self.embedding.contract
        if (
            contract.space != "sensory-graph-48"
            or contract.dimension != SENSORY_V2_DIMENSION
            or contract.metric != "cosine"
            or contract.registry_sha256 != SENSORY_V2_REGISTRY.registry_sha256
        ):
            raise ValueError("graph48 requires the matching 48D cosine contract")
        if any(not 0.0 <= value <= 1.0 for value in self.embedding.values):
            raise ValueError("graph48 values must be in [0, 1]")
        norm = math.sqrt(math.fsum(value * value for value in self.embedding.values))
        if norm == 0.0:
            raise ZeroGraph48VectorError("graph48 zero vector is quarantined")
        if not math.isclose(norm, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("graph48 values must have unit L2 norm")

    @property
    def cocktail_id(self) -> int:
        return self.embedding.cocktail_id

    @property
    def values(self) -> tuple[float, ...]:
        return self.embedding.values

    @property
    def contract(self) -> EmbeddingContract:
        return self.embedding.contract

    @property
    def vector_sha256(self) -> str:
        return self.embedding.vector_sha256


@dataclass(frozen=True, slots=True)
class Preference48:
    """Unnormalized cocktail affinities for user-query inner-product search."""

    embedding: CocktailEmbedding
    source_sha256: str

    def __post_init__(self) -> None:
        validate_sha256(self.source_sha256, field="source_sha256")
        contract = self.embedding.contract
        if (
            contract.space != "sensory-preference-48"
            or contract.dimension != SENSORY_V2_DIMENSION
            or contract.metric != "inner_product"
            or contract.registry_sha256 != SENSORY_V2_REGISTRY.registry_sha256
        ):
            raise ValueError(
                "preference48 requires the matching 48D inner-product contract"
            )
        if any(not 0.0 <= value <= 1.0 for value in self.embedding.values):
            raise ValueError("preference48 values must be in [0, 1]")

    @property
    def cocktail_id(self) -> int:
        return self.embedding.cocktail_id

    @property
    def values(self) -> tuple[float, ...]:
        return self.embedding.values

    @property
    def contract(self) -> EmbeddingContract:
        return self.embedding.contract

    @property
    def vector_sha256(self) -> str:
        return self.embedding.vector_sha256


def projection_provenance_sha256(
    *,
    cocktail_id: int,
    raw240: Raw240,
    graph48: Graph48,
    preference48: Preference48,
) -> str:
    return canonical_sha256(
        {
            "cocktail_id": cocktail_id,
            "graph_contract_sha256": graph48.contract.contract_sha256,
            "graph_vector_sha256": graph48.vector_sha256,
            "preference_contract_sha256": preference48.contract.contract_sha256,
            "preference_vector_sha256": preference48.vector_sha256,
            "raw_source_sha256": raw240.source_sha256,
            "registry_sha256": raw240.registry_sha256,
            "registry_version": raw240.registry_version,
            "schema": TEACHER_PROJECTION_SCHEMA,
        }
    )


@dataclass(frozen=True, slots=True)
class TeacherEmbeddingBundle:
    """One immutable raw240 source and its two deliberately distinct projections."""

    raw240: Raw240
    graph48: Graph48
    preference48: Preference48
    provenance_sha256: str

    def __post_init__(self) -> None:
        if self.graph48.cocktail_id != self.preference48.cocktail_id:
            raise ValueError("graph48 and preference48 cocktail IDs must match")
        if not (
            self.raw240.source_sha256
            == self.graph48.source_sha256
            == self.preference48.source_sha256
        ):
            raise ValueError("all projections must retain the raw240 source SHA-256")
        expected_graph = _project_graph48(
            self.raw240.values,
            SENSORY_V2_REGISTRY,
            cocktail_id=self.graph48.cocktail_id,
        )
        if self.graph48.values != expected_graph:
            raise ValueError("graph48 does not match the raw240 graph projection")
        expected_preference = _project_preference48(
            self.raw240.values,
            SENSORY_V2_REGISTRY,
        )
        if self.preference48.values != expected_preference:
            raise ValueError(
                "preference48 does not match the raw240 preference projection"
            )
        expected = projection_provenance_sha256(
            cocktail_id=self.graph48.cocktail_id,
            raw240=self.raw240,
            graph48=self.graph48,
            preference48=self.preference48,
        )
        if self.provenance_sha256 != expected:
            raise ValueError("projection provenance SHA-256 does not match bundle")

    @property
    def cocktail_id(self) -> int:
        return self.graph48.cocktail_id

    @property
    def source_sha256(self) -> str:
        return self.raw240.source_sha256

    @property
    def registry_sha256(self) -> str:
        return self.raw240.registry_sha256


def project_teacher_soft_labels(
    cocktail_id: int,
    axes: Sequence[AxisSoftLabels],
    *,
    registry: SensoryRegistry = SENSORY_V2_REGISTRY,
    graph_contract: EmbeddingContract | None = None,
    preference_contract: EmbeddingContract | None = None,
) -> TeacherEmbeddingBundle:
    """Build raw240, graph48, and preference48 without I/O or hidden normalization."""

    _validate_v2_registry(registry)
    raw_values = _raw_from_axes(axes, registry)
    source_sha256 = teacher_source_sha256(registry.registry_sha256, raw_values)
    raw240 = Raw240(
        values=raw_values,
        registry_version=registry.version,
        registry_sha256=registry.registry_sha256,
        source_sha256=source_sha256,
    )
    graph_embedding = graph48_cocktail_adapter(
        graph_contract or graph48_contract(registry=registry)
    ).adapt(
        cocktail_id,
        _project_graph48(raw240.values, registry, cocktail_id=cocktail_id),
    )
    preference_embedding = preference48_cocktail_adapter(
        preference_contract or preference48_contract(registry=registry)
    ).adapt(
        cocktail_id,
        _project_preference48(raw240.values, registry),
    )
    graph = Graph48(embedding=graph_embedding, source_sha256=source_sha256)
    preference = Preference48(
        embedding=preference_embedding,
        source_sha256=source_sha256,
    )
    provenance_sha256 = projection_provenance_sha256(
        cocktail_id=cocktail_id,
        raw240=raw240,
        graph48=graph,
        preference48=preference,
    )
    return TeacherEmbeddingBundle(
        raw240=raw240,
        graph48=graph,
        preference48=preference,
        provenance_sha256=provenance_sha256,
    )


@dataclass(frozen=True, slots=True)
class TeacherProjection:
    """Auditable raw teacher content plus its sensory-48 embedding."""

    raw_probabilities: tuple[float, ...]
    source_sha256: str
    projected_values: tuple[float, ...]
    embedding: CocktailEmbedding

    def __post_init__(self) -> None:
        raw = coerce_vector(self.raw_probabilities, TEACHER_RAW_DIMENSION)
        projected = coerce_vector(
            self.projected_values,
            SENSORY_V2_DIMENSION,
        )
        if raw != self.raw_probabilities or projected != self.projected_values:
            raise ValueError("teacher projection vectors must be canonical tuples")
        expected_source = teacher_source_sha256(
            self.embedding.contract.registry_sha256,
            raw,
        )
        if self.source_sha256 != expected_source:
            raise ValueError("teacher source SHA-256 does not match raw probabilities")
        contract = self.embedding.contract
        if (
            contract.space != "sensory-preference-48"
            or contract.dimension != SENSORY_V2_DIMENSION
            or contract.metric != "inner_product"
            or contract.registry_sha256 != SENSORY_V2_REGISTRY.registry_sha256
        ):
            raise ValueError(
                "teacher projection requires the matching sensory_48 contract"
            )
        expected_projection = _project_preference48(
            raw,
            SENSORY_V2_REGISTRY,
        )
        if projected != expected_projection:
            raise ValueError(
                "projected values do not match the raw teacher utility projection"
            )
        if projected != self.embedding.values:
            raise ValueError("projected values must equal CocktailEmbedding values")

    @property
    def cocktail_id(self) -> int:
        return self.embedding.cocktail_id


@dataclass(frozen=True, slots=True)
class TeacherSoftLabelProjector:
    """Validate registry-ordered A-E distributions and derive one 48D vector."""

    contract: EmbeddingContract
    registry: SensoryRegistry = SENSORY_V2_REGISTRY

    def __post_init__(self) -> None:
        _validate_v2_registry(self.registry)
        if (
            self.contract.space != "sensory-preference-48"
            or self.contract.dimension != SENSORY_V2_DIMENSION
            or self.contract.metric != "inner_product"
            or self.contract.registry_sha256 != self.registry.registry_sha256
        ):
            raise ValueError(
                "teacher projection requires the matching sensory_48 contract"
            )

    def project(
        self,
        cocktail_id: int,
        axes: Sequence[AxisSoftLabels],
    ) -> TeacherProjection:
        raw = _raw_from_axes(axes, self.registry)
        projected = _project_preference48(raw, self.registry)
        embedding = preference48_cocktail_adapter(self.contract).adapt(
            cocktail_id,
            projected,
        )
        return TeacherProjection(
            raw_probabilities=raw,
            source_sha256=teacher_source_sha256(
                self.registry.registry_sha256,
                raw,
            ),
            projected_values=projected,
            embedding=embedding,
        )
