"""The exact ordered 48-axis sensory v2 registry."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.sensory_embedding.contracts import (
    canonical_sha256,
    validate_identifier,
    validate_sha256,
)

SENSORY_V2_DIMENSION = 48
SENSORY_V2_VERSION = "sensory-48-ae-v2"
SENSORY_V2_LEVELS = ("A", "B", "C", "D", "E")
SENSORY_V2_SOURCE_SHA256 = (
    "f365ae66f7707c1c6ca15df380548f3682e488e0600a2a1a5d382fffe86c3fcf"
)


@dataclass(frozen=True, slots=True)
class SensoryAxis:
    axis_order: int
    axis_id: str
    category: str

    def __post_init__(self) -> None:
        if type(self.axis_order) is not int or self.axis_order < 0:
            raise ValueError("axis_order must be a non-negative integer")
        validate_identifier(self.axis_id, field="axis_id")
        validate_identifier(self.category, field="category")


@dataclass(frozen=True, slots=True)
class SensoryRegistry:
    version: str
    axes: tuple[SensoryAxis, ...]
    source_sha256: str
    levels: tuple[str, ...] = SENSORY_V2_LEVELS

    def __post_init__(self) -> None:
        validate_identifier(self.version, field="registry version")
        validate_sha256(self.source_sha256, field="source_sha256")
        if not isinstance(self.axes, tuple) or not all(
            isinstance(axis, SensoryAxis) for axis in self.axes
        ):
            raise ValueError("registry axes must be an immutable SensoryAxis tuple")
        if not isinstance(self.levels, tuple):
            raise ValueError("sensory registry levels must be an immutable tuple")
        if self.levels != SENSORY_V2_LEVELS:
            raise ValueError("sensory registry levels must be exactly A, B, C, D, E")
        if not self.axes:
            raise ValueError("registry must contain at least one axis")
        if tuple(axis.axis_order for axis in self.axes) != tuple(range(len(self.axes))):
            raise ValueError("registry axes must be ordered contiguously from zero")
        if len({axis.axis_id for axis in self.axes}) != len(self.axes):
            raise ValueError("registry axis_id values must be unique")

    @property
    def dimension(self) -> int:
        return len(self.axes)

    @property
    def registry_sha256(self) -> str:
        return canonical_sha256(
            {
                "axes": [
                    {
                        "axis_id": axis.axis_id,
                        "axis_order": axis.axis_order,
                        "category": axis.category,
                    }
                    for axis in self.axes
                ],
                "levels": list(self.levels),
                "source_sha256": self.source_sha256,
                "version": self.version,
            }
        )

    @property
    def category_counts(self) -> dict[str, int]:
        return dict(Counter(axis.category for axis in self.axes))

    @property
    def raw240_coordinates(self) -> tuple[tuple[str, str], ...]:
        """The immutable flattening order: axis registry order, then p_A..p_E."""

        return tuple(
            (axis.axis_id, f"p_{level}") for axis in self.axes for level in self.levels
        )


_AXES: tuple[tuple[str, str], ...] = (
    ("sweetness", "taste_chemosensory"),
    ("saltiness", "taste_chemosensory"),
    ("sourness", "taste_chemosensory"),
    ("bitterness", "taste_chemosensory"),
    ("umami", "taste_chemosensory"),
    ("pungency", "taste_chemosensory"),
    ("astringency", "taste_chemosensory"),
    ("fattiness", "taste_chemosensory"),
    ("citrus_fruit", "fruit"),
    ("berry_fruit", "fruit"),
    ("stone_fruit", "fruit"),
    ("pome_fruit", "fruit"),
    ("tropical_fruit", "fruit"),
    ("melon_fruit", "fruit"),
    ("grape_dried_fruit", "fruit"),
    ("herb_botanical_aroma", "aroma"),
    ("spice_aroma", "aroma"),
    ("floral_aroma", "aroma"),
    ("vanilla_caramel_aroma", "aroma"),
    ("nutty_aroma", "aroma"),
    ("coffee_chocolate_aroma", "aroma"),
    ("oak_smoky_aroma", "aroma"),
    ("mineral_briny_aroma", "aroma"),
    ("cream_dairy_aroma", "aroma"),
    ("malt_grain_aroma", "aroma"),
    ("refreshing_crisp_mouthfeel", "mouthfeel"),
    ("smooth_silky_mouthfeel", "mouthfeel"),
    ("creamy_velvety_mouthfeel", "mouthfeel"),
    ("juicy_mouthfeel", "mouthfeel"),
    ("carbonation_tingle", "mouthfeel"),
    ("clean_finish", "finish"),
    ("dry_finish", "finish"),
    ("sweet_finish", "finish"),
    ("sour_finish", "finish"),
    ("bitter_finish", "finish"),
    ("fruity_finish", "finish"),
    ("herb_spice_finish", "finish"),
    ("creamy_nutty_finish", "finish"),
    ("finish_length", "finish"),
    ("light_body", "body"),
    ("rich_body", "body"),
    ("heavy_body", "body"),
    ("dense_body", "body"),
    ("coldness", "temperature"),
    ("alcohol_smoothness", "alcohol"),
    ("clean_spirit_character", "alcohol"),
    ("alcohol_heat", "alcohol"),
    ("alcohol_intensity", "alcohol"),
)

SENSORY_V2_REGISTRY = SensoryRegistry(
    version=SENSORY_V2_VERSION,
    axes=tuple(
        SensoryAxis(axis_order=index, axis_id=axis_id, category=category)
        for index, (axis_id, category) in enumerate(_AXES)
    ),
    source_sha256=SENSORY_V2_SOURCE_SHA256,
)

if SENSORY_V2_REGISTRY.dimension != SENSORY_V2_DIMENSION:
    raise RuntimeError("sensory v2 registry dimension drifted")

RAW240_COORDINATES = SENSORY_V2_REGISTRY.raw240_coordinates
