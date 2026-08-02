"""Pure helpers and configuration for cocktail image generation."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
from csv import DictReader
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from PIL import Image, ImageFilter
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

MAIN_IMAGE_SIZE = (400, 300)
THUMBNAIL_IMAGE_SIZE = (128, 96)
CONTENTS_PROMPT_VERSION = 2
IMAGE_OUTPUT_VERSION = 4
IMAGE_PROMPT_MAX_CHARS = 800


def _prompt_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "prompts" / filename


class ImageGenerationSettings(BaseSettings):
    """Settings used only by the offline image-generation worker."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: SecretStr = SecretStr("")
    gemini_text_model: str = "gemini-3.5-flash-lite"
    gemini_min_request_interval_seconds: float = 32.0
    gemini_max_transient_retries: int = 3
    cocktail_image_base_url: str = "https://api.cocktail-mate.com/media/cocktails"
    cocktail_image_output_dir: Path = Path("/srv/cocktail-mate/media/cocktails")
    cocktail_image_state_dir: Path = Path("/srv/cocktail-mate/image-generation-state")
    cocktail_image_background_prompts_path: Path = Field(
        default_factory=lambda: _prompt_path("base_backgrounds.csv")
    )
    cocktail_image_glass_prompts_path: Path = Field(
        default_factory=lambda: _prompt_path("glass_prompts.csv")
    )
    cocktail_image_composition_prompt_path: Path = Field(
        default_factory=lambda: _prompt_path("composition.txt")
    )


@dataclass(frozen=True, slots=True)
class CocktailSource:
    id: int
    name: str
    name_en: str | None
    glass: str | None
    base_tag: str | None
    recipe: tuple[str, ...]
    image_url: str | None = None

    @property
    def validation_error(self) -> str | None:
        if not self.name.strip():
            return "name is required"
        if not self.glass or not self.glass.strip():
            return "glass is required"
        if not self.recipe:
            return "recipe is required"
        return None


GenerationStatus = Literal[
    "prompt_ready",
    "quota_wait",
    "image_ready",
    "completed",
    "failed",
]


class PromptExpansion(BaseModel):
    cocktail_contents: str = Field(
        description=(
            "A detailed English image-generation prompt describing only the visible "
            "cocktail contents, ice, layers, foam, garnish, and condensation."
        ),
    )


class GenerationState(BaseModel):
    cocktail_id: int
    input_fingerprint: str
    contents_fingerprint: str | None = None
    text_model: str
    image_model: str
    status: GenerationStatus
    cocktail_contents_prompt: str | None = None
    glass_prompt: str | None = None
    background_prompt: str | None = None
    composition_prompt: str | None = None
    expanded_prompt: str | None = None
    main_filename: str | None = None
    thumbnail_filename: str | None = None
    image_sha256: str | None = None
    attempts: int = 0
    last_error: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class BackgroundPrompt:
    base_tag: str
    display_name: str
    color_hex: str
    prompt: str


@dataclass(frozen=True, slots=True)
class PromptParts:
    glass: str
    background: str
    composition: str


@dataclass(frozen=True, slots=True)
class PromptCatalog:
    backgrounds: dict[str, BackgroundPrompt]
    glasses: dict[str, str]
    composition: str

    def parts_for(self, cocktail: CocktailSource) -> PromptParts:
        base_tag = (cocktail.base_tag or "other").strip().lower()
        background = self.backgrounds.get(base_tag)
        if background is None:
            raise ValueError(
                f"No background prompt configured for base_tag={base_tag!r}"
            )

        glass_key = _normalize_lookup_key(cocktail.glass or "")
        glass_shape = self.glasses.get(glass_key)
        if glass_shape is None:
            raise ValueError(f"No glass prompt configured for glass={cocktail.glass!r}")

        return PromptParts(
            glass=glass_shape,
            background=background.prompt,
            composition=self.composition,
        )


def _normalize_lookup_key(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _load_nonempty_text(path: Path, label: str) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"{label} prompt is empty: {path}")
    return value


def _load_backgrounds(path: Path) -> dict[str, BackgroundPrompt]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = DictReader(file)
        expected = {"base_tag", "display_name", "color_hex", "prompt"}
        if set(rows.fieldnames or ()) != expected:
            raise ValueError(
                f"Background CSV columns must be {sorted(expected)}: {path}"
            )
        result: dict[str, BackgroundPrompt] = {}
        for row in rows:
            base_tag = _normalize_lookup_key(row["base_tag"])
            if not base_tag or base_tag in result:
                raise ValueError(f"Invalid or duplicate base_tag={base_tag!r}: {path}")
            color_hex = row["color_hex"].strip().lower()
            if len(color_hex) != 7 or not color_hex.startswith("#"):
                raise ValueError(f"Invalid color_hex={color_hex!r}: {path}")
            prompt = row["prompt"].strip()
            if not prompt:
                raise ValueError(f"Empty background prompt for {base_tag}: {path}")
            result[base_tag] = BackgroundPrompt(
                base_tag=base_tag,
                display_name=row["display_name"].strip(),
                color_hex=color_hex,
                prompt=prompt,
            )
    if "other" not in result:
        raise ValueError(f"Background CSV requires an 'other' fallback: {path}")
    return result


def _load_glasses(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = DictReader(file)
        expected = {"glass", "prompt"}
        if set(rows.fieldnames or ()) != expected:
            raise ValueError(f"Glass CSV columns must be {sorted(expected)}: {path}")
        result: dict[str, str] = {}
        for row in rows:
            glass = _normalize_lookup_key(row["glass"])
            if not glass or glass in result:
                raise ValueError(f"Invalid or duplicate glass={glass!r}: {path}")
            prompt = row["prompt"].strip()
            if not prompt:
                raise ValueError(f"Empty glass prompt for {glass}: {path}")
            result[glass] = prompt
    return result


def load_prompt_catalog(settings: ImageGenerationSettings) -> PromptCatalog:
    return PromptCatalog(
        backgrounds=_load_backgrounds(settings.cocktail_image_background_prompts_path),
        glasses=_load_glasses(settings.cocktail_image_glass_prompts_path),
        composition=_load_nonempty_text(
            settings.cocktail_image_composition_prompt_path,
            "Composition",
        ),
    )


def build_contents_expansion_prompt(cocktail: CocktailSource) -> str:
    """Build facts for a model that describes only the drink contents."""

    recipe_lines = "\n".join(
        f"{index}. {step.strip()}"
        for index, step in enumerate(cocktail.recipe, start=1)
        if step.strip()
    )
    english_name = cocktail.name_en.strip() if cocktail.name_en else "(not provided)"
    return f"""You are writing section 1 of a production prompt for a cocktail photograph.

Use the recipe as the primary factual source. The cocktail names may help you
recognize a classic presentation, but the recipe overrides prior knowledge.
Describe only the visible drink contents: liquid color and clarity, distinct
layers, foam, ice, garnish, and condensation on the glass. Do not describe or
choose the glass shape, background, lighting, camera, composition,
image size, taste, aroma, alcohol strength, or drinking experience. Do not
invent branded bottles or printed labels. Mention foam, cream, garnish, or a
layer only when the recipe or a well-established classic presentation clearly
supports it. Stirring or shaking alone is not evidence of foam. When evidence
is uncertain, omit the attribute instead of guessing.

Cocktail content facts:
- Korean name: {cocktail.name.strip()}
- English name: {english_name}
- Recipe:
{recipe_lines}

Return one concise but visually detailed English paragraph for the cocktail
contents only, using no more than 250 characters. Start with the cocktail's
English name when available. Do not repeat any instructions or add headings.
"""


def _compact_at_word_boundary(value: str, max_chars: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    candidate = normalized[: max_chars - 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return f"{candidate.rstrip('.')}."


def build_final_prompt(cocktail_contents: str, parts: PromptParts) -> str:
    """Concatenate the dynamic drink description with stable prompt fragments."""

    fixed_sections = (
        f"Glass: {parts.glass.strip()}",
        f"Background: {parts.background.strip()}",
        f"Style: {parts.composition.strip()}",
    )
    fixed_text = "\n".join(fixed_sections)
    drink_prefix = "Drink: "
    available_for_drink = (
        IMAGE_PROMPT_MAX_CHARS - len(drink_prefix) - len(fixed_text) - 1
    )
    if available_for_drink < 80:
        raise ValueError("Stable image prompt fragments leave too little drink space")
    compact_contents = _compact_at_word_boundary(
        cocktail_contents,
        available_for_drink,
    )
    final_prompt = f"{drink_prefix}{compact_contents}\n{fixed_text}"
    if len(final_prompt) > IMAGE_PROMPT_MAX_CHARS:
        raise ValueError("Final image prompt exceeds 800 characters")
    return final_prompt


def build_contents_fingerprint(
    cocktail: CocktailSource,
    *,
    text_model: str,
) -> str:
    """Fingerprint only facts used by the reusable contents expansion call."""

    payload = {
        "name": cocktail.name,
        "name_en": cocktail.name_en,
        "recipe": cocktail.recipe,
        "text_model": text_model,
        "contents_prompt_version": CONTENTS_PROMPT_VERSION,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_input_fingerprint(
    cocktail: CocktailSource,
    parts: PromptParts,
    *,
    text_model: str,
    image_model: str,
    image_config: Mapping[str, object] | None = None,
) -> str:
    payload = {
        "id": cocktail.id,
        "name": cocktail.name,
        "name_en": cocktail.name_en,
        "glass": cocktail.glass,
        "base_tag": cocktail.base_tag,
        "recipe": cocktail.recipe,
        "glass_prompt": parts.glass,
        "background_prompt": parts.background,
        "composition_prompt": parts.composition,
        "contents_prompt_version": CONTENTS_PROMPT_VERSION,
        "text_model": text_model,
        "image_model": image_model,
        "image_config": dict(image_config or {}),
        "output_version": IMAGE_OUTPUT_VERSION,
        "main_size": MAIN_IMAGE_SIZE,
        "thumbnail_size": THUMBNAIL_IMAGE_SIZE,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class ModelRateLimiter:
    """Minimum-start-interval limiter, keyed by provider model name."""

    def __init__(
        self,
        interval_seconds: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if interval_seconds < 0:
            raise ValueError("interval_seconds must be non-negative")
        self._interval_seconds = interval_seconds
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_started_at: dict[str, float] = {}

    def wait(self, model: str) -> float:
        now = self._monotonic()
        last_started_at = self._last_started_at.get(model)
        waited = 0.0
        if last_started_at is not None:
            waited = max(0.0, self._interval_seconds - (now - last_started_at))
            if waited:
                self._sleep(waited)
        self._last_started_at[model] = self._monotonic()
        return waited


def _crop_to_ratio(
    image: Image.Image, width_ratio: int, height_ratio: int
) -> Image.Image:
    target_ratio = width_ratio / height_ratio
    source_ratio = image.width / image.height
    if source_ratio > target_ratio:
        crop_width = round(image.height * target_ratio)
        left = (image.width - crop_width) // 2
        box = (left, 0, left + crop_width, image.height)
    else:
        crop_height = round(image.width / target_ratio)
        top = (image.height - crop_height) // 2
        box = (0, top, image.width, top + crop_height)
    return image.crop(box)


def _encode_webp(image: Image.Image, *, size: tuple[int, int], quality: int) -> bytes:
    resized = image.resize(size, Image.Resampling.LANCZOS)
    sharpen_radius = 0.7 if size == MAIN_IMAGE_SIZE else 0.45
    resized = resized.filter(
        ImageFilter.UnsharpMask(
            radius=sharpen_radius,
            percent=130,
            threshold=2,
        )
    )
    output = io.BytesIO()
    resized.save(output, format="WEBP", quality=quality, method=6)
    return output.getvalue()


def validate_webp(data: bytes, expected_size: tuple[int, int]) -> None:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.format != "WEBP":
            raise ValueError(f"Expected WEBP, got {image.format}")
        if image.size != expected_size:
            raise ValueError(f"Expected image size {expected_size}, got {image.size}")


def create_image_variants(
    raw_image: bytes,
) -> tuple[bytes, bytes, str]:
    """Decode one generated image and return main, thumbnail, and main SHA-256."""

    with Image.open(io.BytesIO(raw_image)) as source:
        source.load()
        if source.width < MAIN_IMAGE_SIZE[0] or source.height < MAIN_IMAGE_SIZE[1]:
            raise ValueError(
                "Generated image is smaller than the required 400x300 output"
            )
        rgb_source = source.convert("RGB")
        cropped = _crop_to_ratio(rgb_source, 4, 3)
        main = _encode_webp(cropped, size=MAIN_IMAGE_SIZE, quality=92)
        thumbnail = _encode_webp(
            cropped,
            size=THUMBNAIL_IMAGE_SIZE,
            quality=88,
        )

    validate_webp(main, MAIN_IMAGE_SIZE)
    validate_webp(thumbnail, THUMBNAIL_IMAGE_SIZE)
    digest = hashlib.sha256(main).hexdigest()
    return main, thumbnail, digest


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def image_filenames(cocktail_id: int, digest: str) -> tuple[str, str]:
    short_digest = digest[:12]
    stem = f"{cocktail_id}-{short_digest}"
    return f"{stem}.webp", f"{stem}-thumb.webp"


def build_main_url(base_url: str, filename: str) -> str:
    return f"{base_url.rstrip('/')}/{filename}"


def build_thumbnail_url(image_url: str, canonical_base_url: str) -> str:
    """Return the sibling thumbnail URL only for canonical Cocktail Mate images."""

    image = urlparse(image_url)
    base = urlparse(canonical_base_url.rstrip("/"))
    expected_prefix = f"{base.path.rstrip('/')}/"
    if (
        image.scheme != base.scheme
        or image.netloc != base.netloc
        or not image.path.startswith(expected_prefix)
        or not image.path.endswith(".webp")
        or image.query
        or image.fragment
    ):
        return image_url
    return image_url[:-5] + "-thumb.webp"


class StateStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path_for(self, cocktail_id: int) -> Path:
        return self.directory / f"{cocktail_id}.json"

    def load(self, cocktail_id: int) -> GenerationState | None:
        path = self.path_for(cocktail_id)
        if not path.exists():
            return None
        return GenerationState.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, state: GenerationState) -> None:
        state.updated_at = datetime.now(timezone.utc)
        data = state.model_dump_json(indent=2).encode("utf-8")
        atomic_write(self.path_for(state.cocktail_id), data)

    def all_states(self) -> list[GenerationState]:
        if not self.directory.exists():
            return []
        states: list[GenerationState] = []
        for path in sorted(self.directory.glob("*.json")):
            states.append(
                GenerationState.model_validate_json(path.read_text(encoding="utf-8"))
            )
        return states
