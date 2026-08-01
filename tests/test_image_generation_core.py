from __future__ import annotations

import io

from PIL import Image

from app.image_generation.core import (
    MAIN_IMAGE_SIZE,
    THUMBNAIL_IMAGE_SIZE,
    CocktailSource,
    ImageGenerationSettings,
    ModelRateLimiter,
    build_contents_expansion_prompt,
    build_final_prompt,
    build_thumbnail_url,
    create_image_variants,
    image_filenames,
    load_prompt_catalog,
)


def test_expansion_prompt_uses_visual_facts_without_taste_description() -> None:
    cocktail = CocktailSource(
        id=1,
        name="어 데이 앳 더 비치",
        name_en="A Day at the Beach",
        glass="하이볼 글라스",
        base_tag="tequila",
        recipe=(
            "하이볼 잔에 얼음을 채운다.",
            "그레나딘을 천천히 부어 가라앉힌다.",
        ),
    )

    prompt = build_contents_expansion_prompt(cocktail)

    assert "어 데이 앳 더 비치" in prompt
    assert "A Day at the Beach" in prompt
    assert "그레나딘을 천천히" in prompt
    assert "Do not describe or" in prompt
    assert "glass shape" in prompt
    assert "Stirring or shaking alone is not evidence of foam" in prompt


def test_create_image_variants_center_crops_and_encodes_webp() -> None:
    source = Image.new("RGB", (900, 900), color=(210, 80, 30))
    raw = io.BytesIO()
    source.save(raw, format="PNG")

    main, thumbnail, digest = create_image_variants(raw.getvalue())

    with Image.open(io.BytesIO(main)) as main_image:
        assert main_image.format == "WEBP"
        assert main_image.size == MAIN_IMAGE_SIZE
    with Image.open(io.BytesIO(thumbnail)) as thumbnail_image:
        assert thumbnail_image.format == "WEBP"
        assert thumbnail_image.size == THUMBNAIL_IMAGE_SIZE
    assert len(digest) == 64
    assert image_filenames(42, digest) == (
        f"42-{digest[:12]}.webp",
        f"42-{digest[:12]}-thumb.webp",
    )


def test_create_image_variants_uses_exact_flux2_klein_crop() -> None:
    source = Image.new("RGB", (1184, 880), color=(60, 120, 180))
    raw = io.BytesIO()
    source.save(raw, format="PNG")

    main, thumbnail, _digest = create_image_variants(
        raw.getvalue(),
        crop_size=(1172, 879),
    )

    with Image.open(io.BytesIO(main)) as main_image:
        assert main_image.size == (400, 300)
    with Image.open(io.BytesIO(thumbnail)) as thumbnail_image:
        assert thumbnail_image.size == (128, 96)


def test_thumbnail_url_is_derived_only_for_canonical_images() -> None:
    base_url = "https://api.cocktail-mate.com/media/cocktails"
    canonical = f"{base_url}/42-abcdef123456.webp"

    assert build_thumbnail_url(canonical, base_url) == (
        f"{base_url}/42-abcdef123456-thumb.webp"
    )
    assert (
        build_thumbnail_url("https://example.com/42.webp", base_url)
        == "https://example.com/42.webp"
    )
    assert build_thumbnail_url(f"{canonical}?v=1", base_url) == f"{canonical}?v=1"


def test_model_rate_limiter_tracks_each_model_independently() -> None:
    class Clock:
        value = 100.0

        def monotonic(self) -> float:
            return self.value

        def sleep(self, seconds: float) -> None:
            self.value += seconds

    clock = Clock()
    limiter = ModelRateLimiter(
        32,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert limiter.wait("text") == 0
    clock.value += 5
    assert limiter.wait("image") == 0
    clock.value += 5
    assert limiter.wait("text") == 22
    assert clock.value == 132
    assert limiter.wait("image") == 5
    assert clock.value == 137


def test_prompt_catalog_reuses_base_glass_and_composition_fragments() -> None:
    catalog = load_prompt_catalog(ImageGenerationSettings(gemini_api_key="test"))
    cocktail = CocktailSource(
        id=1,
        name="테킬라 선라이즈",
        name_en="Tequila Sunrise",
        glass="하이볼 글라스",
        base_tag="tequila",
        recipe=("재료를 따른다.",),
    )

    parts = catalog.parts_for(cocktail)
    final_prompt = build_final_prompt(
        "A vivid ruby-red to tropical-orange layered drink with crushed ice.",
        parts,
    )

    assert "#f0d366" in final_prompt
    assert "straight-sided highball glass" in final_prompt
    assert "4:3 1K" in final_prompt
    assert "Fixed 85mm lens at a slight 10-degree top-down angle" in final_prompt
    assert "soft-edged cast shadow" in final_prompt
    assert "upper-right (2 o'clock)" in final_prompt
    assert "No visible surface, floor" in final_prompt
    assert "horizon" in final_prompt.lower()
    assert "f/11 realism" in final_prompt
    assert len(final_prompt) <= 800
    assert final_prompt.index("Drink:") < final_prompt.index("Glass:")
    assert final_prompt.index("Glass:") < final_prompt.index("Background:")
    assert final_prompt.index("Background:") < final_prompt.index("Style:")


def test_prompt_catalog_covers_all_production_glasses_and_base_tags() -> None:
    catalog = load_prompt_catalog(ImageGenerationSettings(gemini_api_key="test"))

    assert {
        base_tag: background.color_hex
        for base_tag, background in catalog.backgrounds.items()
    } == {
        "rum": "#f2a878",
        "whiskey": "#d4a56e",
        "tequila": "#f0d366",
        "non_alcoholic": "#7ed08e",
        "gin": "#ef97c8",
        "vodka": "#8fb4ee",
        "liqueur": "#c39ae8",
        "brandy": "#e57373",
        "other": "#8f949b",
    }
    assert len(catalog.glasses) == 14
