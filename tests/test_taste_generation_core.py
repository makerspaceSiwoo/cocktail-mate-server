from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.taste_generation.core import (
    CocktailTasteSource,
    FlavorProfile,
    IngredientSource,
    build_taste_prompt,
    csv_row_for,
    read_existing_rows,
    render_embedding_text,
    row_matches_source,
    write_csv_rows,
)


def _source(recipe: tuple[str, ...] | None = None) -> CocktailTasteSource:
    return CocktailTasteSource(
        id=42,
        name="화이트 레이디",
        name_en="White Lady",
        recipe=recipe
        or (
            "모든 재료를 얼음과 함께 셰이크한다.",
            "차갑게 식힌 잔에 더블 스트레인한다.",
        ),
        abv=24.0,
        base_tag="gin",
        ingredients=(
            IngredientSource(
                name="진",
                name_en="Gin",
                category="spirit",
                amount=40,
                unit="ml",
                description="주니퍼와 허브 풍미",
                abv=40,
            ),
            IngredientSource(
                name="레몬 주스",
                name_en="Lemon juice",
                category="juice",
                amount=20,
                unit="ml",
                description="신선한 산미",
                abv=None,
            ),
        ),
    )


def _profile() -> FlavorProfile:
    return FlavorProfile.model_validate(
        {
            "sweetness": "MEDIUM",
            "acidity": "HIGH",
            "bitterness": "LOW",
            "salinity": "VERY_LOW",
            "umami": "NONE",
            "aroma_intensity": "HIGH",
            "fruit_aromas": ["레몬"],
            "other_aromas": ["주니퍼", "허브"],
            "palate_fruit_notes": ["레몬"],
            "palate_other_notes": ["주니퍼"],
            "body": "MEDIUM",
            "carbonation": "NONE",
            "creaminess": "VERY_LOW",
            "mouthfeel": ["산뜻함", "매끄러움"],
            "serving_temperature": "VERY_COLD",
            "alcohol_presence": "MEDIUM",
            "alcohol_character": "CLEAN",
            "finish_length": "MEDIUM",
            "finish_character": ["새콤함", "드라이함"],
            "embedding_text": (
                "레몬의 선명한 시트러스 산미와 주니퍼·허브 향이 생생하게 "
                "어우러지고, 은은한 단맛이 전체 풍미의 균형을 잡는다. "
                "차갑고 산뜻한 질감과 매끄러운 목넘김 뒤로 깨끗한 스피릿 "
                "온기와 새콤하고 드라이한 여운이 또렷하게 이어진다."
            ),
        }
    )


def test_prompt_uses_recipe_ingredients_and_strict_sensory_scope() -> None:
    prompt = build_taste_prompt(_source())

    assert "화이트 레이디" in prompt
    assert "White Lady" in prompt
    assert "amount=40 ml" in prompt
    assert "주니퍼와 허브 풍미" in prompt
    assert "color, clarity, layers, appearance" in prompt
    assert "Shaking does not by itself imply creaminess" in prompt
    assert "in descending dominance" in prompt
    assert "embedding_text is the only text that will be vectorized" in prompt
    assert "sweetness NONE/VERY_LOW/LOW" in prompt
    assert "never use 단맛, 달콤함, 당도" in prompt
    assert "omit every carbonation-related word" in prompt


def test_renderer_returns_only_positive_natural_embedding_text() -> None:
    description = render_embedding_text(_profile())

    assert description.startswith("레몬의 선명한 시트러스 산미")
    assert "은은한 단맛" in description
    assert "매끄러운 목넘김" in description
    assert "깨끗한 스피릿 온기" in description
    assert "|" not in description
    assert "=" not in description
    assert "없음" not in description
    assert "탄산" not in description
    assert "도수" not in description
    assert all(
        forbidden not in description
        for forbidden in ("색상", "투명도", "잔=", "가니시", "역사", "유래")
    )


def test_profile_rejects_negative_axis_wording_in_embedding_text() -> None:
    payload = _profile().model_dump(mode="json")
    payload["embedding_text"] = (
        "레몬과 주니퍼 향이 산뜻하게 펼쳐지며 허브 풍미가 중심을 잡는다. "
        "단맛 없음과 탄산 없음이 특징이고 매끄러운 질감과 드라이한 여운이 "
        "깔끔하게 이어지는 칵테일이다."
    )

    with pytest.raises(ValueError, match="negative or level wording"):
        FlavorProfile.model_validate(payload)


def test_non_dominant_sweetness_and_acidity_can_be_omitted() -> None:
    payload = _profile().model_dump(mode="json")
    payload.update(
        {
            "sweetness": "NONE",
            "acidity": "NONE",
            "embedding_text": (
                "레몬과 주니퍼의 향이 생생하게 어우러지고 허브의 쌉쌀한 "
                "풍미가 입안에서 또렷하게 펼쳐진다. 차갑고 매끄러운 목넘김 "
                "뒤로 깨끗한 스피릿 온기와 드라이한 여운이 길게 이어진다."
            ),
        }
    )

    profile = FlavorProfile.model_validate(payload)

    assert "단맛" not in profile.embedding_text
    assert "산미" not in profile.embedding_text


def test_csv_is_utf8_bom_with_exact_columns_and_resumable_rows(
    tmp_path: Path,
) -> None:
    source = _source()
    output = tmp_path / "taste.csv"
    row = csv_row_for(source, render_embedding_text(_profile()))

    write_csv_rows(output, {source.id: row})

    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    with output.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == [
            "cocktail_id",
            "cocktail_name_ko",
            "cocktail_name_en",
            "recipe",
            "embedding_text",
        ]
        parsed = list(reader)
    assert parsed == [row]

    existing = read_existing_rows(output)
    assert row_matches_source(existing[42], source)
    assert not row_matches_source(
        existing[42],
        _source(("레시피가 변경되었다.",)),
    )
