from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace

from app.taste_generation.core import CocktailTasteSource, FlavorProfile
from app.taste_generation.pipeline import (
    CocktailTastePipeline,
    ExportOptions,
)


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def verify_model(self) -> None:
        return None

    def generate_profile(self, prompt: str) -> FlavorProfile:
        self.calls.append(prompt)
        return FlavorProfile.model_validate(
            {
                "sweetness": "MEDIUM",
                "acidity": "HIGH",
                "bitterness": "LOW",
                "salinity": "NONE",
                "umami": "NONE",
                "aroma_intensity": "HIGH",
                "fruit_aromas": ["레몬"],
                "other_aromas": ["주니퍼"],
                "palate_fruit_notes": ["레몬"],
                "palate_other_notes": ["주니퍼"],
                "body": "MEDIUM",
                "carbonation": "NONE",
                "creaminess": "LOW",
                "mouthfeel": ["산뜻함", "매끄러움"],
                "serving_temperature": "VERY_COLD",
                "alcohol_presence": "MEDIUM",
                "alcohol_character": "CLEAN",
                "finish_length": "MEDIUM",
                "finish_character": ["새콤함", "드라이함"],
                "embedding_text": (
                    "레몬의 선명한 시트러스 산미와 주니퍼 향이 생생하게 "
                    "어우러지고, 은은한 단맛이 풍미의 균형을 잡는다. "
                    "차갑고 산뜻한 질감과 매끄러운 목넘김 뒤로 깨끗한 "
                    "스피릿 온기와 드라이한 여운이 또렷하게 이어진다."
                ),
            }
        )


class RecordingPipeline(CocktailTastePipeline):
    def __init__(
        self,
        gateway: FakeGateway,
        source: CocktailTasteSource,
    ) -> None:
        super().__init__(None, gateway)  # type: ignore[arg-type]
        self.source = source

    def _load_sources(
        self,
        cocktail_ids: Iterable[int],
    ) -> Iterator[CocktailTasteSource]:
        ids = tuple(cocktail_ids)
        if not ids or self.source.id in ids:
            yield self.source


def _source(
    recipe: tuple[str, ...] = ("얼음과 함께 셰이크한다.",),
) -> CocktailTasteSource:
    return CocktailTasteSource(
        id=7,
        name="테스트 칵테일",
        name_en="Test Cocktail",
        recipe=recipe,
        abv=18.0,
        base_tag="gin",
        ingredients=(),
    )


def test_pipeline_checkpoints_csv_and_skips_matching_row(tmp_path: Path) -> None:
    gateway = FakeGateway()
    output = tmp_path / "taste.csv"

    first = RecordingPipeline(gateway, _source()).export(output, ExportOptions())
    second = RecordingPipeline(gateway, _source()).export(output, ExportOptions())

    assert first.generated == 1
    assert second.skipped == 1
    assert len(gateway.calls) == 1
    with output.open(encoding="utf-8-sig", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["cocktail_id"] == "7"
    assert row["recipe"] == '["얼음과 함께 셰이크한다."]'
    assert "레몬의 선명한 시트러스 산미" in row["embedding_text"]
    assert "없음" not in row["embedding_text"]
    assert "|" not in row["embedding_text"]


def test_changed_recipe_is_regenerated_and_dry_run_does_not_write(
    tmp_path: Path,
) -> None:
    output = tmp_path / "taste.csv"
    first_gateway = FakeGateway()
    RecordingPipeline(first_gateway, _source()).export(output, ExportOptions())
    before = output.read_bytes()

    dry_gateway = FakeGateway()
    summary = RecordingPipeline(
        dry_gateway,
        _source(("새 레시피로 변경한다.",)),
    ).export(output, ExportOptions(dry_run=True))

    assert summary.dry_run == 1
    assert dry_gateway.calls == []
    assert output.read_bytes() == before


def test_db_rows_are_grouped_into_one_source_with_ordered_ingredients() -> None:
    rows = [
        SimpleNamespace(
            cocktail_id=1,
            cocktail_name="진 사워",
            cocktail_name_en="Gin Sour",
            recipe=["셰이크한다."],
            cocktail_abv=20.0,
            base_tag="gin",
            link_id=10,
            amount=45.0,
            unit="ml",
            ingredient_name="진",
            ingredient_name_en="Gin",
            category="spirit",
            ingredient_description="주니퍼",
            ingredient_abv=40.0,
        ),
        SimpleNamespace(
            cocktail_id=1,
            cocktail_name="진 사워",
            cocktail_name_en="Gin Sour",
            recipe=["셰이크한다."],
            cocktail_abv=20.0,
            base_tag="gin",
            link_id=11,
            amount=20.0,
            unit="ml",
            ingredient_name="레몬 주스",
            ingredient_name_en="Lemon juice",
            category="juice",
            ingredient_description="산미",
            ingredient_abv=None,
        ),
        SimpleNamespace(
            cocktail_id=2,
            cocktail_name="재료 미등록 칵테일",
            cocktail_name_en=None,
            recipe=["차갑게 젓는다."],
            cocktail_abv=None,
            base_tag=None,
            link_id=None,
            amount=None,
            unit=None,
            ingredient_name=None,
            ingredient_name_en=None,
            category=None,
            ingredient_description=None,
            ingredient_abv=None,
        ),
    ]

    class FakeSession:
        def __enter__(self) -> FakeSession:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, _statement: object) -> list[SimpleNamespace]:
            return rows

    class FakeSessionFactory:
        def __call__(self) -> FakeSession:
            return FakeSession()

    pipeline = CocktailTastePipeline(
        FakeSessionFactory(),  # type: ignore[arg-type]
        FakeGateway(),
    )

    sources = list(pipeline._load_sources(()))

    assert [source.id for source in sources] == [1, 2]
    assert [ingredient.name for ingredient in sources[0].ingredients] == [
        "진",
        "레몬 주스",
    ]
    assert sources[0].ingredients[0].amount == 45.0
    assert sources[1].ingredients == ()
