from __future__ import annotations

from app.sensory_embedding.vertex_batch import FrozenCocktail
from scripts.prepare_sensory_real_pilot import (
    recipe_features,
    select_representative_rows,
)


def _row(cocktail_id: int) -> FrozenCocktail:
    return FrozenCocktail(
        cocktail_id=cocktail_id,
        source_column="recipe_facts",
        recipe_facts={
            "method": ("shake", "stir", "build")[cocktail_id % 3],
            "mixing_ice": cocktail_id % 2 == 0,
            "serving_ice": cocktail_id % 4 == 0,
            "carbonation": cocktail_id % 5 == 0,
            "estimated_pre_dilution_abv": float(cocktail_id * 3),
            "ingredients": [
                {
                    "canonical_name": f"ingredient-{cocktail_id % 7}",
                    "category": f"category-{cocktail_id % 4}",
                    "normalized_amount_ratio": 1.0,
                    "presence_only": False,
                }
            ],
        },
    )


def test_representative_selection_is_order_independent_and_deterministic() -> None:
    rows = tuple(_row(cocktail_id) for cocktail_id in range(1, 31))

    first = select_representative_rows(rows)
    second = select_representative_rows(tuple(reversed(rows)))

    assert [row.cocktail_id for row in first] == [row.cocktail_id for row in second]
    assert len(first) == len({row.cocktail_id for row in first}) == 10
    assert [row.cocktail_id for row in first] == sorted(
        row.cocktail_id for row in first
    )


def test_recipe_features_cover_recipe_structure_without_identity() -> None:
    row = _row(7)

    features = recipe_features(row)

    assert any(value.startswith("ingredient_canonical_name:") for value in features)
    assert any(value.startswith("ingredient_category:") for value in features)
    assert any(value.startswith("method:") for value in features)
    assert any(value.startswith("abv_bin_5:") for value in features)
    assert all(str(row.cocktail_id) not in value for value in features)
