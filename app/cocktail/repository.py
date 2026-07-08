"""칵테일 데이터 접근 계층.

현재는 mock 데이터를 반환한다. ERD 확정 후 SQLAlchemy 세션 기반 쿼리로 교체한다
(생성자에서 `Session`을 주입받는 형태로 확장 예정).
"""

from sqlalchemy.orm import Session

from app.cocktail.mock import MOCK_COCKTAILS
from cocktail_mate_db.models import Cocktail, CocktailIngredient, Ingredient


class CocktailRepository:
    def list_all(
        self,
        db: Session,
        page: int,
        rpp: int,
        base: str | None,
        liked_ids: set[int] | None = None,
    ) -> dict:
        query = db.query(Cocktail)

        if base:
            query = query.filter(Cocktail.base_tag == base)

        cocktails = (
            query.order_by(Cocktail.id).offset((page - 1) * rpp).limit(rpp + 1).all()
        )

        has_next_page = len(cocktails) > rpp
        cocktails = cocktails[:rpp]
        liked_ids = liked_ids or set()

        return {
            "items": [
                {
                    "id": cocktail.id,
                    "name": cocktail.name,
                    "nameEn": cocktail.name_en,
                    "imageUrl": cocktail.image_url,
                    "baseTag": cocktail.base_tag,
                    "description": cocktail.description,
                    "abv": cocktail.abv,
                    "glass": cocktail.glass,
                    "isLiked": cocktail.id in liked_ids,
                }
                for cocktail in cocktails
            ],
            "meta": {
                "page": page,
                "rpp": rpp,
                "hasNextPage": has_next_page,
            },
        }

    def get_base_tags(self, db: Session) -> dict:
        rows = (
            db.query(Cocktail.base_tag)
            .filter(Cocktail.base_tag.isnot(None))
            .distinct()
            .order_by(Cocktail.base_tag)
            .all()
        )

        return {"items": [row[0] for row in rows]}

    def search(self, keyword: str) -> list[dict]:
        kw = keyword.lower()
        return [
            c
            for c in MOCK_COCKTAILS
            if kw in c["name"].lower()
            or kw in c["baseTag"].lower()
            or kw in c["description"].lower()
        ]

    def find_detail_by_id(self, db: Session, cocktail_id: int) -> dict | None:
        cocktail = db.query(Cocktail).filter(Cocktail.id == cocktail_id).first()

        if cocktail is None:
            return None

        rows = (
            db.query(CocktailIngredient, Ingredient)
            .join(Ingredient, CocktailIngredient.ingredient_id == Ingredient.id)
            .filter(CocktailIngredient.cocktail_id == cocktail_id)
            .order_by(CocktailIngredient.id)
            .all()
        )

        return {
            "id": cocktail.id,
            "name": cocktail.name,
            "nameEn": cocktail.name_en,
            "imageUrl": cocktail.image_url,
            "glass": cocktail.glass,
            "abv": cocktail.abv,
            "recipe": cocktail.recipe,
            "description": cocktail.description,
            "baseTag": cocktail.base_tag,
            "ingredients": [
                {
                    "id": ingredient.id,
                    "name": ingredient.name,
                    "nameEn": ingredient.name_en,
                    "category": ingredient.category,
                    "amount": cocktail_ingredient.amount,
                    "unit": cocktail_ingredient.unit,
                    "description": ingredient.description,
                    "abv": ingredient.abv,
                    "imageUrl": ingredient.image_url,
                    "potency": ingredient.potency,
                }
                for cocktail_ingredient, ingredient in rows
            ],
        }
