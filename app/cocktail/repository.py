"""칵테일 데이터 접근 계층.

현재는 mock 데이터를 반환한다. ERD 확정 후 SQLAlchemy 세션 기반 쿼리로 교체한다
(생성자에서 `Session`을 주입받는 형태로 확장 예정).
"""
from app.cocktail.mock import MOCK_COCKTAILS

from sqlalchemy.orm import Session

from cocktail_mate_db.models.cocktail import Cocktail


class CocktailRepository:
    def list_all(self) -> list[dict]:
        return MOCK_COCKTAILS

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
        cocktail = (
            db.query(Cocktail)
            .filter(Cocktail.id == cocktail_id)
            .first()
        )

        if cocktail is None:
            return None

        return {
            "name": cocktail.name,
            "recipe": cocktail.recipe,
            "description": cocktail.description,
        }
