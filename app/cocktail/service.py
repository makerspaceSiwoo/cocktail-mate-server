"""칵테일 비즈니스 로직 (Controller가 호출).

응답 본문은 기존 mock 구현과 동일하게 유지한다. DB 연동은 ERD 확정 후.
"""

from sqlalchemy.orm import Session

from app.cocktail.repository import CocktailRepository


class CocktailService:
    def __init__(self, repository: CocktailRepository | None = None) -> None:
        self.repository = repository or CocktailRepository()

    def list_cocktails(
        self,
        db: Session,
        page: int,
        rpp: int,
        base: str | None,
    ) -> dict:
        return self.repository.list_all(db, page, rpp, base)
    
    def get_base_tags(self, db: Session) -> dict:
        return self.repository.get_base_tags(db)

    def get_brief(self, cocktail_id: int) -> dict:
        return {
            "cocktailId": cocktail_id,
            "imageUrl": "https://fastly.picsum.photos/id/73/200/200.jpg?hmac=IYjgRq-Ok9gn3_MVxJ4TlfhLPONQ97qWvp2Ir1Y1z6c",
            "cocktailName": "마가리타",
        }

    def search(self, keyword: str, page: int, rpp: int) -> dict:
        filtered = self.repository.search(keyword)
        total = len(filtered)
        start = (page - 1) * rpp
        end = start + rpp
        return {"total": total, "cocktails": filtered[start:end]}

    def explore(self, cocktail_id: int) -> dict:
        return {
            "cocktailId": cocktail_id,
            "cocktailName": "cocktailtail",
            "imageUrl": "https://fastly.picsum.photos/id/73/200/200.jpg?hmac=IYjgRq-Ok9gn3_MVxJ4TlfhLPONQ97qWvp2Ir1Y1z6c",
            "glass": "glass1",
            "ABV": 21.3,
            "numLike": 10,
            "recipe": "mix",
            "description": "salty, sugary",
            "baseTag": "rum",
            "isLiked": False,
        }

    def drink_of_the_day(self) -> dict:
        return {
            "cocktailId": 123,
            "cocktailName": "magarita",
            "imageUrl": "https://fastly.picsum.photos/id/73/200/200.jpg?hmac=IYjgRq-Ok9gn3_MVxJ4TlfhLPONQ97qWvp2Ir1Y1z6c",
            "description": "salty, sugary",
            "ABV": 21.3,
        }
