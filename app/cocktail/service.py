"""칵테일 비즈니스 로직 (Controller가 호출).

응답 본문은 기존 mock 구현과 동일하게 유지한다. DB 연동은 ERD 확정 후.
"""

from sqlalchemy.orm import Session

from app.cocktail.repository import CocktailRepository
from app.like.repository import LikeRepository


class CocktailService:
    def __init__(
        self,
        repository: CocktailRepository | None = None,
        like_repository: LikeRepository | None = None,
    ) -> None:
        self.repository = repository or CocktailRepository()
        self.like_repository = like_repository or LikeRepository()

    def list_cocktails(
        self,
        db: Session,
        page: int,
        rpp: int,
        base: str | None,
        user_id: int | None = None,
    ) -> dict:
        # 로그인 유저면 좋아요 집합을 조회해 각 항목의 is_liked 를 채운다.
        liked_ids = (
            self.like_repository.liked_cocktail_ids(db, user_id)
            if user_id is not None
            else None
        )
        return self.repository.list_all(db, page, rpp, base, liked_ids)
    
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
