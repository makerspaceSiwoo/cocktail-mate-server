"""칵테일 비즈니스 로직 (Controller가 호출)."""

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.cocktail.repository import CocktailRepository
from app.like.repository import LikeRepository

from fastapi import HTTPException

# MVP 기준 한국 시간(UTC+9)의 날짜를 오늘의 추천 시드로 사용한다.
KST = timezone(timedelta(hours=9))


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
        liked_ids = (
            self.like_repository.liked_cocktail_ids(db, user_id)
            if user_id is not None
            else None
        )

        response = self.repository.list_all(
            db,
            page,
            rpp,
            base,
            liked_ids,
        )

        cocktail_ids = [item["id"] for item in response["items"]]
        like_counts = self.like_repository.like_counts_for(db, cocktail_ids)

        for item in response["items"]:
            item["likeCount"] = like_counts.get(item["id"], 0)

        return response

    def get_base_tags(self, db: Session) -> dict:
        return self.repository.get_base_tags(db)

    def daily_recommend(self, db: Session, count: int = 5) -> dict:
        # KST 날짜(YYYY-MM-DD)를 시드로 → 같은 날은 모든 유저에게 동일한 추천.
        seed = datetime.now(KST).strftime("%Y-%m-%d")
        return {"items": self.repository.daily_cocktails(db, seed, count)}

    def get_detail(self, db: Session, cocktail_id: int) -> dict:
        cocktail = self.repository.find_detail_by_id(db, cocktail_id)

        if cocktail is None:
            raise HTTPException(status_code=404, detail="Cocktail not found")

        return cocktail
