"""추천 비즈니스 로직 — 클러스터 범위(코사인 유사도 하한) 내 최근접 칵테일."""

from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.recommend.repository import RecommendRepository

# 클러스터 하한: 코사인 유사도 0.65 이상만 (거리 0.35 이하). 낮은 유사도는 억지로 추천하지 않는다.
MIN_SIMILARITY = 0.65
MAX_COSINE_DISTANCE = 1.0 - MIN_SIMILARITY
RECOMMEND_LIMIT = 5


class RecommendService:
    def __init__(self, repository: RecommendRepository | None = None) -> None:
        self.repository = repository or RecommendRepository()

    def recommend(self, db: Session, cocktail_id: int) -> list[dict]:
        cocktail = self.repository.get_by_id(db, cocktail_id)
        if cocktail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="칵테일을 찾을 수 없습니다.",
            )
        # 임베딩이 아직 없는 칵테일 → 추천 불가, 빈 목록.
        if cocktail.embedding is None:
            return []
        return self.repository.nearest_within(
            db,
            cocktail.embedding,
            cocktail_id,
            MAX_COSINE_DISTANCE,
            RECOMMEND_LIMIT,
        )
