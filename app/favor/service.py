"""취향 추천 비즈니스 로직 — 좋아요 임베딩 평균 → 클러스터 범위 내 최근접."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.favor.repository import FavorRepository

# 클러스터 하한: 코사인 유사도 0.65 이상만 (거리 0.35 이하).
MIN_SIMILARITY = 0.65
MAX_COSINE_DISTANCE = 1.0 - MIN_SIMILARITY
FAVOR_LIMIT = 5


class FavorService:
    def __init__(self, repository: FavorRepository | None = None) -> None:
        self.repository = repository or FavorRepository()

    def recommend(self, db: Session, user_id: int) -> list[dict]:
        embeddings = self.repository.liked_embeddings(db, user_id)
        if not embeddings:
            return []
        n = len(embeddings)
        centroid = [sum(col) / n for col in zip(*embeddings)]
        exclude_ids = self.repository.liked_ids(db, user_id)
        return self.repository.nearest_within(
            db,
            centroid,
            exclude_ids,
            MAX_COSINE_DISTANCE,
            FAVOR_LIMIT,
        )
