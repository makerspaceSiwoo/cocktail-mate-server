"""취향 추천 비즈니스 로직 — 최근 좋아요별 ANN 후보 병합."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.favor.repository import FavorRepository

# 코사인 유사도 0.6 이상인 후보만 추천한다 (거리 0.4 이하).
MIN_SIMILARITY = 0.6
MAX_COSINE_DISTANCE = 1.0 - MIN_SIMILARITY
RECENT_LIKES_LIMIT = 5
FAVOR_LIMIT = 5


class FavorService:
    def __init__(self, repository: FavorRepository | None = None) -> None:
        self.repository = repository or FavorRepository()

    def recommend(self, db: Session, user_id: int) -> list[dict]:
        embeddings = self.repository.liked_embeddings(
            db,
            user_id,
            limit=RECENT_LIKES_LIMIT,
        )
        if not embeddings:
            return []
        exclude_ids = self.repository.liked_ids(db, user_id)
        candidates_by_id: dict[int, dict] = {}
        for embedding in embeddings:
            candidates = self.repository.nearest_within(
                db,
                embedding,
                exclude_ids,
                MAX_COSINE_DISTANCE,
                FAVOR_LIMIT,
            )
            for candidate in candidates:
                existing = candidates_by_id.get(candidate["id"])
                if existing is None or candidate["similarity"] > existing["similarity"]:
                    candidates_by_id[candidate["id"]] = candidate

        return sorted(
            candidates_by_id.values(),
            key=lambda candidate: (-candidate["similarity"], candidate["id"]),
        )[:FAVOR_LIMIT]
