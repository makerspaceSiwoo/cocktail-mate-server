"""취향 추천 데이터 접근 — 좋아요 임베딩 조회 + 센트로이드 ANN."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from cocktail_mate_db.models import Cocktail, Like


class FavorRepository:
    def liked_embeddings(self, db: Session, user_id: int) -> list[list[float]]:
        rows = db.execute(
            select(Cocktail.embedding)
            .join(Like, Like.cocktail_id == Cocktail.id)
            .where(Like.user_id == user_id, Cocktail.embedding.isnot(None))
        ).all()
        return [[float(x) for x in row[0]] for row in rows]

    def liked_ids(self, db: Session, user_id: int) -> set[int]:
        rows = db.execute(select(Like.cocktail_id).where(Like.user_id == user_id)).all()
        return {row[0] for row in rows}

    def nearest_within(
        self,
        db: Session,
        centroid: list[float],
        exclude_ids: set[int],
        max_distance: float,
        limit: int,
    ) -> list[dict]:
        dist = Cocktail.embedding.cosine_distance(centroid)
        conditions = [Cocktail.embedding.isnot(None), dist <= max_distance]
        if exclude_ids:
            conditions.append(Cocktail.id.notin_(exclude_ids))
        rows = db.execute(
            select(Cocktail.id, Cocktail.name, dist.label("dist"))
            .where(*conditions)
            .order_by(dist)
            .limit(limit)
        ).all()
        return [
            {
                "id": row.id,
                "name": row.name,
                "similarity": round(1.0 - float(row.dist), 4),
            }
            for row in rows
        ]
