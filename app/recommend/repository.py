"""추천 데이터 접근 — 임베딩 코사인 ANN 조회."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from cocktail_mate_db.models import Cocktail


class RecommendRepository:
    def get_by_id(self, db: Session, cocktail_id: int) -> Cocktail | None:
        return db.get(Cocktail, cocktail_id)

    def nearest_within(
        self,
        db: Session,
        target_embedding,
        exclude_id: int,
        max_distance: float,
        limit: int,
    ) -> list[dict]:
        # 코사인 거리식을 SELECT/WHERE/ORDER BY 에서 재사용 (거리 <= 임계치 → 클러스터 이탈 방지).
        dist = Cocktail.embedding.cosine_distance(target_embedding)
        rows = db.execute(
            select(Cocktail.id, Cocktail.name, dist.label("dist"))
            .where(
                Cocktail.embedding.isnot(None),
                Cocktail.id != exclude_id,
                dist <= max_distance,
            )
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
