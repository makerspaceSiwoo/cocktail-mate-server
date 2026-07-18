"""취향 추천 데이터 접근 — 좋아요 임베딩 조회 + 센트로이드 ANN."""

from __future__ import annotations

from sqlalchemy import select, text
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
        # HNSW 반복 스캔(pgvector 0.8+): 거리 필터 + 좋아요 제외로 후보가 걸러져도 LIMIT 만큼
        # 계속 스캔해 결과가 모자라게(under-fill) 반환되는 것을 막는다. 트랜잭션 로컬 설정.
        db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
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
