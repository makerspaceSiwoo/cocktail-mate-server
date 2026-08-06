"""취향 추천 데이터 접근 — 최근 좋아요 임베딩 조회 + 개별 ANN."""

from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from cocktail_mate_db.models import Cocktail, Like


class FavorRepository:
    def liked_embeddings(
        self,
        db: Session,
        user_id: int,
        *,
        limit: int,
    ) -> list[list[float]]:
        recent_likes = (
            select(Like.cocktail_id.label("cocktail_id"))
            .where(Like.user_id == user_id)
            .order_by(Like.created_at.desc(), Like.id.desc())
            .limit(limit)
            .subquery()
        )
        rows = db.execute(
            select(Cocktail.embedding)
            .join(recent_likes, recent_likes.c.cocktail_id == Cocktail.id)
            .where(Cocktail.embedding.isnot(None))
        ).all()
        return [[float(x) for x in row[0]] for row in rows]

    def liked_ids(self, db: Session, user_id: int) -> set[int]:
        rows = db.execute(select(Like.cocktail_id).where(Like.user_id == user_id)).all()
        return {row[0] for row in rows}

    def nearest_within(
        self,
        db: Session,
        target_embedding: list[float],
        exclude_ids: set[int],
        limit: int,
    ) -> list[dict]:
        # HNSW 반복 스캔(pgvector 0.8+): 좋아요 제외 필터로 후보가 걸러져도 LIMIT 만큼
        # 계속 스캔해 결과가 모자라게(under-fill) 반환되는 것을 막는다. 트랜잭션 로컬 설정.
        db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        # Graph48 코사인 거리(`<=>`). 절대 거리 임계값은 없다 — 모듈 docstring 참고.
        dist = Cocktail.embedding.cosine_distance(target_embedding)
        conditions = [Cocktail.embedding.isnot(None)]
        if exclude_ids:
            conditions.append(Cocktail.id.notin_(exclude_ids))
        rows = db.execute(
            select(Cocktail.id, Cocktail.name, Cocktail.image_url, dist.label("dist"))
            .where(*conditions)
            .order_by(dist)
            .limit(limit)
        ).all()
        return [
            {
                "id": row.id,
                "name": row.name,
                "similarity": round(1.0 - float(row.dist), 4),
                "imageUrl": row.image_url,
            }
            for row in rows
        ]
