"""추천 데이터 접근 — 임베딩 코사인 ANN 조회."""

from __future__ import annotations

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from cocktail_mate_db.models import (
    Cocktail,
    TasteDescriptor,
)


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
        # HNSW 반복 스캔(pgvector 0.8+): 거리 필터로 후보가 걸러져도 LIMIT 만큼 계속 스캔해
        # 결과가 모자라게(under-fill) 반환되는 것을 막는다. 트랜잭션 로컬 설정.
        db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        # 코사인 거리식을 SELECT/WHERE/ORDER BY 에서 재사용 (거리 <= 임계치 → 클러스터 이탈 방지).
        dist = Cocktail.embedding.cosine_distance(target_embedding)
        rows = db.execute(
            select(
                Cocktail.id,
                Cocktail.name,
                Cocktail.description,
                Cocktail.image_url,
                dist.label("dist"),
            )
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
                "description": row.description,
                "imageUrl": row.image_url,
                "similarity": round(1.0 - float(row.dist), 4),
            }
            for row in rows
        ]

    def list_taste_descriptors(self, db: Session) -> list[TasteDescriptor]:
        return list(
            db.scalars(
                select(TasteDescriptor)
                .where(TasteDescriptor.is_active.is_(True))
                .order_by(TasteDescriptor.display_order, TasteDescriptor.id)
            ).all()
        )

    def random_cocktails(self, db: Session, limit: int) -> list[dict]:
        rows = db.execute(
            select(
                Cocktail.id,
                Cocktail.name,
                Cocktail.description,
                Cocktail.image_url,
            )
            .order_by(func.random())
            .limit(limit)
        ).all()
        return [
            {
                "id": row.id,
                "name": row.name,
                "description": row.description,
                "imageUrl": row.image_url,
                "similarity": 0.0,
            }
            for row in rows
        ]

    def active_taste_descriptors_by_ids(
        self,
        db: Session,
        descriptor_ids: list[int],
    ) -> list[TasteDescriptor]:
        return list(
            db.scalars(
                select(TasteDescriptor)
                .where(
                    TasteDescriptor.id.in_(descriptor_ids),
                    TasteDescriptor.is_active.is_(True),
                )
                .order_by(TasteDescriptor.display_order, TasteDescriptor.id)
            ).all()
        )

    def nearest_for_virtual_taste(
        self,
        db: Session,
        target_embedding,
        limit: int,
    ) -> list[dict]:
        db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        distance = Cocktail.embedding.cosine_distance(target_embedding)
        rows = db.execute(
            select(
                Cocktail.id,
                Cocktail.name,
                Cocktail.description,
                Cocktail.image_url,
                distance.label("dist"),
            )
            .where(Cocktail.embedding.isnot(None))
            .order_by(distance)
            .limit(limit)
        ).all()
        return [
            {
                "id": row.id,
                "name": row.name,
                "description": row.description,
                "imageUrl": row.image_url,
                "similarity": round(1.0 - float(row.dist), 4),
            }
            for row in rows
        ]
