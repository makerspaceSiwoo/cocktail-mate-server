"""추천 데이터 접근 — Graph48 코사인 ANN + Preference48 내적(MIPS) 조회."""

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
        limit: int,
    ) -> list[dict]:
        # HNSW 반복 스캔(pgvector 0.8+): 자기 자신 제외로 후보가 걸러져도 LIMIT 만큼 계속
        # 스캔해 결과가 모자라게(under-fill) 반환되는 것을 막는다. 트랜잭션 로컬 설정.
        db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        # Graph48 코사인 거리(`<=>`, idx_cocktails_embedding_hnsw / vector_cosine_ops).
        # 절대 거리 임계값은 없다 — 이유는 app/recommend/service.py 모듈 docstring 참고.
        dist = Cocktail.embedding.cosine_distance(target_embedding)
        rows = db.execute(
            select(
                Cocktail.id,
                Cocktail.name,
                Cocktail.name_en,
                Cocktail.description,
                Cocktail.image_url,
                Cocktail.abv,
                dist.label("dist"),
            )
            .where(
                Cocktail.embedding.isnot(None),
                Cocktail.id != exclude_id,
            )
            .order_by(dist)
            .limit(limit)
        ).all()
        return [
            {
                "id": row.id,
                "name": row.name,
                "nameEn": row.name_en,
                "description": row.description,
                "imageUrl": row.image_url,
                "abv": row.abv,
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
                Cocktail.name_en,
                Cocktail.description,
                Cocktail.image_url,
                Cocktail.abv,
            )
            .order_by(func.random())
            .limit(limit)
        ).all()
        return [
            {
                "id": row.id,
                "name": row.name,
                "nameEn": row.name_en,
                "description": row.description,
                "imageUrl": row.image_url,
                "abv": row.abv,
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

    def nearest_by_preference(
        self,
        db: Session,
        query_vector,
        limit: int,
    ) -> list[dict]:
        """Rank cocktails by maximum inner product against a Preference48 query.

        `preference_embedding`은 정규화하지 않은 [0,1] 벡터라 코사인이 아니라 **내적**으로
        비교한다(`idx_cocktails_preference_embedding_hnsw` / `vector_ip_ops`).
        pgvector의 `<#>`(`max_inner_product`)는 **음의 내적**을 돌려주므로
        오름차순 정렬이 곧 내적 내림차순이고, `similarity = -distance`다.
        """
        db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        distance = Cocktail.preference_embedding.max_inner_product(query_vector)
        rows = db.execute(
            select(
                Cocktail.id,
                Cocktail.name,
                Cocktail.name_en,
                Cocktail.description,
                Cocktail.image_url,
                Cocktail.abv,
                distance.label("dist"),
            )
            .where(Cocktail.preference_embedding.isnot(None))
            .order_by(distance)
            .limit(limit)
        ).all()
        return [
            {
                "id": row.id,
                "name": row.name,
                "nameEn": row.name_en,
                "description": row.description,
                "imageUrl": row.image_url,
                "abv": row.abv,
                "similarity": round(-float(row.dist), 4),
            }
            for row in rows
        ]
