"""탐색 데이터 접근 — 3D 임베딩 좌표가 있는 칵테일 전체 조회."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from cocktail_mate_db.models import Cocktail


class ExploreRepository:
    def list_with_embedding_3d(self, db: Session) -> list[dict]:
        rows = db.execute(
            select(Cocktail.id, Cocktail.name, Cocktail.abv, Cocktail.embedding_3d)
            .where(Cocktail.embedding_3d.isnot(None))
            .order_by(Cocktail.id)
        ).all()
        return [
            {
                "id": row.id,
                "name": row.name,
                "abv": row.abv,
                "embedding3d": [float(x) for x in row.embedding_3d],
            }
            for row in rows
        ]
