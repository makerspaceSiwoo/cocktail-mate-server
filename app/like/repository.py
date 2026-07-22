"""좋아요 데이터 접근 계층 — likes 테이블 CRUD 및 조회.

세션은 호출부(service)가 주입한다.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cocktail_mate_db.models import Cocktail, Like


class LikeRepository:
    def get_like(self, db: Session, user_id: int, cocktail_id: int) -> Like | None:
        return db.execute(
            select(Like).where(Like.user_id == user_id, Like.cocktail_id == cocktail_id)
        ).scalar_one_or_none()

    def add_like(self, db: Session, user_id: int, cocktail_id: int) -> Like:
        like = Like(user_id=user_id, cocktail_id=cocktail_id)
        db.add(like)
        db.flush()
        return like

    def delete_like(self, db: Session, like: Like) -> None:
        db.delete(like)
        db.flush()

    def count_likes(self, db: Session, cocktail_id: int) -> int:
        return db.execute(
            select(func.count())
            .select_from(Like)
            .where(Like.cocktail_id == cocktail_id)
        ).scalar_one()

    def cocktail_exists(self, db: Session, cocktail_id: int) -> bool:
        return (
            db.execute(select(Cocktail.id).where(Cocktail.id == cocktail_id)).first()
            is not None
        )

    def list_liked_cocktails(
        self,
        db: Session,
        user_id: int,
        page: int,
        rpp: int,
    ) -> dict:
        """유저가 좋아요한 칵테일 목록을 최근 좋아요 순으로 조회한다."""
        cocktails = list(
            db.execute(
                select(Cocktail)
                .join(Like, Like.cocktail_id == Cocktail.id)
                .where(Like.user_id == user_id)
                .order_by(
                    Like.created_at.desc(),
                )
                .offset((page - 1) * rpp)
                .limit(rpp + 1)
            ).scalars()
        )

        has_next_page = len(cocktails) > rpp
        cocktails = cocktails[:rpp]

        return {
            "cocktails": cocktails,
            "meta": {
                "page": page,
                "rpp": rpp,
                "hasNextPage": has_next_page,
            },
        }

    def liked_cocktail_ids(self, db: Session, user_id: int) -> set[int]:
        """유저가 좋아요한 cocktail_id 집합 (is_liked 매핑용)."""
        rows = db.execute(select(Like.cocktail_id).where(Like.user_id == user_id)).all()
        return {row[0] for row in rows}

    def like_counts_for(self, db: Session, cocktail_ids: list[int]) -> dict[int, int]:
        """여러 칵테일의 좋아요 수를 한 번에 조회."""
        if not cocktail_ids:
            return {}
        rows = db.execute(
            select(Like.cocktail_id, func.count())
            .where(Like.cocktail_id.in_(cocktail_ids))
            .group_by(Like.cocktail_id)
        ).all()
        return {cocktail_id: count for cocktail_id, count in rows}
