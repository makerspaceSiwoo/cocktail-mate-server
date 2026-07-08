"""좋아요 비즈니스 로직 — 실제 DB(likes) 기반 등록/해제/목록."""

from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.like.repository import LikeRepository


class LikeService:
    def __init__(self, repository: LikeRepository | None = None) -> None:
        self.repository = repository or LikeRepository()

    def like(self, db: Session, user_id: int, cocktail_id: int) -> dict:
        if not self.repository.cocktail_exists(db, cocktail_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="칵테일을 찾을 수 없습니다.",
            )
        existing = self.repository.get_like(db, user_id, cocktail_id)
        if existing is None:
            try:
                self.repository.add_like(db, user_id, cocktail_id)
                db.commit()
            except IntegrityError:
                # 동시 요청 경합 — 이미 좋아요된 상태로 간주.
                db.rollback()
        like_count = self.repository.count_likes(db, cocktail_id)
        return {
            "cocktailId": cocktail_id,
            "isLiked": True,
            "likeCount": like_count,
            "message": "좋아요 성공",
        }

    def unlike(self, db: Session, user_id: int, cocktail_id: int) -> dict:
        if not self.repository.cocktail_exists(db, cocktail_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="칵테일을 찾을 수 없습니다.",
            )
        existing = self.repository.get_like(db, user_id, cocktail_id)
        if existing is not None:
            self.repository.delete_like(db, existing)
            db.commit()
        like_count = self.repository.count_likes(db, cocktail_id)
        return {
            "cocktailId": cocktail_id,
            "isLiked": False,
            "likeCount": like_count,
            "message": "좋아요 취소 성공",
        }

    def like_list(self, db: Session, user_id: int) -> dict:
        cocktails = self.repository.list_liked_cocktails(db, user_id)
        counts = self.repository.like_counts_for(db, [c.id for c in cocktails])
        return {
            "cocktails": [
                {
                    "cocktailId": c.id,
                    "cocktailName": c.name,
                    "imageUrl": c.image_url or "",
                    "baseTag": c.base_tag or "",
                    "likeCount": counts.get(c.id, 0),
                    "isLiked": True,
                }
                for c in cocktails
            ]
        }
