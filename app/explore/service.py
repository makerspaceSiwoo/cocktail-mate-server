"""탐색 비즈니스 로직."""

from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.explore.repository import ExploreRepository


class ExploreService:
    def __init__(self, repository: ExploreRepository | None = None) -> None:
        self.repository = repository or ExploreRepository()

    def list_all(self, db: Session) -> list[dict]:
        return self.repository.list_with_embedding_3d(db)

    def get_detail(self, db: Session, cocktail_id: int) -> dict:
        cocktail = self.repository.get_by_id(db, cocktail_id)
        if cocktail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="칵테일을 찾을 수 없습니다.",
            )
        return cocktail
