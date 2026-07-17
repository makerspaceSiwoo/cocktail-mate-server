"""탐색 비즈니스 로직."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.explore.repository import ExploreRepository


class ExploreService:
    def __init__(self, repository: ExploreRepository | None = None) -> None:
        self.repository = repository or ExploreRepository()

    def list_all(self, db: Session) -> list[dict]:
        return self.repository.list_with_embedding_3d(db)
