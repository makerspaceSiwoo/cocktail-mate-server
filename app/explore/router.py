"""탐색 라우터 — 전체 칵테일의 3D 임베딩 좌표(벡터 맵 시각화용)."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.explore.schemas import ExploreItem
from app.explore.service import ExploreService

router = APIRouter(tags=["explore"])
service = ExploreService()


@router.get("/explore", response_model=list[ExploreItem])
def explore_all(db: Session = Depends(get_db)):
    return service.list_all(db)
