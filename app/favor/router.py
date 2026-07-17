"""취향 추천 라우터 — 로그인 유저의 좋아요 기반 추천 (로그인 필수)."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from cocktail_mate_db.models import User

from app.auth.dependencies import get_current_user
from app.core.database import get_db
from app.favor.schemas import FavorItem
from app.favor.service import FavorService

router = APIRouter(tags=["favor"])
service = FavorService()


@router.get("/user/favor", response_model=list[FavorItem])
def user_favor(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return service.recommend(db, current_user.id)
