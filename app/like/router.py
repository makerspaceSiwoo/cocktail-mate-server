"""좋아요 라우터.

- POST   /like       : 좋아요 등록 (로그인 필수)
- DELETE /unlike     : 좋아요 해제 (로그인 필수)
- GET    /like/list  : 내 좋아요 목록 (로그인 필수)
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from cocktail_mate_db.models import User

from app.auth.dependencies import get_current_user
from app.core.database import get_db
from app.like.schemas import (
    LikeActionResponse,
    LikeListResponse,
    LikeRequest,
)
from app.like.service import LikeService

router = APIRouter(tags=["like"])
service = LikeService()


@router.post("/like", response_model=LikeActionResponse)
def like_cocktail(
    payload: LikeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return service.like(db, current_user.id, payload.cocktailId)


@router.delete("/unlike", response_model=LikeActionResponse)
def unlike_cocktail(
    payload: LikeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return service.unlike(db, current_user.id, payload.cocktailId)


@router.get("/like/list", response_model=LikeListResponse)
def get_like_list(
    page: int = Query(1, ge=1),
    rpp: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return service.like_list(db, current_user.id, page, rpp)
