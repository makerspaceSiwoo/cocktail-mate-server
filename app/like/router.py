"""좋아요 라우터. 경로/응답은 기존 구현과 동일 (prefix 없음)."""
from fastapi import APIRouter

from app.like.schemas import LikeActionResponse, LikeListResponse
from app.like.service import LikeService

router = APIRouter(tags=["like"])
service = LikeService()


@router.get("/like/list", response_model=LikeListResponse)
def get_like_list():
    return service.like_list()


@router.post("/like", response_model=LikeActionResponse)
def like_cocktail():
    return service.like()


@router.delete("/unlike", response_model=LikeActionResponse)
def unlike_cocktail():
    return service.unlike()
