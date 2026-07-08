"""좋아요 응답 스키마."""

from pydantic import BaseModel


class LikeRequest(BaseModel):
    cocktailId: int | None = None


class LikedCocktail(BaseModel):
    cocktailId: int
    cocktailName: str
    imageUrl: str
    baseTag: str
    likeCount: int
    isLiked: bool


class LikeListResponse(BaseModel):
    cocktails: list[LikedCocktail]


class LikeActionResponse(BaseModel):
    cocktailId: int
    isLiked: bool
    likeCount: int
    message: str