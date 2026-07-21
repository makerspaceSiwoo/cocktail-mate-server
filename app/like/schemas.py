"""좋아요 응답 스키마."""

from pydantic import BaseModel, Field


class LikeRequest(BaseModel):
    cocktailId: int = Field(gt=0)


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
