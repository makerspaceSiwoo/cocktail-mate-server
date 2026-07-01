"""칵테일 응답 스키마 (View 계층).

프론트엔드 호환을 위해 필드명은 기존 mock 응답과 동일한 camelCase를 사용한다.
"""
from pydantic import BaseModel


class CocktailSummary(BaseModel):
    id: int
    imageUrl: str
    name: str
    baseTag: str
    description: str
    ABV: float
    numLike: int


class CocktailBrief(BaseModel):
    cocktailId: int
    imageUrl: str
    cocktailName: str


class SearchResult(BaseModel):
    total: int
    cocktails: list[CocktailSummary]


class CocktailExplore(BaseModel):
    cocktailId: int
    cocktailName: str
    imageUrl: str
    glass: str
    ABV: float
    numLike: int
    recipe: str
    description: str
    baseTag: str
    isLiked: bool


class DrinkOfTheDay(BaseModel):
    cocktailId: int
    cocktailName: str
    imageUrl: str
    description: str
    ABV: float


class CocktailDetailResponse(BaseModel):
    name: str
    recipe: str | None = None
    description: str | None = None