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


class CocktailIngredientDetail(BaseModel):
    id: int
    name: str
    nameEn: str | None = None
    category: str | None = None
    amount: float | None = None
    unit: str | None = None
    description: str | None = None
    abv: float | None = None
    imageUrl: str | None = None
    potency: float | None = None


class CocktailDetailResponse(BaseModel):
    id: int
    name: str
    nameEn: str | None = None
    imageUrl: str | None = None
    glass: str | None = None
    abv: float | None = None
    recipe: list[str] | None = None
    description: str | None = None
    baseTag: str | None = None
    ingredients: list[CocktailIngredientDetail]


class CocktailListItem(BaseModel):
    id: int
    name: str
    nameEn: str | None = None
    imageUrl: str | None = None
    baseTag: str | None = None
    description: str | None = None
    abv: float | None = None
    glass: str | None = None
    # 로그인 시 해당 유저의 좋아요 여부. 비로그인은 항상 False.
    isLiked: bool = False


class CocktailListMeta(BaseModel):
    page: int
    rpp: int
    hasNextPage: bool


class CocktailListResponse(BaseModel):
    items: list[CocktailListItem]
    meta: CocktailListMeta


class BaseTagListResponse(BaseModel):
    items: list[str]


class LikeRequest(BaseModel):
    cocktailId: int