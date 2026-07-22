"""칵테일 응답 스키마 (View 계층).

프론트엔드 호환을 위해 필드명은 기존 mock 응답과 동일한 camelCase를 사용한다.
"""

from pydantic import BaseModel


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
    isLiked: bool = False
    likeCount: int = 0
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
    # likes 테이블 기준 칵테일별 좋아요 수.
    likeCount: int = 0


class CocktailListMeta(BaseModel):
    page: int
    rpp: int
    hasNextPage: bool


class CocktailListResponse(BaseModel):
    items: list[CocktailListItem]
    meta: CocktailListMeta


class BaseTagListResponse(BaseModel):
    items: list[str]


class DailyRecommendItem(BaseModel):
    id: int
    name: str
    description: str | None = None
    baseTag: str | None = None
    abv: float | None = None
    imageUrl: str | None = None


class DailyRecommendResponse(BaseModel):
    items: list[DailyRecommendItem]


class AutocompleteItem(BaseModel):
    id: int
    name: str
    nameEn: str | None = None


class AutocompleteResponse(BaseModel):
    keyword: str
    items: list[AutocompleteItem]
