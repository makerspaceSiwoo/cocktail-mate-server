"""칵테일 라우터 (Controller).

경로/응답은 기존 단일 파일 구현과 동일하게 유지한다 (prefix 없음).
"""

from fastapi import APIRouter, Depends, Query

from sqlalchemy.orm import Session

from app.cocktail.schemas import (
    CocktailExplore,
    DrinkOfTheDay,
    SearchResult,
    CocktailListResponse,
    BaseTagListResponse,
)
from app.cocktail.service import CocktailService
from app.auth.dependencies import OptionalUser
from app.core.database import get_db
from app.cocktail.schemas import CocktailDetailResponse

router = APIRouter(tags=["cocktail"])
service = CocktailService()


@router.get("/")
def root():
    return {"message": "Hello World"}


@router.get("/search", response_model=SearchResult)
def search_cocktails(keyword: str = "", page: int = 1, rpp: int = 10):
    return service.search(keyword, page, rpp)


@router.get("/cocktail/base-tags", response_model=BaseTagListResponse)
def get_base_tags(db: Session = Depends(get_db)):
    return service.get_base_tags(db)


@router.get("/list", response_model=CocktailListResponse)
def cocktail_list(
    current_user: OptionalUser,
    page: int = Query(1, ge=1),
    rpp: int = Query(10, ge=1, le=50),
    base: str | None = None,
    db: Session = Depends(get_db),
):
    user_id = current_user.id if current_user is not None else None
    return service.list_cocktails(db, page, rpp, base, user_id)


@router.get("/explore/{id}", response_model=CocktailExplore)
def explore_cocktail(id: int):
    return service.explore(id)


@router.get("/drink-of-the-day", response_model=DrinkOfTheDay)
def drink_of_the_day():
    return service.drink_of_the_day()


@router.get("/cocktail/{id}", response_model=CocktailDetailResponse)
def get_cocktail_detail(id: int, db: Session = Depends(get_db)):
    return service.get_detail(db, id)
