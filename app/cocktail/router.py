"""칵테일 라우터 (Controller).

경로/응답은 기존 단일 파일 구현과 동일하게 유지한다 (prefix 없음).
"""
from fastapi import APIRouter, Depends

from sqlalchemy.orm import Session

from app.cocktail.schemas import (
    CocktailBrief,
    CocktailExplore,
    CocktailSummary,
    DrinkOfTheDay,
    SearchResult,
)
from app.cocktail.service import CocktailService

from app.core.database import get_db

router = APIRouter(tags=["cocktail"])
service = CocktailService()


@router.get("/")
def root():
    return {"message": "Hello World"}


@router.get("/cocktail/{id}", response_model=CocktailBrief)
def get_cocktail(id: int):
    return service.get_brief(id)


@router.get("/search", response_model=SearchResult)
def search_cocktails(keyword: str = "", page: int = 1, rpp: int = 10):
    return service.search(keyword, page, rpp)


@router.get("/list", response_model=list[CocktailSummary])
def cocktail_list(db: Session = Depends(get_db)):
    return service.list_cocktails(db)


@router.get("/explore/{id}", response_model=CocktailExplore)
def explore_cocktail(id: int):
    return service.explore(id)


@router.get("/drink-of-the-day", response_model=DrinkOfTheDay)
def drink_of_the_day():
    return service.drink_of_the_day()
