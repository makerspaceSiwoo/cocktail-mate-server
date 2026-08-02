"""추천 라우터 — 특정 칵테일과 유사한 칵테일(클러스터 범위 내 최근접)."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.rate_limit import get_proxy_client_ip, limiter
from app.recommend.schemas import (
    RecommendItem,
    TasteDescriptorCatalogResponse,
    TasteRecommendRequest,
)
from app.recommend.service import RecommendService

router = APIRouter(tags=["recommend"])
service = RecommendService()

FLAVOR_RECOMMEND_RATE_LIMIT = "20/minute"


@router.get("/cocktail/{id}/recommend", response_model=list[RecommendItem])
def recommend_cocktail(id: int, db: Session = Depends(get_db)):
    return service.recommend(db, id)


@router.get(
    "/taste-descriptors",
    response_model=TasteDescriptorCatalogResponse,
)
def taste_descriptors(db: Session = Depends(get_db)):
    return service.taste_descriptor_catalog(db)


@router.post(
    "/flavor/recommend",
    response_model=list[RecommendItem],
)
@limiter.limit(FLAVOR_RECOMMEND_RATE_LIMIT, key_func=get_proxy_client_ip)
def recommend_by_flavor(
    request: Request,
    payload: TasteRecommendRequest,
    db: Session = Depends(get_db),
):
    return service.recommend_by_taste(db, payload.descriptorIds)
