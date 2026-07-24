"""칵테일 라우터 (Controller).

경로/응답은 기존 단일 파일 구현과 동일하게 유지한다 (prefix 없음).
"""

from fastapi import APIRouter, Depends, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from sqlalchemy.orm import Session

from app.cocktail.schemas import (
    AutocompleteResponse,
    DailyRecommendResponse,
    CocktailListResponse,
    BaseTagListResponse,
)
from app.cocktail import service as cocktail_service
from app.cocktail.service import CocktailService
from app.auth.dependencies import OptionalUser
from app.core.database import get_db
from app.core.rate_limit import limiter
from app.cocktail.schemas import CocktailDetailResponse

router = APIRouter(tags=["cocktail"])
service = CocktailService()

# 자동완성은 인증 없이 열려 있고 요청당 인덱스 전수 스캔 비용이 든다
# (622건 기준 일반 1~2ms, MAX_LEN 입력 ~5ms). 프론트에서도 스로틀링하지만,
# 그건 클라이언트 선의에 의존하므로 서버에서 IP당 상한을 별도로 둔다.
# 사람의 실제 타이핑에는 넉넉하고 단일 IP가 CPU를 포화시키지는 못하는 값.
AUTOCOMPLETE_RATE_LIMIT = "120/minute"


@router.get("/")
def root():
    return {"message": "Hello World"}


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


@router.get("/daily-recommend", response_model=DailyRecommendResponse)
def daily_recommend(db: Session = Depends(get_db)):
    return service.daily_recommend(db)


@router.get("/cocktail/{id}", response_model=CocktailDetailResponse)
def get_cocktail_detail(
    id: int,
    current_user: OptionalUser,
    db: Session = Depends(get_db),
):
    user_id = current_user.id if current_user is not None else None
    return service.get_detail(db, id, user_id)


@router.get("/search/autocomplete", response_model=AutocompleteResponse)
@limiter.limit(AUTOCOMPLETE_RATE_LIMIT)
def search_autocomplete(
    # `request` is required by slowapi to resolve the client key (remote IP).
    # FastAPI treats it as a special param, so the query contract is unchanged.
    request: Request,
    keyword: str = Query(""),
    limit: int = Query(10, ge=1, le=50),
    debug: bool = Query(False),
    db: Session = Depends(get_db),
):
    result = cocktail_service.autocomplete(db, keyword, limit, debug)
    if debug:
        # Intentional: JSONResponse bypasses response_model so debug-only fields
        # (score, tier, matchedField) are preserved; response_model would strip them.
        return JSONResponse(content=jsonable_encoder(result))
    return result


@router.get("/search", response_model=CocktailListResponse)
def search_cocktails(
    current_user: OptionalUser,
    keyword: str = Query(...),
    page: int = Query(1, ge=1),
    rpp: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    user_id = current_user.id if current_user is not None else None

    return service.search_cocktails(
        db=db,
        keyword=keyword,
        page=page,
        rpp=rpp,
        user_id=user_id,
    )