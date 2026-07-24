"""칵테일 비즈니스 로직 (Controller가 호출)."""

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.cocktail.repository import CocktailRepository
from app.like.repository import LikeRepository

from fastapi import HTTPException

from app.cocktail.search import registry
from app.cocktail.search.normalize import sanitize_keyword
from app.cocktail.schemas import AutocompleteItem, AutocompleteResponse

# MVP 기준 한국 시간(UTC+9)의 날짜를 오늘의 추천 시드로 사용한다.
KST = timezone(timedelta(hours=9))


class CocktailService:
    def __init__(
        self,
        repository: CocktailRepository | None = None,
        like_repository: LikeRepository | None = None,
    ) -> None:
        self.repository = repository or CocktailRepository()
        self.like_repository = like_repository or LikeRepository()

    def list_cocktails(
        self,
        db: Session,
        page: int,
        rpp: int,
        base: str | None,
        user_id: int | None = None,
    ) -> dict:
        liked_ids = (
            self.like_repository.liked_cocktail_ids(db, user_id)
            if user_id is not None
            else None
        )
        result = self.repository.list_all(db, page, rpp, base, liked_ids)

        # 현재 페이지 칵테일들의 좋아요 수를 한 번에 조회해 likeCount 를 채운다.
        like_counts = self.like_repository.like_counts_for(
            db, [item["id"] for item in result["items"]]
        )
        for item in result["items"]:
            item["likeCount"] = like_counts.get(item["id"], 0)

        return result

    def get_base_tags(self, db: Session) -> dict:
        return self.repository.get_base_tags(db)

    def daily_recommend(self, db: Session, count: int = 5) -> dict:
        # KST 날짜(YYYY-MM-DD)를 시드로 → 같은 날은 모든 유저에게 동일한 추천.
        seed = datetime.now(KST).strftime("%Y-%m-%d")
        return {"items": self.repository.daily_cocktails(db, seed, count)}

    def get_detail(
        self,
        db: Session,
        cocktail_id: int,
        user_id: int | None = None,
    ) -> dict:
        cocktail = self.repository.find_detail_by_id(db, cocktail_id)

        if cocktail is None:
            raise HTTPException(status_code=404, detail="Cocktail not found")

        cocktail["likeCount"] = self.like_repository.count_likes(
            db,
            cocktail_id,
        )

        cocktail["isLiked"] = (
            self.like_repository.get_like(
                db,
                user_id,
                cocktail_id,
            )
            is not None
            if user_id is not None
            else False
        )

        return cocktail


    def search_cocktails(
        self,
        db: Session,
        keyword: str,
        page: int,
        rpp: int,
        user_id: int | None = None,
    ) -> dict:
        keyword = sanitize_keyword(keyword)
        offset = (page - 1) * rpp

        index = registry.ensure_index(db)
        search_result = index.search(
            keyword=keyword,
            limit=rpp,
            offset=offset,
        )

        cocktail_ids = [hit.id for hit in search_result.hits]

        liked_ids = (
            self.like_repository.liked_cocktail_ids(db, user_id)
            if user_id is not None
            else None
        )

        items = self.repository.list_by_ids(
            db=db,
            cocktail_ids=cocktail_ids,
            liked_ids=liked_ids,
        )

        like_counts = self.like_repository.like_counts_for(
            db,
            cocktail_ids,
        )

        for item in items:
            item["likeCount"] = like_counts.get(item["id"], 0)

        return {
            "items": items,
            "meta": {
                "page": page,
                "rpp": rpp,
                "hasNextPage": (
                    offset + len(search_result.hits) < search_result.total
                ),
            },
        }

def autocomplete(
    db: Session,
    keyword: str,
    limit: int,
    debug: bool,
) -> AutocompleteResponse | dict:
    """Run the cocktail name search core and shape the API response.

    Response contract (binding -- do not fold into a shared item builder
    without preserving this split):
    - `debug=False` (normal): returns an `AutocompleteResponse` whose items
      carry ONLY `id`, `name`, `nameEn`. `nameEn` is always present (may be
      `None`).
    - `debug=True`: returns a plain `dict` (not a pydantic model) whose
      items additionally carry `matchType`, `matchedField`, `score`, and
      `tier`. Returning a plain dict here is deliberate: `router.py` wraps
      it in a `JSONResponse` to bypass `response_model`, since the router's
      declared `response_model=AutocompleteResponse` would otherwise strip
      the debug-only fields.
    """
    idx = registry.ensure_index(db)
    kw = sanitize_keyword(keyword)
    result = idx.search(kw, limit, offset=0)
    if debug:
        items = [
            {
                "id": hit.id,
                "name": hit.name,
                "nameEn": hit.name_en,
                "matchType": hit.match_type,
                "matchedField": hit.matched_field,
                "score": hit.score,
                "tier": hit.tier,
            }
            for hit in result.hits
        ]
        return {"keyword": kw, "items": items}
    items_pydantic: list[AutocompleteItem] = [
        AutocompleteItem(
            id=hit.id,
            name=hit.name,
            nameEn=hit.name_en,
        )
        for hit in result.hits
    ]
    return AutocompleteResponse(keyword=kw, items=items_pydantic)
