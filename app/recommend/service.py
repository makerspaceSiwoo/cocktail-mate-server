"""추천 비즈니스 로직 — 클러스터 범위(코사인 유사도 하한) 내 최근접 칵테일."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Protocol

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.recommend.cache import TasteRecommendationCache
from app.recommend.repository import RecommendRepository
from app.taste_query.model import get_taste_query_model

# 클러스터 하한: 코사인 유사도 0.65 이상만 (거리 0.35 이하). 낮은 유사도는 억지로 추천하지 않는다.
MIN_SIMILARITY = 0.65
MAX_COSINE_DISTANCE = 1.0 - MIN_SIMILARITY
RECOMMEND_LIMIT = 5


class TasteQueryEncoder(Protocol):
    def encode(self, codes: list[str] | tuple[str, ...]): ...


class RecommendService:
    def __init__(
        self,
        repository: RecommendRepository | None = None,
        taste_model_loader: Callable[[], TasteQueryEncoder] = get_taste_query_model,
        taste_cache: TasteRecommendationCache | None = None,
    ) -> None:
        self.repository = repository or RecommendRepository()
        self.taste_model_loader = taste_model_loader
        self.taste_cache = taste_cache or TasteRecommendationCache()

    def recommend(self, db: Session, cocktail_id: int) -> list[dict]:
        cocktail = self.repository.get_by_id(db, cocktail_id)
        if cocktail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="칵테일을 찾을 수 없습니다.",
            )
        # 임베딩이 아직 없는 칵테일 → 추천 불가, 빈 목록.
        if cocktail.embedding is None:
            return []
        return self.repository.nearest_within(
            db,
            cocktail.embedding,
            cocktail_id,
            MAX_COSINE_DISTANCE,
            RECOMMEND_LIMIT,
        )

    def taste_descriptor_catalog(self, db: Session) -> dict:
        descriptors = self.repository.list_taste_descriptors(db)
        try:
            supported_codes = self.taste_model_loader().supported_codes
        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="맛 추천 모델을 사용할 수 없습니다.",
            ) from error
        descriptors = [
            descriptor
            for descriptor in descriptors
            if descriptor.code in supported_codes
        ]
        return {
            "items": [
                {
                    "id": descriptor.id,
                    "code": descriptor.code,
                    "labelKo": descriptor.label_ko,
                    "category": descriptor.category,
                }
                for descriptor in descriptors
            ],
            "maxSelectionsPerCategory": 1,
        }

    def recommend_by_taste(
        self,
        db: Session,
        descriptor_ids: list[int],
    ) -> list[dict]:
        if not descriptor_ids:
            return self.repository.random_cocktails(db, RECOMMEND_LIMIT)
        cache_key = tuple(sorted(descriptor_ids))
        return self.taste_cache.get_or_compute(
            cache_key,
            lambda: self._recommend_by_taste_uncached(db, list(cache_key)),
        )

    def _recommend_by_taste_uncached(
        self,
        db: Session,
        descriptor_ids: list[int],
    ) -> list[dict]:
        descriptors = self.repository.active_taste_descriptors_by_ids(
            db,
            descriptor_ids,
        )
        found_ids = {descriptor.id for descriptor in descriptors}
        missing_ids = sorted(set(descriptor_ids) - found_ids)
        if missing_ids:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "unknown_taste_descriptors",
                    "descriptorIds": missing_ids,
                },
            )
        category_ids: dict[str, list[int]] = defaultdict(list)
        for descriptor in descriptors:
            category_ids[descriptor.category].append(descriptor.id)
        duplicate_categories = [
            {"category": category, "descriptorIds": ids}
            for category, ids in sorted(category_ids.items())
            if len(ids) > 1
        ]
        if duplicate_categories:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "multiple_descriptors_in_category",
                    "categories": duplicate_categories,
                },
            )
        try:
            query_embedding = self.taste_model_loader().encode(
                [descriptor.code for descriptor in descriptors]
            )
        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="맛 추천 모델을 사용할 수 없습니다.",
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "unsupported_taste_descriptors",
                    "message": str(error),
                },
            ) from error
        return self.repository.nearest_for_virtual_taste(
            db,
            query_embedding.tolist(),
            RECOMMEND_LIMIT,
        )
