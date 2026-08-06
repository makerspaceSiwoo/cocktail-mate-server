"""추천 비즈니스 로직 — 48D 감각 벡터 기반 최근접 추천.

칵테일↔칵테일 추천은 `cocktails.embedding`(Graph48, unit-L2, 코사인)을 쓰고,
맛 선택 추천은 `cocktails.preference_embedding`(Preference48, 비정규화, 내적)을 쓴다.

## 절대 유사도 임계값을 두지 않는 이유 (실측 근거 — 되돌리지 말 것)

과거에는 `MIN_SIMILARITY = 0.65`(= 코사인 거리 0.35 이하)라는 절대 임계값이 있었다.
그 값은 **32D 임베딩 시절에 보정된 값**이며 48D Graph48에서는 아무것도 걸러내지 못한다.
실측(602×601 전체 쌍, Graph48):

    코사인 p10 = 0.5479, p50 = 0.7546, p90 = 0.8974

즉 `cos >= 0.65`는 **전체 쌍의 75.4%**를 통과시킨다 — 소스 하나당 601개 후보 중
평균 453개다. 필터로서 죽은 값이다. 그래서 절대 임계값을 제거하고 **항상 최근접 5개**를
반환한다(`nearest_for_virtual_taste`가 원래 이 형태였다). 결과적으로 모든 칵테일이
항상 5개를 받고 빈 응답이 사라진다.

**32D 시절 상수를 다시 도입하지 말 것.** 다시 필요해지면 위 분포를 새로 측정한 뒤
rank 기반(예: 소스별 rank-5 코사인 중앙값 0.9485)으로 재보정해야 한다.
"""

from __future__ import annotations

from collections import defaultdict

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.recommend.cache import FLAVOR_CACHE_EPOCH, TasteRecommendationCache
from app.recommend.repository import RecommendRepository
from app.sensory_embedding import build_user_query

RECOMMEND_LIMIT = 5


class RecommendService:
    def __init__(
        self,
        repository: RecommendRepository | None = None,
        taste_cache: TasteRecommendationCache | None = None,
    ) -> None:
        self.repository = repository or RecommendRepository()
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
            RECOMMEND_LIMIT,
        )

    def taste_descriptor_catalog(self, db: Session) -> dict:
        # 48개 감각축 어휘를 그대로 노출한다. 예전에는 GNN 아티팩트의
        # `supported_codes`로 걸렀지만 GNN 경로가 폐기되면서 그 필터는 영구 503을
        # 뜻하게 됐다. 어휘의 canonical 출처는 이제 `taste_descriptors` 테이블뿐이다.
        descriptors = self.repository.list_taste_descriptors(db)
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
        # 캐시 키에 epoch를 섞는다. 임베딩을 다시 적재하면 같은 descriptor 조합이
        # 다른 결과를 뜻하므로, epoch를 올려 24시간 TTL을 즉시 무효화한다.
        cache_key = (FLAVOR_CACHE_EPOCH, *sorted(descriptor_ids))
        return self.taste_cache.get_or_compute(
            cache_key,
            lambda: self._recommend_by_taste_uncached(db, sorted(descriptor_ids)),
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
        # `taste_descriptors.code`는 감각축 `axis_id`와 같은 값이다. 선택한 축에만
        # 질량을 주는 48D Preference 쿼리(비음수, L1 == 1)를 만든다.
        try:
            query = build_user_query(
                {descriptor.code: 1.0 for descriptor in descriptors}
            )
        except ValueError as error:
            # DB 어휘가 감각축 레지스트리를 벗어난 경우에만 발생한다(스키마 불일치).
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "unsupported_taste_descriptors",
                    "message": str(error),
                },
            ) from error
        matches = self.repository.nearest_by_preference(
            db,
            list(query.values),
            RECOMMEND_LIMIT,
        )
        if not matches:
            # `preference_embedding`이 전부 NULL(=적재 미완료)일 때만 도달한다.
            # 예외는 캐시에 저장되지 않으므로 적재 후 즉시 정상 응답으로 돌아온다.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="맛 추천 임베딩이 아직 준비되지 않았습니다.",
            )
        return matches
