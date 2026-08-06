"""취향 추천 비즈니스 로직 — 최근 좋아요별 ANN 후보 병합.

`cocktails.embedding`(Graph48, unit-L2, 코사인)을 읽는다.

## 절대 유사도 임계값을 두지 않는 이유 (실측 근거 — 되돌리지 말 것)

과거에는 `MIN_SIMILARITY = 0.6`(= 코사인 거리 0.4 이하)이 있었다. 32D 시절 값이며
48D Graph48에서는 필터 역할을 못 한다. 실측(602×601 전체 쌍, Graph48):

    코사인 p10 = 0.5479, p50 = 0.7546, p90 = 0.8974

`cos >= 0.65`가 전체 쌍의 **75.4%**(소스당 평균 453/601개), `cos >= 0.6`은 **83.9%**
(평균 504개)를 통과시킨다. 그래서 절대 임계값을 제거하고 좋아요마다 **최근접 5개**를
가져와 병합한다. 자세한 배경은 `app/recommend/service.py` 모듈 docstring 참고.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.favor.repository import FavorRepository

RECENT_LIKES_LIMIT = 5
FAVOR_LIMIT = 5


class FavorService:
    def __init__(self, repository: FavorRepository | None = None) -> None:
        self.repository = repository or FavorRepository()

    def recommend(self, db: Session, user_id: int) -> list[dict]:
        embeddings = self.repository.liked_embeddings(
            db,
            user_id,
            limit=RECENT_LIKES_LIMIT,
        )
        if not embeddings:
            return []
        exclude_ids = self.repository.liked_ids(db, user_id)
        candidates_by_id: dict[int, dict] = {}
        for embedding in embeddings:
            candidates = self.repository.nearest_within(
                db,
                embedding,
                exclude_ids,
                FAVOR_LIMIT,
            )
            for candidate in candidates:
                existing = candidates_by_id.get(candidate["id"])
                if existing is None or candidate["similarity"] > existing["similarity"]:
                    candidates_by_id[candidate["id"]] = candidate

        return sorted(
            candidates_by_id.values(),
            key=lambda candidate: (-candidate["similarity"], candidate["id"]),
        )[:FAVOR_LIMIT]
