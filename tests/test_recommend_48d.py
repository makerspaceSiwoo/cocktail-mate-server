"""48D 전환 후의 추천 API 계약 테스트.

- `/cocktail/{id}/recommend` 와 `/user/favor` 는 `cocktails.embedding` 을 코사인으로
  읽고 **절대 거리 임계값이 없다**.
- `/flavor/recommend` 는 `cocktails.preference_embedding` 을 **내적(`<#>`)** 으로 읽는다.
- `/taste-descriptors` 는 48행 8카테고리를 그대로 돌려주며 폐기된 GNN을 건드리지 않는다.
- `build_user_query` → pgvector `<#>` 순위가 오프라인 `score_preference_mips` 와 일치한다.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.dialects import postgresql

from app.core.database import get_db
from app.core.rate_limit import limiter
from app.favor.repository import FavorRepository
from app.recommend.cache import FLAVOR_CACHE_EPOCH, TasteRecommendationCache
from app.recommend.repository import RecommendRepository
from app.recommend.router import (
    router as recommend_router,
    service as recommend_service,
)
from app.sensory_embedding import (
    SENSORY_V2_REGISTRY,
    build_user_query,
    preference48_cocktail_adapter,
    preference48_contract,
    score_preference_mips,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _compiled(statement) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def _last_select_sql(db: Mock) -> str:
    """마지막 `db.execute(select(...))` 호출을 PostgreSQL SQL 문자열로 돌려준다."""

    for call in reversed(db.execute.call_args_list):
        statement = call.args[0]
        if hasattr(statement, "compile") and hasattr(statement, "whereclause"):
            return _compiled(statement)
    raise AssertionError("no SELECT statement was executed")


# --- 컴파일 SQL: Graph48 코사인, 임계값 없음 -------------------------------------


def test_cocktail_recommend_uses_cosine_on_embedding_without_a_threshold() -> None:
    db = Mock()
    db.execute.return_value.all.return_value = []

    RecommendRepository().nearest_within(db, [0.1] * 48, exclude_id=7, limit=5)

    sql = _last_select_sql(db)
    assert "cocktails.embedding <=>" in sql
    assert "ORDER BY cocktails.embedding <=>" in sql
    assert "cocktails.preference_embedding" not in sql
    assert "ingredients" not in sql
    # 거리 임계값이 있었다면 WHERE 절에 `<= 0.35` 같은 비교가 남는다.
    assert "<= 0.35" not in sql
    assert "0.35" not in sql
    where_clause = sql.split("WHERE", 1)[1].split("ORDER BY", 1)[0]
    assert "<=>" not in where_clause
    assert "LIMIT 5" in sql


def test_favor_recommend_uses_cosine_on_embedding_without_a_threshold() -> None:
    db = Mock()
    db.execute.return_value.all.return_value = []

    FavorRepository().nearest_within(db, [0.1] * 48, exclude_ids={3, 4}, limit=5)

    sql = _last_select_sql(db)
    assert "cocktails.embedding <=>" in sql
    assert "ORDER BY cocktails.embedding <=>" in sql
    assert "0.4" not in sql
    where_clause = sql.split("WHERE", 1)[1].split("ORDER BY", 1)[0]
    assert "<=>" not in where_clause
    assert "cocktails.id NOT IN" in sql
    assert "LIMIT 5" in sql


def test_favor_service_no_longer_exposes_a_similarity_threshold() -> None:
    import app.favor.service as favor_service
    import app.recommend.service as recommend_module

    for module in (favor_service, recommend_module):
        assert not hasattr(module, "MIN_SIMILARITY")
        assert not hasattr(module, "MAX_COSINE_DISTANCE")


# --- 컴파일 SQL: Preference48 내적 -----------------------------------------------


def test_flavor_recommend_ranks_preference_embedding_by_inner_product() -> None:
    db = Mock()
    db.execute.return_value.all.return_value = []

    RecommendRepository().nearest_by_preference(db, [0.5] * 48, limit=5)

    sql = _last_select_sql(db)
    assert "cocktails.preference_embedding <#>" in sql
    assert "ORDER BY cocktails.preference_embedding <#>" in sql
    # 48D 코사인 컬럼은 이 경로에서 전혀 쓰이지 않는다.
    assert "<=>" not in sql
    assert "cocktails.preference_embedding IS NOT NULL" in sql
    assert "LIMIT 5" in sql


def test_flavor_recommend_negates_the_pgvector_inner_product_distance() -> None:
    db = Mock()
    db.execute.return_value.all.return_value = [
        SimpleNamespace(
            id=11,
            name="A",
            name_en="A",
            description=None,
            image_url=None,
            abv=None,
            dist=-0.1234,
        ),
        SimpleNamespace(
            id=12,
            name="B",
            name_en="B",
            description=None,
            image_url=None,
            abv=None,
            dist=-0.0100,
        ),
    ]

    rows = RecommendRepository().nearest_by_preference(db, [0.5] * 48, limit=5)

    # `<#>` 는 음의 내적이므로 similarity = -distance 이고, 오름차순 거리는 곧
    # 내림차순 유사도다.
    assert [row["similarity"] for row in rows] == [0.1234, 0.01]


# --- HTTP 레벨 -------------------------------------------------------------------


class _FakeDescriptor:
    def __init__(self, descriptor_id: int, code: str, category: str) -> None:
        self.id = descriptor_id
        self.code = code
        self.label_ko = f"라벨-{code}"
        self.category = category
        self.is_active = True


ALL_DESCRIPTORS = [
    _FakeDescriptor(axis.axis_order + 1, axis.axis_id, axis.category)
    for axis in SENSORY_V2_REGISTRY.axes
]
DESCRIPTOR_BY_ID = {descriptor.id: descriptor for descriptor in ALL_DESCRIPTORS}


class _FakeRepository:
    def __init__(self) -> None:
        self.preference_calls: list[list[float]] = []
        self.random_calls = 0
        self.preference_rows: list[dict] = [
            {
                "id": 100 + index,
                "name": f"C{index}",
                "nameEn": None,
                "description": None,
                "imageUrl": None,
                "abv": None,
                "similarity": 0.5 - index / 100,
            }
            for index in range(5)
        ]

    def list_taste_descriptors(self, db):  # noqa: ARG002
        return list(ALL_DESCRIPTORS)

    def active_taste_descriptors_by_ids(self, db, descriptor_ids):  # noqa: ARG002
        return [
            DESCRIPTOR_BY_ID[descriptor_id]
            for descriptor_id in sorted(descriptor_ids)
            if descriptor_id in DESCRIPTOR_BY_ID
        ]

    def nearest_by_preference(self, db, query_vector, limit):  # noqa: ARG002
        self.preference_calls.append(list(query_vector))
        return self.preference_rows[:limit]

    def random_cocktails(self, db, limit):  # noqa: ARG002
        self.random_calls += 1
        return self.preference_rows[:limit]


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    repository = _FakeRepository()
    monkeypatch.setattr(recommend_service, "repository", repository)
    monkeypatch.setattr(
        recommend_service,
        "taste_cache",
        TasteRecommendationCache(),
    )
    limiter.reset()

    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(recommend_router)
    app.dependency_overrides[get_db] = lambda: Mock()
    with TestClient(app) as test_client:
        test_client.repository = repository
        yield test_client


def test_taste_descriptors_returns_48_items_across_8_categories(client) -> None:
    response = client.get("/taste-descriptors")

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 48
    assert body["maxSelectionsPerCategory"] == 1
    assert sorted(body["items"][0]) == ["category", "code", "id", "labelKo"]
    categories = {item["category"] for item in body["items"]}
    assert len(categories) == 8
    assert categories == {
        "alcohol",
        "aroma",
        "body",
        "finish",
        "fruit",
        "mouthfeel",
        "taste_chemosensory",
        "temperature",
    }


def test_recommend_package_never_imports_the_retired_gnn() -> None:
    """`app.taste_query` 는 dormant다 — 추천 경로가 import조차 하지 않아야 한다."""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, app.recommend.router, app.recommend.service;"
            " print('app.taste_query' in sys.modules)",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "False"
    assert (REPOSITORY_ROOT / "app" / "taste_query" / "model.py").exists()


def test_flavor_recommend_builds_a_category_balanced_48d_query(client) -> None:
    # sweetness(taste_chemosensory) + citrus_fruit(fruit)
    ids = [
        next(d.id for d in ALL_DESCRIPTORS if d.code == "sweetness"),
        next(d.id for d in ALL_DESCRIPTORS if d.code == "citrus_fruit"),
    ]

    response = client.post("/flavor/recommend", json={"descriptorIds": ids})

    assert response.status_code == 200
    assert len(response.json()) == 5
    (query_vector,) = client.repository.preference_calls
    assert len(query_vector) == 48
    assert min(query_vector) >= 0.0
    assert sum(query_vector) == pytest.approx(1.0, abs=1e-12)
    assert sum(1 for value in query_vector if value > 0.0) == 2
    expected = build_user_query({"sweetness": 1.0, "citrus_fruit": 1.0})
    assert query_vector == pytest.approx(list(expected.values), abs=0.0)


def test_flavor_recommend_rejects_unknown_descriptor_ids(client) -> None:
    response = client.post("/flavor/recommend", json={"descriptorIds": [1, 9999]})

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "unknown_taste_descriptors",
        "descriptorIds": [9999],
    }
    assert client.repository.preference_calls == []


def test_flavor_recommend_rejects_two_descriptors_in_one_category(client) -> None:
    ids = [
        next(d.id for d in ALL_DESCRIPTORS if d.code == "sweetness"),
        next(d.id for d in ALL_DESCRIPTORS if d.code == "bitterness"),
    ]

    response = client.post("/flavor/recommend", json={"descriptorIds": ids})

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "multiple_descriptors_in_category",
        "categories": [
            {"category": "taste_chemosensory", "descriptorIds": sorted(ids)}
        ],
    }
    assert client.repository.preference_calls == []


def test_flavor_recommend_with_an_empty_selection_returns_random_five(client) -> None:
    response = client.post("/flavor/recommend", json={"descriptorIds": []})

    assert response.status_code == 200
    assert len(response.json()) == 5
    assert client.repository.random_calls == 1
    assert client.repository.preference_calls == []


def test_flavor_recommend_rejects_more_than_seven_descriptors(client) -> None:
    response = client.post(
        "/flavor/recommend",
        json={"descriptorIds": [descriptor.id for descriptor in ALL_DESCRIPTORS[:8]]},
    )

    assert response.status_code == 422
    assert client.repository.preference_calls == []


def test_flavor_recommend_rejects_duplicate_descriptor_ids(client) -> None:
    response = client.post("/flavor/recommend", json={"descriptorIds": [1, 1]})

    assert response.status_code == 422
    assert client.repository.preference_calls == []


def test_flavor_recommend_returns_503_when_no_preference_embedding_is_loaded(
    client,
) -> None:
    client.repository.preference_rows = []
    ids = [next(d.id for d in ALL_DESCRIPTORS if d.code == "sweetness")]

    response = client.post("/flavor/recommend", json={"descriptorIds": ids})

    assert response.status_code == 503
    # 503은 캐시되지 않는다 — 적재가 끝나면 즉시 정상 응답으로 돌아와야 한다.
    client.repository.preference_rows = [
        {
            "id": 1,
            "name": "X",
            "nameEn": None,
            "description": None,
            "imageUrl": None,
            "abv": None,
            "similarity": 0.9,
        }
    ]
    assert client.post(
        "/flavor/recommend", json={"descriptorIds": ids}
    ).status_code == (200)


# --- 캐시 무효화 -------------------------------------------------------------------


def test_flavor_cache_key_is_namespaced_by_the_embedding_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.recommend.service import RecommendService

    cache = TasteRecommendationCache()
    repository = _FakeRepository()
    service = RecommendService(repository=repository, taste_cache=cache)
    db = Mock()
    ids = [next(d.id for d in ALL_DESCRIPTORS if d.code == "sweetness")]

    service.recommend_by_taste(db, ids)
    service.recommend_by_taste(db, ids)

    assert len(repository.preference_calls) == 1
    (key,) = cache._entries
    assert key == (FLAVOR_CACHE_EPOCH, *ids)

    # epoch를 올리면 같은 선택이 새로 계산된다(배포/재적재 시 무효화 경로).
    monkeypatch.setattr("app.recommend.service.FLAVOR_CACHE_EPOCH", "next-run")
    service.recommend_by_taste(db, ids)
    assert len(repository.preference_calls) == 2
    assert ("next-run", *ids) in cache._entries

    cache.clear()
    assert not cache._entries


# --- 실 데이터 정합: build_user_query -> pgvector `<#>` == score_preference_mips ----

REHEARSAL_DSN = os.getenv("REHEARSAL_DATABASE_URL", "").strip()
PARITY_SELECTIONS = (
    ("sweetness",),
    ("sweetness", "citrus_fruit"),
    ("bitterness", "herb_botanical_aroma", "coldness"),
    ("sourness", "berry_fruit", "creamy_velvety_mouthfeel", "finish_length"),
    ("umami", "tropical_fruit", "floral_aroma", "light_body", "alcohol_heat"),
    (
        "pungency",
        "melon_fruit",
        "oak_smoky_aroma",
        "juicy_mouthfeel",
        "dry_finish",
        "heavy_body",
        "coldness",
    ),
)


@pytest.mark.skipif(
    not REHEARSAL_DSN,
    reason="REHEARSAL_DATABASE_URL is not set (live 48D database required)",
)
def test_preference_mips_parity_between_pgvector_and_the_offline_scorer() -> None:
    from cocktail_mate_db.models import Cocktail
    from sqlalchemy import create_engine, select, text
    from sqlalchemy.orm import Session

    contract = preference48_contract(registry=SENSORY_V2_REGISTRY)
    adapter = preference48_cocktail_adapter(contract)
    known_axes = {axis.axis_id for axis in SENSORY_V2_REGISTRY.axes}

    engine = create_engine(REHEARSAL_DSN, future=True)
    repository = RecommendRepository()
    try:
        with Session(engine) as session:
            rows = session.execute(
                select(Cocktail.id, Cocktail.preference_embedding)
                .where(Cocktail.preference_embedding.isnot(None))
                .order_by(Cocktail.id)
            ).all()
            assert len(rows) == 602
            offline = [
                adapter.adapt(row[0], [float(value) for value in row[1]])
                for row in rows
            ]

            for codes in PARITY_SELECTIONS:
                assert set(codes) <= known_axes, codes
                query = build_user_query({code: 1.0 for code in codes})
                expected = score_preference_mips(query, offline, 5)

                session.execute(text("SET LOCAL enable_indexscan = off"))
                session.execute(text("SET LOCAL enable_bitmapscan = off"))
                actual = repository.nearest_by_preference(
                    session,
                    list(query.values),
                    5,
                )
                session.rollback()

                assert [row["id"] for row in actual] == [
                    match.cocktail_id for match in expected
                ], codes
                for row, match in zip(actual, expected, strict=True):
                    assert row["similarity"] == pytest.approx(
                        round(match.score, 4),
                        abs=2e-4,
                    ), (codes, row["id"])
    finally:
        engine.dispose()
