"""pytest 공용 픽스처.

- 로컬 docker PostgreSQL 의 별도 테스트 DB(cocktail_mate_test)를 사용한다.
  (원격 OCI DB 접속 금지 — 아래 TEST_DATABASE_URL 로 강제 override)
- 테이블은 cocktail_mate_db Base.metadata 로 매 세션 생성/드롭.
- 메일 발송·카카오 httpx 는 테스트에서 mock 한다.
"""
from __future__ import annotations

import os

# ── 테스트 환경변수 강제(앱/설정 import 전에 반드시 세팅) ──
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg2://cm_admin:PcQntRJSFTvXF4rVydRSIxJsJzQOVC+T"
    "@localhost:5432/cocktail_mate_test",
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ.setdefault("SECRET_KEY", "test-secret-key-0123456789abcdef0123456789")
os.environ.setdefault("COOKIE_SECURE", "false")
os.environ.setdefault("FRONTEND_URL", "http://localhost:3000")
os.environ.setdefault("KAKAO_CLIENT_ID", "test-kakao-id")
os.environ.setdefault("KAKAO_REDIRECT_URI", "http://localhost:8000/auth/kakao/callback")
# rate limit 은 fixture 에서 app.state.limiter.enabled 로 직접 토글한다.

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from cocktail_mate_db.base import Base  # noqa: E402
import cocktail_mate_db.models  # noqa: E402,F401 — 모든 모델을 metadata 에 등록

test_engine = create_engine(TEST_DATABASE_URL, future=True)
TestingSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)


@pytest.fixture(scope="session", autouse=True)
def _create_schema():
    """세션 시작 시 스키마 생성, 종료 시 드롭."""
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(autouse=True)
def _clean_tables():
    """각 테스트 전에 데이터 정리 (스키마는 유지)."""
    with test_engine.begin() as conn:
        conn.exec_driver_sql(
            "TRUNCATE TABLE likes, refresh_tokens, "
            "cocktail_ingredients, cocktails, ingredients, users "
            "RESTART IDENTITY CASCADE"
        )
    yield


@pytest.fixture
def db():
    """테스트에서 직접 DB 를 조작할 때 쓰는 세션."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    """get_db 를 테스트 세션으로 override 한 TestClient."""
    from app.core.database import get_db
    from app.main import app

    def _override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_get_db
    # rate limit 은 기본적으로 끔 (전용 테스트에서 client_rl 로 활성).
    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_rl():
    """rate limit 을 켠 상태의 TestClient (429 테스트 전용)."""
    from app.core.database import get_db
    from app.main import app

    def _override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_get_db
    app.state.limiter.enabled = True
    app.state.limiter.reset()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    app.state.limiter.enabled = False


@pytest.fixture
def kakao_callback(client, monkeypatch):
    """카카오 콜백을 mock provider 로 수행하는 헬퍼.

    provider 의 exchange_code/fetch_profile 를 monkeypatch 로 대체해 실제 카카오 호출 없이
    라우터+service upsert 를 태운다. profile 필드/Origin 헤더를 지정해 다양한 시나리오를 만든다.
    콜백 응답(302 리다이렉트, Set-Cookie 포함)을 반환한다.
    """
    from app.auth.providers import SocialProfile
    from app.auth.providers.kakao import KakaoProvider

    def _run(
        *,
        provider_id: str = "12345",
        nickname: str | None = "카카오유저",
        email: str | None = None,
        profile_image_url: str | None = None,
        origin: str | None = None,
    ):
        resp = client.get("/auth/kakao/login", follow_redirects=False)
        assert resp.status_code == 302
        state = client.cookies.get("oauth_state")
        assert state

        async def _exchange(self, code, http):
            return "kakao-access-token"

        async def _profile(self, token, http):
            return SocialProfile(
                provider="kakao",
                provider_id=provider_id,
                email=email,
                nickname=nickname,
                profile_image_url=profile_image_url,
            )

        monkeypatch.setattr(KakaoProvider, "exchange_code", _exchange)
        monkeypatch.setattr(KakaoProvider, "fetch_profile", _profile)

        headers = {"Origin": origin} if origin else {}
        return client.get(
            "/auth/kakao/callback",
            params={"code": "authcode", "state": state},
            headers=headers,
            follow_redirects=False,
        )

    return _run


@pytest.fixture
def seed_cocktail(db):
    """테스트용 칵테일 1개 시드 후 id 반환."""
    from cocktail_mate_db.models import Cocktail

    def _seed(name: str = "마가리타", base_tag: str = "tequila") -> int:
        c = Cocktail(name=name, base_tag=base_tag, image_url="http://img/1.jpg")
        db.add(c)
        db.commit()
        db.refresh(c)
        return c.id

    return _seed
