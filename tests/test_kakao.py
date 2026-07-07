"""카카오 소셜 로그인 + 세션 테스트 (httpx 는 mock).

실제 카카오 호출 금지 — provider 의 exchange_code/fetch_profile 를 mock 한다.
카카오 콜백 → service upsert → 세션(access/refresh) 발급 흐름을 검증한다.
"""
from __future__ import annotations

import asyncio


from cocktail_mate_db.models import User
from app.auth.providers.kakao import KakaoProvider


# ── 콜백: 가입/로그인 ────────────────────────────────────────
def test_kakao_new_user_signup(kakao_callback, db):
    resp = kakao_callback(provider_id="12345", nickname="카카오유저", email="k@example.com")
    assert resp.status_code == 302, resp.text
    assert "access_token" in resp.cookies

    user = db.query(User).filter_by(provider="kakao", provider_id="12345").one()
    assert user.email == "k@example.com"
    assert user.nickname == "카카오유저"
    assert user.is_active is True


def test_kakao_signup_without_email(kakao_callback, db):
    """이메일 없는 카카오 프로필도 가입된다 (email NULL 저장)."""
    resp = kakao_callback(provider_id="55555", nickname="이메일없음", email=None)
    assert resp.status_code == 302, resp.text

    user = db.query(User).filter_by(provider="kakao", provider_id="55555").one()
    assert user.email is None
    assert user.nickname == "이메일없음"


def test_kakao_signup_without_nickname_gets_default(kakao_callback, db):
    """닉네임도 이메일도 없으면 provider 기반 기본 닉네임을 부여한다."""
    resp = kakao_callback(provider_id="98765432100", nickname=None, email=None)
    assert resp.status_code == 302, resp.text

    user = db.query(User).filter_by(provider="kakao", provider_id="98765432100").one()
    assert user.nickname  # 비어있지 않음
    assert user.nickname.startswith("kakao")


def test_kakao_duplicate_nickname_allowed(kakao_callback, db):
    """닉네임 중복 허용 — 서로 다른 provider_id 면 같은 닉네임으로 가입된다."""
    kakao_callback(provider_id="111", nickname="같은닉네임")
    kakao_callback(provider_id="222", nickname="같은닉네임")
    users = db.query(User).filter_by(nickname="같은닉네임").all()
    assert len(users) == 2


def test_kakao_existing_user_login(kakao_callback, db):
    """같은 (provider, provider_id) 재로그인 시 새 유저가 생기지 않는다."""
    db.add(
        User(
            email=None,
            nickname="기존카카오",
            provider="kakao",
            provider_id="777",
            is_active=True,
        )
    )
    db.commit()
    before = db.query(User).count()

    resp = kakao_callback(provider_id="777", nickname="기존카카오")
    assert resp.status_code == 302
    assert "access_token" in resp.cookies
    assert db.query(User).count() == before


# ── CSRF state / 알 수 없는 provider ─────────────────────────
def test_kakao_state_mismatch_rejected(client):
    """쿠키의 state 와 쿼리 state 불일치 → CSRF 거부(400)."""
    client.get("/auth/kakao/login", follow_redirects=False)
    client.cookies.set("oauth_state", "real-state", path="/auth/kakao/callback")
    resp = client.get(
        "/auth/kakao/callback",
        params={"code": "c", "state": "attacker-state"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "state" in resp.json()["detail"]


def test_unknown_provider_404(client):
    assert client.get("/auth/google/login", follow_redirects=False).status_code == 404


# ── 세션: /me, /logout, /refresh ─────────────────────────────
def test_me_after_login(kakao_callback, client):
    kakao_callback(provider_id="333", nickname="미유저")
    resp = client.get("/auth/me")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provider"] == "kakao"
    assert body["nickname"] == "미유저"


def test_logout_then_refresh_rejected(kakao_callback, client):
    kakao_callback(provider_id="444", nickname="로그아웃유저")
    assert client.post("/auth/logout").status_code == 200
    # 로그아웃으로 refresh 행이 삭제됨 → refresh 거부.
    assert client.post("/auth/refresh").status_code == 401


def test_refresh_rotation_invalidates_old_token(kakao_callback, client):
    kakao_callback(provider_id="666", nickname="로테이션유저")
    old_refresh = client.cookies.get("refresh_token")
    assert old_refresh

    r1 = client.post("/auth/refresh")
    assert r1.status_code == 200, r1.text
    new_refresh = client.cookies.get("refresh_token")
    assert new_refresh and new_refresh != old_refresh  # 토큰 회전됨

    # 기존(회전 전) refresh 토큰으로 재시도하면 거부돼야 한다.
    client.cookies.set("refresh_token", old_refresh, path="/auth")
    assert client.post("/auth/refresh").status_code == 401


# ── provider 파싱 ────────────────────────────────────────────
def test_kakao_profile_parsing_without_email(monkeypatch):
    """KakaoProvider.fetch_profile: 이메일 scope 없이 닉네임/프로필사진만 파싱."""

    class _FakeResp:
        status_code = 200

        def json(self):
            return {
                "id": 999,
                "kakao_account": {
                    "profile": {
                        "nickname": "홍길동",
                        "profile_image_url": "http://img/k.jpg",
                    }
                },
            }

    class _FakeClient:
        async def get(self, *a, **k):
            return _FakeResp()

    provider = KakaoProvider()
    profile = asyncio.run(provider.fetch_profile("tok", _FakeClient()))
    assert profile.provider_id == "999"
    assert profile.nickname == "홍길동"
    assert profile.email is None


def test_kakao_authorize_url_has_no_email_scope():
    """authorize URL scope 에 account_email 이 없어야 한다(개인 앱 동작)."""
    url = KakaoProvider().authorize_url("state123")
    assert "account_email" not in url
    assert "profile_nickname" in url
