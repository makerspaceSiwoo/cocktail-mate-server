"""카카오 소셜 로그인 콜백 테스트 (httpx 는 mock).

provider 의 exchange_code/fetch_profile/unlink 를 mock 해 라우터+service upsert 를 검증한다.
(실제 카카오 호출 금지 — 키 없음)
"""
from __future__ import annotations

import pytest

from cocktail_mate_db.models import User
from app.auth.providers import EmailConsentRequired, SocialProfile
from app.auth.providers.kakao import KakaoProvider


def _prime_state(client):
    """login 리다이렉트로 state 쿠키를 세팅하고 그 state 값을 반환."""
    resp = client.get("/auth/kakao/login", follow_redirects=False)
    assert resp.status_code == 302
    # authorize URL 로 리다이렉트 + oauth_state 쿠키 세팅
    assert "kauth.kakao.com" in resp.headers["location"]
    state = client.cookies.get("oauth_state")
    assert state
    return state


def test_kakao_new_user_signup(client, db, monkeypatch):
    state = _prime_state(client)

    async def _exchange(self, code, http):
        return "kakao-access-token"

    async def _profile(self, token, http):
        return SocialProfile(
            provider="kakao",
            provider_id="12345",
            email="kakao@example.com",
            nickname="카카오유저",
            profile_image_url="http://img/k.jpg",
        )

    monkeypatch.setattr(KakaoProvider, "exchange_code", _exchange)
    monkeypatch.setattr(KakaoProvider, "fetch_profile", _profile)

    resp = client.get(
        "/auth/kakao/callback",
        params={"code": "authcode", "state": state},
        cookies={"oauth_state": state},
        follow_redirects=False,
    )
    assert resp.status_code == 302, resp.text
    assert "access_token" in resp.cookies

    user = db.query(User).filter_by(provider="kakao", provider_id="12345").one()
    assert user.email == "kakao@example.com"
    assert user.password_hash is None
    assert user.is_active is True


def test_kakao_existing_user_login(client, db, monkeypatch):
    # 기존 카카오 유저
    db.add(
        User(
            email="kakao@example.com",
            password_hash=None,
            nickname="기존카카오",
            provider="kakao",
            provider_id="777",
            is_active=True,
        )
    )
    db.commit()
    before = db.query(User).count()

    state = _prime_state(client)

    async def _exchange(self, code, http):
        return "tok"

    async def _profile(self, token, http):
        return SocialProfile(
            provider="kakao", provider_id="777", email="kakao@example.com",
            nickname="기존카카오",
        )

    monkeypatch.setattr(KakaoProvider, "exchange_code", _exchange)
    monkeypatch.setattr(KakaoProvider, "fetch_profile", _profile)

    resp = client.get(
        "/auth/kakao/callback",
        params={"code": "c", "state": state},
        cookies={"oauth_state": state},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert "access_token" in resp.cookies
    # 새 유저가 생기지 않음
    assert db.query(User).count() == before


def test_kakao_email_consent_required(client, db, monkeypatch):
    state = _prime_state(client)
    unlinked = {}

    async def _exchange(self, code, http):
        return "consent-token"

    async def _profile(self, token, http):
        raise EmailConsentRequired("consent-token")

    async def _unlink(self, token, http):
        unlinked["token"] = token

    monkeypatch.setattr(KakaoProvider, "exchange_code", _exchange)
    monkeypatch.setattr(KakaoProvider, "fetch_profile", _profile)
    monkeypatch.setattr(KakaoProvider, "unlink", _unlink)

    resp = client.get(
        "/auth/kakao/callback",
        params={"code": "c", "state": state},
        cookies={"oauth_state": state},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "이메일 제공 동의" in resp.json()["detail"]
    # 연결 해제(unlink) 호출됨
    assert unlinked.get("token") == "consent-token"
    # 유저 미생성
    assert db.query(User).count() == 0


def test_kakao_state_mismatch_rejected(client, monkeypatch):
    _prime_state(client)
    resp = client.get(
        "/auth/kakao/callback",
        params={"code": "c", "state": "attacker-state"},
        cookies={"oauth_state": "real-state"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "state" in resp.json()["detail"]


def test_unknown_provider_404(client):
    assert client.get("/auth/google/login", follow_redirects=False).status_code == 404


def test_kakao_profile_parsing_email_consent(monkeypatch):
    """KakaoProvider.fetch_profile: 이메일 미동의 응답 파싱 → EmailConsentRequired."""
    import asyncio

    class _FakeResp:
        status_code = 200

        def json(self):
            return {
                "id": 999,
                "kakao_account": {"email_needs_agreement": True},
            }

    class _FakeClient:
        async def get(self, *a, **k):
            return _FakeResp()

    provider = KakaoProvider()
    with pytest.raises(EmailConsentRequired):
        asyncio.run(provider.fetch_profile("tok", _FakeClient()))
