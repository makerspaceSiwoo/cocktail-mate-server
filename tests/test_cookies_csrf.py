"""동적 쿠키 플래그(Origin 분기) + CSRF Origin allowlist 미들웨어 테스트."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth.cookies import resolve_cookie_flags
from app.core.config import Settings
from app.core.csrf import CSRFOriginMiddleware
from tests.test_auth_email import _create_local_user


def _set_cookie_for(resp, name: str) -> str:
    """응답 Set-Cookie 헤더들 중 name 을 세팅하는 항목 반환."""
    for header in resp.headers.get_list("set-cookie"):
        if header.startswith(f"{name}="):
            return header
    raise AssertionError(f"{name} Set-Cookie 없음: {resp.headers.get_list('set-cookie')}")


# ── Change 1: 동적 쿠키 플래그 ──────────────────────────────
def test_localhost_origin_cookie_lax_no_secure(client, db):
    """localhost Origin → SameSite=lax, Secure 없음."""
    _create_local_user(db)
    resp = client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "abcd1234!"},
        headers={"Origin": "http://localhost:3000"},
    )
    assert resp.status_code == 200, resp.text
    access = _set_cookie_for(resp, "access_token")
    assert "SameSite=lax" in access, access
    assert "Secure" not in access, access


def test_deployed_origin_cookie_none_secure(client, db):
    """배포(non-localhost https) Origin → SameSite=None; Secure."""
    # 배포 origin 이 CSRF allowlist 를 통과하도록 실행 중 미들웨어 allowlist 에 주입.
    # (미들웨어는 app 생성 시점 settings 를 캡처하고, 테스트 앱은 local 설정으로 빌드됨.)
    from app.main import app as main_app

    _inject_allowed_origin(main_app, "https://cocktail-mate.vercel.app")

    _create_local_user(db)
    resp = client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "abcd1234!"},
        headers={"Origin": "https://cocktail-mate.vercel.app"},
    )
    assert resp.status_code == 200, resp.text
    access = _set_cookie_for(resp, "access_token")
    assert "samesite=none" in access.lower(), access
    assert "Secure" in access, access


def _inject_allowed_origin(app: FastAPI, origin: str) -> None:
    """실행 중 앱의 CSRF 미들웨어 인스턴스 allowlist 에 origin 을 추가한다.

    미들웨어는 __init__ 에서 allowlist 를 확정하므로, 이미 빌드된 스택의
    인스턴스를 찾아 _allowed 에 넣는다. TestClient(app) 진입 시 스택이 빌드된다.
    """
    stack = getattr(app, "middleware_stack", None)
    target = stack
    while target is not None:
        if isinstance(target, CSRFOriginMiddleware):
            target._allowed.add(origin)
            return
        target = getattr(target, "app", None)


def test_no_origin_falls_back_to_static(monkeypatch):
    """Origin 헤더 없음 → 정적 settings fallback."""
    from app.core import config as config_mod

    class _FakeSettings:
        cookie_secure = True
        cookie_samesite = "lax"

    monkeypatch.setattr(config_mod, "get_settings", lambda: _FakeSettings())
    # cookies 모듈은 config.get_settings 를 직접 참조.
    monkeypatch.setattr("app.auth.cookies.get_settings", lambda: _FakeSettings())

    secure, samesite = resolve_cookie_flags(None)
    assert (secure, samesite) == (True, "lax")


def test_resolve_flags_localhost_and_deployed(monkeypatch):
    """resolve_cookie_flags 순수 로직 검증."""
    class _Req:
        def __init__(self, origin):
            self.headers = {"origin": origin} if origin else {}

    class _FakeSettings:
        cookie_secure = True
        cookie_samesite = "lax"

    monkeypatch.setattr("app.auth.cookies.get_settings", lambda: _FakeSettings())

    assert resolve_cookie_flags(_Req("http://localhost:5173")) == (False, "lax")
    assert resolve_cookie_flags(_Req("http://127.0.0.1:3000")) == (False, "lax")
    assert resolve_cookie_flags(_Req("https://foo.vercel.app")) == (True, "none")
    assert resolve_cookie_flags(_Req(None)) == (True, "lax")


# ── Change 2: CSRF Origin allowlist 미들웨어 ────────────────
def _build_app(settings: Settings) -> FastAPI:
    """CSRF 미들웨어만 얹은 최소 앱 (unsafe/safe 라우트 포함)."""
    app = FastAPI()
    app.add_middleware(CSRFOriginMiddleware, settings=settings)

    @app.post("/echo")
    def echo():
        return {"ok": True}

    @app.get("/echo")
    def echo_get():
        return {"ok": True}

    return app


_ALLOWED = "https://cocktail-mate.vercel.app"


def _prod_settings() -> Settings:
    return Settings(app_env="production", cors_origins=_ALLOWED)


def _local_settings() -> Settings:
    return Settings(app_env="local", cors_origins=_ALLOWED)


def test_csrf_prod_allowed_origin_passes():
    """production: 허용 Origin 의 POST → 통과."""
    with TestClient(_build_app(_prod_settings())) as c:
        r = c.post("/echo", headers={"Origin": _ALLOWED})
        assert r.status_code == 200, r.text


def test_csrf_prod_disallowed_origin_403():
    """production: 허용되지 않은 Origin 의 POST → 403."""
    with TestClient(_build_app(_prod_settings())) as c:
        r = c.post("/echo", headers={"Origin": "https://evil.example.com"})
        assert r.status_code == 403, r.text


def test_csrf_prod_absent_origin_403():
    """production: Origin 없는 POST → 403."""
    with TestClient(_build_app(_prod_settings())) as c:
        r = c.post("/echo")
        assert r.status_code == 403, r.text


def test_csrf_prod_safe_method_passes_without_origin():
    """production: safe 메서드(GET, 카카오 콜백류)는 Origin 없어도 통과."""
    with TestClient(_build_app(_prod_settings())) as c:
        r = c.get("/echo")
        assert r.status_code == 200, r.text


def test_csrf_local_absent_origin_passes():
    """local/dev: Origin 없는 POST(curl/테스트/툴) → 통과."""
    with TestClient(_build_app(_local_settings())) as c:
        r = c.post("/echo")
        assert r.status_code == 200, r.text


def test_csrf_local_localhost_origin_passes():
    """local/dev: localhost Origin POST → 통과 (regex 허용)."""
    with TestClient(_build_app(_local_settings())) as c:
        r = c.post("/echo", headers={"Origin": "http://localhost:5173"})
        assert r.status_code == 200, r.text


def test_csrf_local_disallowed_origin_403():
    """local/dev: 허용 목록에도 없고 localhost 도 아닌 Origin → 403."""
    with TestClient(_build_app(_local_settings())) as c:
        r = c.post("/echo", headers={"Origin": "https://evil.example.com"})
        assert r.status_code == 403, r.text
