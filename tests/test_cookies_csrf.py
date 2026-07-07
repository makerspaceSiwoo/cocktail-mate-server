"""동적 쿠키 플래그(Origin 분기) + CSRF Origin allowlist 미들웨어 테스트."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth.cookies import resolve_cookie_flags
from app.core.config import Settings
from app.core.csrf import CSRFOriginMiddleware


def _set_cookie_for(resp, name: str) -> str:
    """응답 Set-Cookie 헤더들 중 name 을 세팅하는 항목 반환."""
    for header in resp.headers.get_list("set-cookie"):
        if header.startswith(f"{name}="):
            return header
    raise AssertionError(f"{name} Set-Cookie 없음: {resp.headers.get_list('set-cookie')}")


# ── Change 1: 동적 쿠키 플래그 (카카오 콜백 GET 흐름으로 검증) ──
def test_localhost_origin_cookie_lax_no_secure(kakao_callback):
    """localhost Origin → SameSite=lax, Secure 없음."""
    resp = kakao_callback(provider_id="cookie1", origin="http://localhost:3000")
    assert resp.status_code == 302, resp.text
    access = _set_cookie_for(resp, "access_token")
    assert "SameSite=lax" in access, access
    assert "Secure" not in access, access


def test_deployed_origin_cookie_none_secure(kakao_callback):
    """배포(non-localhost https) Origin → SameSite=None; Secure.

    콜백은 GET(safe method)이라 CSRF Origin allowlist 대상이 아니므로 그대로 통과한다.
    """
    resp = kakao_callback(
        provider_id="cookie2", origin="https://cocktail-mate.vercel.app"
    )
    assert resp.status_code == 302, resp.text
    access = _set_cookie_for(resp, "access_token")
    assert "samesite=none" in access.lower(), access
    assert "Secure" in access, access


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
