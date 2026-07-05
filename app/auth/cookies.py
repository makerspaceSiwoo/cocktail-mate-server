"""인증 쿠키 세팅/삭제 헬퍼.

- access_token: path="/" (전체 API).
- refresh_token: path="/auth" (일반 API 표면에서는 스코프 축소하되,
  /auth/refresh 와 /auth/logout 양쪽에 전송되어야 로그아웃 시 서버 폐기가 가능).

쿠키 플래그(secure/samesite)는 **요청 Origin 기준으로 동적 결정**한다
(`resolve_cookie_flags`). 크로스 도메인 배포(프론트 Vercel ↔ API DuckDNS/nip.io)에서
브라우저가 쿠키를 실어 보내려면 SameSite=None; Secure 가 필요하고, 로컬 http 개발에서는
Secure 를 붙일 수 없어 SameSite=Lax 로 떨어뜨려야 하기 때문.
"""
from __future__ import annotations

from urllib.parse import urlsplit

from fastapi import Request, Response

from app.core.config import get_settings

ACCESS_COOKIE = "access_token"
REFRESH_COOKIE = "refresh_token"
REFRESH_COOKIE_PATH = "/auth"

_LOCALHOST_HOSTS = {"localhost", "127.0.0.1"}


def _is_localhost_origin(origin: str) -> bool:
    """Origin 이 http://localhost(또는 127.0.0.1):<port> 형태인지."""
    parts = urlsplit(origin)
    return parts.scheme == "http" and parts.hostname in _LOCALHOST_HOSTS


def resolve_cookie_flags(request: Request | None) -> tuple[bool, str]:
    """요청 Origin 기준으로 (secure, samesite) 를 결정한다.

    - localhost origin(http, localhost/127.0.0.1, 임의 포트) → (False, "lax")
      로컬 http 개발: Secure 불가라 Lax 로.
    - 그 외 배포 cross-site origin → (True, "none")
      크로스 도메인 쿠키 전송을 위해 SameSite=None; Secure.
    - Origin 헤더 없음(same-origin 서버간 호출, 카카오 콜백 같은 top-level GET 리다이렉트)
      → 정적 fallback (settings.cookie_secure / settings.cookie_samesite).

    ⚠️ 불변식: SameSite=None 은 브라우저 규격상 Secure=true 를 요구한다.
    secure=False 로 떨어지는 유일한 분기(localhost)는 samesite="lax" 를 쓰므로
    "none + not secure" 조합은 절대 나오지 않는다.
    """
    settings = get_settings()
    origin = request.headers.get("origin") if request is not None else None
    if not origin:
        return settings.cookie_secure, settings.cookie_samesite
    if _is_localhost_origin(origin):
        return False, "lax"
    return True, "none"


def set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
    request: Request | None = None,
) -> None:
    settings = get_settings()
    secure, samesite = resolve_cookie_flags(request)
    # 불변식 방어: SameSite=None 인데 Secure 가 아니면 브라우저가 쿠키를 거부한다.
    assert not (samesite == "none" and not secure), "SameSite=None 은 Secure 필수"
    response.set_cookie(
        ACCESS_COOKIE,
        access_token,
        httponly=True,
        secure=secure,
        samesite=samesite,
        max_age=settings.access_token_expire_minutes * 60,
        path="/",
    )
    response.set_cookie(
        REFRESH_COOKIE,
        refresh_token,
        httponly=True,
        secure=secure,
        samesite=samesite,
        max_age=settings.refresh_token_expire_days * 86400,
        path=REFRESH_COOKIE_PATH,
    )


def clear_auth_cookies(response: Response, request: Request | None = None) -> None:
    # 삭제 시에도 세팅 때와 동일한 flag 로 delete 해야 브라우저가 같은 쿠키로 인식해 지운다.
    secure, samesite = resolve_cookie_flags(request)
    response.delete_cookie(ACCESS_COOKIE, path="/", secure=secure, samesite=samesite)
    response.delete_cookie(
        REFRESH_COOKIE, path=REFRESH_COOKIE_PATH, secure=secure, samesite=samesite
    )
