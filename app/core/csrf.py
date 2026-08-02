"""CSRF 방어 미들웨어 (Origin allowlist).

배포 환경은 SameSite=None 쿠키를 쓰므로(크로스 도메인) 브라우저가 다른 사이트의
cross-site POST 에도 피해자의 쿠키를 실어 보낸다. 토큰 기반 CSRF 대신,
unsafe 메서드(POST/PUT/PATCH/DELETE)에 대해 요청 Origin 을 CORS 허용 목록과 대조해
위조 요청을 차단하는 경량 방어를 둔다.

허용 판단:
- safe 메서드(GET/HEAD/OPTIONS) → 항상 통과 (카카오 콜백 GET 포함).
- unsafe 메서드:
    - Origin 헤더가 허용 목록에 있으면 통과.
    - Origin 이 허용 목록에 없으면 403.
    - Origin 헤더가 아예 없으면:
        - production → 403 (브라우저 요청은 cross-site POST 시 Origin 을 항상 보냄).
        - local/dev → 통과 (curl/테스트/툴 호환).

허용 목록 = CORS 가 허용하는 것과 동일:
  settings.cors_origin_list (production 등록 origin) + (production 이 아니면) localhost regex.
"""

from __future__ import annotations

import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import Settings

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
_LOCALHOST_ORIGIN_RE = re.compile(r"https?://(localhost|127\.0\.0\.1)(:\d+)?$")


class CSRFOriginMiddleware(BaseHTTPMiddleware):
    """unsafe 메서드에 대해 Origin allowlist 로 CSRF 를 방어한다."""

    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self._settings = settings
        self._allowed = set(settings.cors_origin_list)
        self._allow_localhost = not settings.is_production
        # CORSMiddleware 와 동일한 서브도메인 와일드카드 규칙을 공유(설정 시).
        regex = settings.cors_origin_regex
        self._origin_re = re.compile(regex) if regex else None

    def _origin_allowed(self, origin: str) -> bool:
        if origin in self._allowed:
            return True
        # CORS 와 동일하게 fullmatch (부분 일치로 인한 우회 방지).
        if self._origin_re is not None and self._origin_re.fullmatch(origin):
            return True
        if self._allow_localhost and _LOCALHOST_ORIGIN_RE.match(origin):
            return True
        return False

    async def dispatch(self, request: Request, call_next):
        if request.method not in _SAFE_METHODS:
            origin = request.headers.get("origin")
            if origin is None:
                # Origin 없는 unsafe 요청: production 은 거부, 로컬/개발은 허용.
                if self._settings.is_production:
                    return _forbidden("Origin 헤더가 필요합니다.")
            elif not self._origin_allowed(origin):
                return _forbidden("허용되지 않은 Origin 입니다.")
        return await call_next(request)


def _forbidden(detail: str) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": detail})
