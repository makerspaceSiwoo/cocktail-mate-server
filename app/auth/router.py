"""인증 라우터 (prefix /auth).

소셜 로그인 전용:
- 소셜 로그인/가입: GET /auth/{provider}/login → provider 인가 → GET /auth/{provider}/callback
- 세션: /auth/refresh(rotation), /auth/logout, /auth/my-info
"""

from __future__ import annotations

import secrets

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.auth.cookies import (
    REFRESH_COOKIE,
    clear_auth_cookies,
    set_auth_cookies,
)
from app.auth.dependencies import CurrentUser
from app.auth.providers import SocialAuthError, get_provider
from app.auth.schemas import (
  MessageResponse, 
  NicknameChangeRequest,
  UserResponse,
)
from app.auth.service import AuthService
from app.core.config import get_settings
from app.core.database import get_db

router = APIRouter(prefix="/auth", tags=["auth"])
my_router = APIRouter(prefix="/my", tags=["my"])
service = AuthService()

# 카카오 콜백 CSRF 방어용 임시 state 쿠키.
_STATE_COOKIE = "oauth_state"


def _user_response(user) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        nickname=user.nickname,
        provider=user.provider,
        profile_image_url=user.profile_image_url,
    )


# ── 세션 (refresh / logout / my-info) ───────────────────────
@router.post("/refresh", response_model=MessageResponse)
def refresh(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    raw_refresh = request.cookies.get(REFRESH_COOKIE)
    access_token, refresh_token = service.refresh(db, raw_refresh or "")
    set_auth_cookies(response, access_token, refresh_token, request)
    return MessageResponse(message="토큰이 갱신되었습니다.")


@router.post("/logout", response_model=MessageResponse)
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    raw_refresh = request.cookies.get(REFRESH_COOKIE)
    service.logout(db, raw_refresh)
    clear_auth_cookies(response, request)
    return MessageResponse(message="로그아웃 되었습니다.")


@my_router.get("/info", response_model=UserResponse)
def my_info(current_user: CurrentUser):
    return _user_response(current_user)


@my_router.patch("/info", response_model=UserResponse)
def change_nickname(
    payload: NicknameChangeRequest,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    updated_user = service.change_nickname(
        db,
        current_user,
        payload.nickname,
    )
    return _user_response(updated_user)


# ── 소셜 로그인 (provider 일반화; 현재 kakao 만 등록) ────────
@router.get("/{provider}/login")
def social_login_redirect(provider: str, response: Response):
    """소셜 로그인 시작 — provider 인가 페이지로 302 리다이렉트.

    ⚠️ Swagger 의 **Execute 버튼은 쓰지 마세요.** fetch(XHR)라 302 를 따라 provider 로
    cross-origin 요청이 가 CORS 로 막힙니다. 아래 링크를 **클릭**하면 브라우저가 이동해
    카카오 로그인 페이지가 정상적으로 뜹니다(로그인 후 이 문서로 돌아옵니다):

    👉 [카카오 로그인 시작하기](/auth/kakao/login)
    """
    social = get_provider(provider)
    if social is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="지원하지 않는 소셜 로그인입니다.",
        )
    state = secrets.token_urlsafe(32)
    settings = get_settings()
    redirect = RedirectResponse(
        url=social.authorize_url(state),
        status_code=status.HTTP_302_FOUND,
    )
    # CSRF 방어: state 를 임시 HttpOnly 쿠키에 저장해 콜백에서 대조.
    redirect.set_cookie(
        _STATE_COOKIE,
        state,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=600,
        path=f"/auth/{provider}/callback",
    )
    return redirect


@router.get("/{provider}/callback")
async def social_login_callback(
    provider: str,
    request: Request,
    code: str | None = None,
    state: str | None = None,
    db: Session = Depends(get_db),
):
    social = get_provider(provider)
    if social is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="지원하지 않는 소셜 로그인입니다.",
        )

    # state 검증 (CSRF).
    cookie_state = request.cookies.get(_STATE_COOKIE)
    if not state or not cookie_state or not secrets.compare_digest(state, cookie_state):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="유효하지 않은 요청입니다. (state 불일치)",
        )
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="인가 코드가 없습니다.",
        )

    settings = get_settings()
    async with httpx.AsyncClient() as client:
        try:
            access_token = await social.exchange_code(code, client)
            profile = await social.fetch_profile(access_token, client)
        except SocialAuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from None

    user, jwt_access, jwt_refresh = service.social_login(db, profile)

    redirect = RedirectResponse(
        url=settings.frontend_url, status_code=status.HTTP_302_FOUND
    )
    # 카카오 콜백은 top-level GET 리다이렉트라 Origin 헤더가 없다 →
    # resolve_cookie_flags 가 정적 fallback(settings) 을 쓴다. request 를 넘겨도 무방.
    set_auth_cookies(redirect, jwt_access, jwt_refresh, request)
    redirect.delete_cookie(_STATE_COOKIE, path=f"/auth/{provider}/callback")
    return redirect
