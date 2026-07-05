"""인증 라우터 (prefix /auth).

- 이메일 매직링크 가입: request-verification → verification-status(폴링) → verify → signup
- 로그인/refresh/logout/me
- 카카오 소셜 로그인 (provider 확장 가능: /auth/{provider}/... )

rate limiting 은 slowapi 데코레이터로 지정 (IP 기준 + 일부 엔드포인트는 이메일 기준 병행).
"""
from __future__ import annotations

import logging
import secrets

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from app.auth.cookies import (
    REFRESH_COOKIE,
    clear_auth_cookies,
    set_auth_cookies,
)
from app.auth.dependencies import CurrentUser
from app.auth.providers import EmailConsentRequired, SocialAuthError, get_provider
from app.auth.schemas import (
    LoginRequest,
    MessageResponse,
    RequestVerificationRequest,
    RequestVerificationResponse,
    SignupRequest,
    UserResponse,
    VerificationStatusResponse,
    VerifyEmailRequest,
    VerifyEmailResponse,
)
from app.auth.service import AuthService
from app.core.config import get_settings
from app.core.database import get_db
from app.core.rate_limit import limiter

logger = logging.getLogger("app.auth.router")

router = APIRouter(prefix="/auth", tags=["auth"])
service = AuthService()

# 카카오 콜백 CSRF 방어용 임시 state 쿠키.
_STATE_COOKIE = "oauth_state"


def _email_from_json_key(request: Request) -> str:
    """이메일 기준 rate limit key — 캐시된 본문의 email (없으면 IP)."""
    email = getattr(request.state, "rl_email", None)
    return f"email:{email}" if email else get_remote_address(request)


def _user_response(user) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        nickname=user.nickname,
        provider=user.provider,
        profile_image_url=user.profile_image_url,
    )


# ── 이메일 인증 ─────────────────────────────────────────────
@router.post(
    "/email/request-verification", response_model=RequestVerificationResponse
)
@limiter.limit("5/hour;10/hour", key_func=_email_from_json_key)  # 동일 이메일 5회/시간
@limiter.limit("1/minute", key_func=_email_from_json_key)  # 동일 이메일 1회/분
@limiter.limit("10/hour")  # IP 10회/시간
def request_verification(
    request: Request,
    payload: RequestVerificationRequest,
    db: Session = Depends(get_db),
):
    # 이메일 기준 limit key 를 위해 state 에 저장 (limiter key_func 가 참조).
    request.state.rl_email = str(payload.email)
    request_id = service.request_verification(db, str(payload.email))
    return RequestVerificationResponse(request_id=request_id)


@router.get(
    "/email/verification-status", response_model=VerificationStatusResponse
)
@limiter.limit("30/minute")  # IP 30회/분
def verification_status(
    request: Request,
    request_id: str,
    db: Session = Depends(get_db),
):
    result = service.verification_status(db, request_id)
    return VerificationStatusResponse(**result)


@router.post("/email/verify", response_model=VerifyEmailResponse)
@limiter.limit("10/minute")  # IP 10회/분
def verify_email(
    request: Request,
    payload: VerifyEmailRequest,
    db: Session = Depends(get_db),
):
    # POST 인 이유: 메일 스캐너의 GET 프리페치가 링크를 소진하지 못하게.
    service.verify_email(db, payload.token)
    return VerifyEmailResponse(message="이메일 인증이 완료되었습니다.")


# ── 가입/로그인 ─────────────────────────────────────────────
@router.post("/signup", response_model=UserResponse)
@limiter.limit("5/hour")  # IP 5회/시간
def signup(
    request: Request,
    payload: SignupRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    user, access_token, refresh_token = service.signup(db, payload)
    set_auth_cookies(response, access_token, refresh_token)
    return _user_response(user)


@router.post("/login", response_model=UserResponse)
@limiter.limit("5/5minute", key_func=_email_from_json_key)  # 동일 이메일 5회/5분
@limiter.limit("10/minute")  # IP 10회/분
def login(
    request: Request,
    payload: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    request.state.rl_email = str(payload.email)
    user, access_token, refresh_token = service.login(
        db, str(payload.email), payload.password
    )
    set_auth_cookies(response, access_token, refresh_token)
    return _user_response(user)


@router.post("/refresh", response_model=MessageResponse)
def refresh(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    raw_refresh = request.cookies.get(REFRESH_COOKIE)
    access_token, refresh_token = service.refresh(db, raw_refresh or "")
    set_auth_cookies(response, access_token, refresh_token)
    return MessageResponse(message="토큰이 갱신되었습니다.")


@router.post("/logout", response_model=MessageResponse)
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    raw_refresh = request.cookies.get(REFRESH_COOKIE)
    service.logout(db, raw_refresh)
    clear_auth_cookies(response)
    return MessageResponse(message="로그아웃 되었습니다.")


@router.get("/me", response_model=UserResponse)
def me(current_user: CurrentUser):
    return _user_response(current_user)


# ── 소셜 로그인 (provider 일반화; 현재 kakao 만 등록) ────────
@router.get("/{provider}/login")
def social_login_redirect(provider: str, response: Response):
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
        except EmailConsentRequired as exc:
            # 이메일 미동의 → 가입 거부 + 연결 해제.
            try:
                await social.unlink(exc.access_token, client)
            except Exception:  # noqa: BLE001 - unlink 실패는 로깅만
                logger.exception("소셜 unlink 실패 provider=%s", provider)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이메일 제공 동의가 필요합니다.",
            ) from None
        except SocialAuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from None

    user, jwt_access, jwt_refresh = service.social_login(db, profile)

    redirect = RedirectResponse(
        url=settings.frontend_url, status_code=status.HTTP_302_FOUND
    )
    set_auth_cookies(redirect, jwt_access, jwt_refresh)
    redirect.delete_cookie(_STATE_COOKIE, path=f"/auth/{provider}/callback")
    return redirect
