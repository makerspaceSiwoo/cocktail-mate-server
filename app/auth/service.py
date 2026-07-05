"""인증 비즈니스 로직.

핵심 보안 원칙:
- "인증된 메일"의 증명은 프론트 주장이 아니라 서버 DB의 verified_at + 추측 불가 request_id 로 서버가 직접 확인.
- 로그인 실패는 계정 유무와 무관하게 항상 동일한 401 (타이밍 공격 방어: 유저 없어도 더미 해시 검증).
- 토큰 원본은 저장 안 함 — 항상 sha256 해시만 저장/대조.

세션 커밋은 이 계층에서 관리한다 (repository 는 flush 까지만).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from cocktail_mate_db.models import EmailVerification, RefreshToken, User

from app.auth.mail import send_password_reset_email, send_verification_email
from app.auth.providers import SocialProfile
from app.auth.repository import AuthRepository
from app.auth.schemas import (
    PasswordChangeRequest,
    PasswordResetRequest,
    SignupRequest,
)
from app.core.config import get_settings
from app.core.security import (
    JWTError,
    create_access_token,
    create_password_reset_token,
    create_refresh_token,
    decode_password_reset_token,
    extract_reset_token_subject,
    generate_random_token,
    hash_password,
    hash_token,
    verify_dummy_password,
    verify_password,
)

logger = logging.getLogger("app.auth.service")

LOCAL_PROVIDER = "local"
# 로그인 실패 시 어느 쪽이 틀렸는지 노출하지 않는 고정 메시지.
LOGIN_FAILED_MESSAGE = "메일 또는 비밀번호가 잘못되었습니다."


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(dt: datetime) -> datetime:
    """DB에서 온 naive datetime 을 UTC aware 로 보정 (SQLite 등 대비)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class AuthService:
    def __init__(self, repository: AuthRepository | None = None) -> None:
        self.repository = repository or AuthRepository()

    # ── 닉네임 자동 생성 ─────────────────────────────────────
    def _unique_nickname(self, db: Session, base: str) -> str:
        """base 닉네임이 충돌하면 숫자 suffix 를 붙여 유니크한 값을 찾는다.

        (siwoo → siwoo1 → siwoo2 ...). 최종 유니크성은 DB UNIQUE 제약이 보장.
        """
        base = (base or "user").strip() or "user"
        # nickname 컬럼 길이(255) 여유 있음 — base 는 email local-part/카카오 닉네임.
        candidate = base
        suffix = 0
        while self.repository.nickname_exists(db, candidate):
            suffix += 1
            candidate = f"{base}{suffix}"
        return candidate

    # ── 이메일 인증 요청 ─────────────────────────────────────
    def request_verification(self, db: Session, email: str) -> str:
        """매직 링크 인증 요청 생성 + 메일 발송. request_id 반환.

        이미 (local,email) 유저가 있어도 정보 노출을 막기 위해 흐름은 동일하게 진행한다
        (링크 메일은 발송하되, 로그로 안내). 주된 방어는 signup 단계에서의 (provider,email) UNIQUE.
        """
        settings = get_settings()

        existing = self.repository.get_user_by_provider_email(db, LOCAL_PROVIDER, email)
        if existing is not None:
            logger.info("request-verification: 이미 가입된 이메일 email=%s", email)

        raw_token = generate_random_token()
        request_id = generate_random_token()
        expires_at = _now() + timedelta(hours=settings.email_verify_expire_hours)

        verification = EmailVerification(
            email=email,
            token_hash=hash_token(raw_token),
            request_id=request_id,
            expires_at=expires_at,
        )
        self.repository.add_email_verification(db, verification)
        db.commit()

        verify_url = f"{settings.frontend_url}/verify-email?token={raw_token}"
        send_verification_email(email, verify_url)

        return request_id

    # ── 인증 상태 폴링 ───────────────────────────────────────
    def verification_status(self, db: Session, request_id: str) -> dict:
        verification = self.repository.get_verification_by_request_id(db, request_id)
        if verification is None:
            # 존재하지 않는 request_id — 만료로 취급(정보 최소화).
            return {"verified": False, "expired": True}

        expired = _as_aware(verification.expires_at) < _now()
        verified = verification.verified_at is not None
        return {"verified": verified, "expired": expired}

    # ── 매직 링크 검증 ───────────────────────────────────────
    def verify_email(self, db: Session, raw_token: str) -> None:
        """매직 링크 토큰 검증 → verified_at 기록. 실패 시 400."""
        token_hash = hash_token(raw_token)
        verification = self.repository.get_verification_by_token_hash(db, token_hash)
        if verification is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 인증 링크입니다.",
            )
        if _as_aware(verification.expires_at) < _now():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="인증 링크가 만료되었습니다.",
            )
        if verification.consumed_at is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이미 사용된 인증 링크입니다.",
            )
        if verification.verified_at is None:
            verification.verified_at = _now()
            db.commit()

    # ── 가입 ─────────────────────────────────────────────────
    def signup(self, db: Session, payload: SignupRequest) -> tuple[User, str, str]:
        """가입 완료 → (user, access_token, refresh_token) 반환 (가입 즉시 로그인)."""
        verification = self.repository.get_verification_by_request_id(
            db, payload.request_id
        )
        if verification is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 인증 요청입니다.",
            )
        if verification.email != payload.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="인증한 이메일과 일치하지 않습니다.",
            )
        if verification.verified_at is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이메일 인증이 완료되지 않았습니다.",
            )
        if verification.consumed_at is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이미 가입에 사용된 인증 요청입니다.",
            )
        if _as_aware(verification.expires_at) < _now():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="인증 요청이 만료되었습니다.",
            )

        # 닉네임: 사용자 입력값 사용. UNIQUE 제약 위반이 (provider,email) 인지 nickname 인지
        # IntegrityError 로는 구분이 어려우므로 생성 전에 중복 조회로 미리 걸러 409 를 명확히 한다.
        # (중복이면 rollback 전이라 consumed_at 이 찍히지 않아 닉네임만 바꿔 재시도 가능.)
        if self.repository.nickname_exists(db, payload.nickname):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="이미 사용 중인 닉네임입니다.",
            )

        user = User(
            email=payload.email,
            password_hash=hash_password(payload.password),
            nickname=payload.nickname,
            provider=LOCAL_PROVIDER,
            provider_id=None,
            is_active=True,
        )
        try:
            self.repository.add_user(db, user)
            verification.consumed_at = _now()
            db.flush()
        except IntegrityError:
            db.rollback()
            # (provider,email) UNIQUE 위반 = 이미 가입된 계정.
            # (닉네임 경합은 위에서 선조회로 대부분 걸러지지만, 동시성으로 여기 올 수도 있음.)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="이미 가입된 이메일입니다.",
            ) from None

        access_token, refresh_token = self._issue_session(db, user.id)
        db.commit()
        return user, access_token, refresh_token

    # ── 로그인 ───────────────────────────────────────────────
    def login(self, db: Session, email: str, password: str) -> tuple[User, str, str]:
        """(local,email) 로그인. 실패는 계정 유무와 무관하게 동일한 401."""
        user = self.repository.get_user_by_provider_email(db, LOCAL_PROVIDER, email)
        if user is None or user.password_hash is None:
            # 타이밍 공격 방어: 유저가 없어도 더미 해시 검증을 수행.
            verify_dummy_password(password)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=LOGIN_FAILED_MESSAGE
            )
        if not verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=LOGIN_FAILED_MESSAGE
            )
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=LOGIN_FAILED_MESSAGE
            )

        access_token, refresh_token = self._issue_session(db, user.id)
        db.commit()
        return user, access_token, refresh_token

    # ── 소셜 로그인 (upsert) ─────────────────────────────────
    def social_login(
        self, db: Session, profile: SocialProfile
    ) -> tuple[User, str, str]:
        """소셜 프로필로 로그인/가입 (provider,provider_id 기준 upsert)."""
        user = self.repository.get_user_by_provider_id(
            db, profile.provider, profile.provider_id
        )
        if user is None:
            base_nickname = profile.nickname or profile.email.split("@", 1)[0]
            nickname = self._unique_nickname(db, base_nickname)
            user = User(
                email=profile.email,
                password_hash=None,
                nickname=nickname,
                provider=profile.provider,
                provider_id=profile.provider_id,
                is_active=True,
                profile_image_url=profile.profile_image_url,
            )
            try:
                self.repository.add_user(db, user)
                db.flush()
            except IntegrityError:
                db.rollback()
                # 동시 콜백 경합 등 — 다시 조회.
                user = self.repository.get_user_by_provider_id(
                    db, profile.provider, profile.provider_id
                )
                if user is None:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="소셜 계정 생성에 실패했습니다.",
                    ) from None

        access_token, refresh_token = self._issue_session(db, user.id)
        db.commit()
        return user, access_token, refresh_token

    # ── refresh rotation ─────────────────────────────────────
    def refresh(self, db: Session, raw_refresh_token: str) -> tuple[str, str]:
        """refresh 쿠키 검증 → rotation(기존 폐기 + 신규 발급) → (access, refresh)."""
        if not raw_refresh_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="인증이 필요합니다."
            )
        token_hash = hash_token(raw_refresh_token)
        stored = self.repository.get_refresh_token_by_hash(db, token_hash)
        if stored is None or stored.revoked_at is not None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="유효하지 않은 refresh 토큰입니다.",
            )
        if _as_aware(stored.expires_at) < _now():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="refresh 토큰이 만료되었습니다.",
            )

        # rotation: 신규 발급. _issue_session 이 기존 행(stored 포함)을 먼저 삭제하므로
        # 별도 revoke 는 불필요 — 삭제 기반 단일 세션.
        access_token, refresh_token = self._issue_session(db, stored.user_id)
        db.commit()
        return access_token, refresh_token

    # ── 로그아웃 ─────────────────────────────────────────────
    def logout(self, db: Session, raw_refresh_token: str | None) -> None:
        """refresh 토큰이 있으면 해당 user 의 refresh 행을 삭제 (없어도 성공).

        삭제 기반 단일 세션: revoked 행을 남기지 않고 0행으로 정리한다.
        """
        if not raw_refresh_token:
            return
        stored = self.repository.get_refresh_token_by_hash(
            db, hash_token(raw_refresh_token)
        )
        if stored is not None:
            self.repository.delete_refresh_tokens_for_user(db, stored.user_id)
            db.commit()

    # ── 비밀번호 찾기(재설정 링크 발송) ─────────────────────
    def password_forgot(self, db: Session, email: str) -> None:
        """(local,email) 유저가 있으면 재설정 링크 메일 발송. 없어도 항상 성공.

        가입 여부 비노출 정책 — 유저 존재 여부와 무관하게 동일하게 반환한다.
        토큰은 DB 저장 없이 secret_key+password_hash 로 서명(비번 바뀌면 자동 무효).
        """
        settings = get_settings()
        user = self.repository.get_user_by_provider_email(db, LOCAL_PROVIDER, email)
        # 소셜 유저(password_hash NULL)는 재설정 대상 아님 → 조용히 무시.
        if user is None or user.password_hash is None:
            logger.info("password-forgot: 대상 없음/소셜 email=%s", email)
            return

        token = create_password_reset_token(user.id, user.password_hash)
        reset_url = f"{settings.frontend_url}/reset-password?token={token}"
        send_password_reset_email(email, reset_url)

    # ── 비밀번호 재설정(토큰 검증 → 갱신) ────────────────────
    def password_reset(self, db: Session, payload: PasswordResetRequest) -> None:
        """재설정 토큰 검증 후 password_hash 갱신 + 기존 refresh 전부 삭제(강제 재로그인)."""
        # 1) 서명 검증 전에 sub(user_id)만 추출 → 어떤 유저인지 파악.
        try:
            user_id = extract_reset_token_subject(payload.token)
        except (JWTError, ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 링크입니다.",
            ) from None

        user = self.repository.get_user_by_id(db, user_id)
        if user is None or user.password_hash is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 링크입니다.",
            )

        # 2) 그 유저의 현재 password_hash 로 정식 검증(서명/만료/type).
        try:
            decode_password_reset_token(payload.token, user.password_hash)
        except (JWTError, ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="만료되었거나 잘못된 링크입니다.",
            ) from None

        # 3) 비번 갱신 + 강제 재로그인(refresh 전부 삭제).
        user.password_hash = hash_password(payload.new_password)
        self.repository.delete_refresh_tokens_for_user(db, user.id)
        db.commit()

    # ── 비밀번호 변경(로그인 상태) ───────────────────────────
    def password_change(
        self, db: Session, user: User, payload: PasswordChangeRequest
    ) -> None:
        """현재 비번 확인 후 password_hash 갱신. 현재 세션은 유지(재로그인 강제 안 함)."""
        if user.password_hash is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="비밀번호를 사용할 수 없는 계정입니다.",
            )
        if not verify_password(payload.current_password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="현재 비밀번호가 일치하지 않습니다.",
            )
        if payload.new_password == payload.current_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="새 비밀번호가 기존과 동일합니다.",
            )

        user.password_hash = hash_password(payload.new_password)
        db.commit()

    # ── 세션(access+refresh) 발급 헬퍼 ───────────────────────
    def _issue_session(self, db: Session, user_id: int) -> tuple[str, str]:
        """access JWT + refresh JWT 발급 & refresh_tokens 행 생성. 커밋은 호출부.

        단일 세션 정책: 신규 행 삽입 전에 해당 user 의 기존 refresh 행을 모두 삭제한다.
        → login/signup/refresh/kakao 모두 user 당 정확히 1행으로 수렴.
        """
        settings = get_settings()
        self.repository.delete_refresh_tokens_for_user(db, user_id)
        access_token = create_access_token(user_id)
        refresh_token = create_refresh_token(user_id)
        expires_at = _now() + timedelta(days=settings.refresh_token_expire_days)
        self.repository.add_refresh_token(
            db,
            RefreshToken(
                user_id=user_id,
                token_hash=hash_token(refresh_token),
                expires_at=expires_at,
            ),
        )
        return access_token, refresh_token
