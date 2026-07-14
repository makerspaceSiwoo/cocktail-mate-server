"""인증 비즈니스 로직 (소셜 로그인 전용).

핵심 원칙:
- 유저 식별은 (provider, provider_id) 기준 upsert. 이메일/비밀번호 개념 없음.
- refresh 토큰 원본은 저장 안 함 — 항상 sha256 해시만 저장/대조. 단일 세션(user당 1행).

세션 커밋은 이 계층에서 관리한다 (repository 는 flush 까지만).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from cocktail_mate_db.models import RefreshToken, User

from app.auth.nickname import generate_random_nickname
from app.auth.providers import SocialProfile
from app.auth.repository import AuthRepository
from app.core.config import get_settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_token,
)

logger = logging.getLogger("app.auth.service")


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

    # ── 소셜 로그인 (upsert) ─────────────────────────────────
    def social_login(
        self, db: Session, profile: SocialProfile
    ) -> tuple[User, str, str]:
        """소셜 프로필로 로그인/가입 ((provider, provider_id) 기준 upsert).

        이메일은 프로필에 있으면 저장하고 없으면 NULL. 닉네임은 중복 허용(식별자는 provider_id)이며
        프로필에 없으면 랜덤 닉네임(형용사+명사)을 부여한다.
        """
        user = self.repository.get_user_by_provider_id(
            db, profile.provider, profile.provider_id
        )
        if user is None:
            nickname = (profile.nickname or "").strip() or generate_random_nickname()
            user = User(
                email=profile.email,
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

    # ── 닉네임 변경 ──────────────────────────────────────────
    def change_nickname(
            self,
            db: Session,
            user: User,
            nickname: str,
    ) -> User:
        """현재 로그인한 사용자의 닉네임을 변경한다."""
        if user.nickname == nickname:
            return user
        
        try:
            updated_user = self.repository.update_nickname(
                db,
                user,
                nickname,
            )
            db.commit()
            db.refresh(updated_user)
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="이미 사용 중인 닉네임입니다.",
            ) from None
        return updated_user

    # ── 세션(access+refresh) 발급 헬퍼 ───────────────────────
    def _issue_session(self, db: Session, user_id: int) -> tuple[str, str]:
        """access JWT + refresh JWT 발급 & refresh_tokens 행 생성. 커밋은 호출부.

        단일 세션 정책: 신규 행 삽입 전에 해당 user 의 기존 refresh 행을 모두 삭제한다.
        → 소셜 로그인/refresh 모두 user 당 정확히 1행으로 수렴.
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
