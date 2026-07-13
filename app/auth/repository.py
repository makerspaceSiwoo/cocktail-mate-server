"""인증 데이터 접근 계층 — users / refresh_tokens.

세션(Session)은 호출부(service)가 주입한다. 커밋/롤백은 service 가 관리한다.
"""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from cocktail_mate_db.models import RefreshToken, User


class AuthRepository:
    # ── User ────────────────────────────────────────────────
    def get_user_by_provider_id(
        self, db: Session, provider: str, provider_id: str
    ) -> User | None:
        return db.execute(
            select(User).where(
                User.provider == provider, User.provider_id == provider_id
            )
        ).scalar_one_or_none()

    def get_user_by_id(self, db: Session, user_id: int) -> User | None:
        return db.get(User, user_id)

    def add_user(self, db: Session, user: User) -> User:
        db.add(user)
        db.flush()  # id 확보 (커밋은 service)
        return user
    
    def update_nickname(
            self,
            db: Session,
            user: User,
            nickname: str,
    ) -> User:
        user.nickname = nickname
        db.flush()
        return user

    # ── RefreshToken ────────────────────────────────────────
    def add_refresh_token(
        self, db: Session, refresh_token: RefreshToken
    ) -> RefreshToken:
        db.add(refresh_token)
        db.flush()
        return refresh_token

    def get_refresh_token_by_hash(
        self, db: Session, token_hash: str
    ) -> RefreshToken | None:
        return db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        ).scalar_one_or_none()

    def delete_refresh_tokens_for_user(self, db: Session, user_id: int) -> None:
        """해당 user 의 refresh_tokens 행을 전부 삭제 (단일 세션 보장용).

        revoked_at 로 남겨두면 행이 무한 누적되므로 삭제로 정리한다.
        """
        db.execute(delete(RefreshToken).where(RefreshToken.user_id == user_id))
        db.flush()
