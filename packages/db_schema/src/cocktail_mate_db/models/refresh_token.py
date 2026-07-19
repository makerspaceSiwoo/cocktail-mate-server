"""리프레시 토큰 ORM 모델.

JWT access token 재발급에 사용되는 리프레시 토큰을 관리한다.
토큰 원본은 절대 저장하지 않고, token_hash(SHA-256 등)만 저장한다.
"""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Identity, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from cocktail_mate_db.base import Base


class RefreshToken(Base):
    """리프레시 토큰 테이블.

    - user_id: 토큰 소유 유저 (CASCADE 삭제)
    - token_hash: 리프레시 토큰의 해시값 (원본 저장 금지)
    - expires_at: 토큰 만료 시각
    - revoked_at: 토큰 폐기 시각 (NULL이면 유효)
    """

    __tablename__ = "refresh_tokens"
    __table_args__ = (
        Index("idx_refresh_tokens_user_id", "user_id"),
        Index("idx_refresh_tokens_token_hash", "token_hash"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
