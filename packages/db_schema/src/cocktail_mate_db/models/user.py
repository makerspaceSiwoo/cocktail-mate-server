"""유저 ORM 모델.

소셜 로그인 전용 모델이다. 유저 식별자(LoginID)는 (provider, provider_id) 조합이다.
이메일/비밀번호 로그인은 지원하지 않는다 — password_hash 컬럼 없음.
email/nickname/profile_image_url 은 소셜 프로필에서 받아 저장하되, 이메일은 제공되지 않을 수 있어 nullable 이다.
"""
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Identity, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from cocktail_mate_db.base import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        # 소셜 유저 식별용 유니크: provider_id 는 항상 존재(NOT NULL)하므로 partial 이 아닌 일반 unique.
        UniqueConstraint("provider", "provider_id", name="uq_users_provider_provider_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    provider: Mapped[str] = mapped_column(String(20), nullable=False)  # 'kakao' | 'google' 등
    provider_id: Mapped[str] = mapped_column(String(255), nullable=False)  # 소셜 제공자의 고유 사용자 ID
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)  # 소셜 프로필 이메일 (미제공 가능 → nullable)
    nickname: Mapped[str] = mapped_column(String(255), nullable=False)  # 소셜 닉네임 (중복 허용 — 식별자는 provider_id)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    profile_image_url: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
