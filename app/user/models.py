"""유저 ORM 모델. db/init/02-schema.sql 의 users 테이블과 대응.

로그인 식별자(LoginID)는 이메일(email)로 사용한다. 비밀번호는 평문이 아닌 해시로 저장한다.
"""
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)  # 로그인 식별자
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    nickname: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    profile_image_url: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
