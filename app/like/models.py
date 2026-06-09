"""좋아요 ORM 모델 (user-cocktail 다대다). db/init/02-schema.sql 의 likes 테이블과 대응.

(user_id, cocktail_id) 조합은 UNIQUE — 한 유저가 같은 칵테일을 중복 좋아요 불가.
"""
from datetime import datetime

from sqlalchemy import BigInteger, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Like(Base):
    __tablename__ = "likes"
    __table_args__ = (
        UniqueConstraint("user_id", "cocktail_id", name="uq_user_cocktail_like"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    cocktail_id: Mapped[int] = mapped_column(
        ForeignKey("cocktails.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
