"""좋아요 ORM 모델 (user-cocktail 다대다).

(user_id, cocktail_id) 조합은 UNIQUE — 한 유저가 같은 칵테일을 중복 좋아요 불가.
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from cocktail_mate_db.base import Base


class Like(Base):
    __tablename__ = "likes"
    __table_args__ = (
        # 제약 이름은 운영 DB(inline UNIQUE의 PG 기본 이름)와 동일하게 유지
        UniqueConstraint(
            "user_id", "cocktail_id", name="likes_user_id_cocktail_id_key"
        ),
        Index("idx_likes_user", "user_id"),
        Index("idx_likes_cocktail", "cocktail_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    cocktail_id: Mapped[int] = mapped_column(
        ForeignKey("cocktails.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
