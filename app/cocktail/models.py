"""칵테일/재료 ORM 모델.

DB 스키마는 db/init/02-schema.sql 과 1:1 대응한다. 응답(View) 필드명은 camelCase 이지만
DB/ORM 컬럼은 snake_case 이며, 매핑은 service/schemas 계층에서 처리한다.
"""
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

# 임베딩 차원 (미정 → 우선 1536, OpenAI text-embedding-3-small 기준). 추후 변경 가능.
EMBEDDING_DIM = 1536


class Cocktail(Base):
    __tablename__ = "cocktails"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    image_url: Mapped[str | None] = mapped_column(Text)
    glass: Mapped[str | None] = mapped_column(String(255))
    abv: Mapped[float | None] = mapped_column(Float)
    num_like: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    recipe: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    base_tag: Mapped[str | None] = mapped_column(String(50))  # 베이스 술타입
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    ingredients: Mapped[list["CocktailIngredient"]] = relationship(
        back_populates="cocktail", cascade="all, delete-orphan"
    )


class Ingredient(Base):
    __tablename__ = "ingredients"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)


class CocktailIngredient(Base):
    """칵테일-재료 조인. amount(용량) + unit(표기: ml/oz/dash 등, ERD의 'Field')."""

    __tablename__ = "cocktail_ingredients"
    __table_args__ = (
        UniqueConstraint("cocktail_id", "ingredient_id", name="uq_cocktail_ingredient"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    cocktail_id: Mapped[int] = mapped_column(
        ForeignKey("cocktails.id", ondelete="CASCADE"), nullable=False
    )
    ingredient_id: Mapped[int] = mapped_column(
        ForeignKey("ingredients.id", ondelete="RESTRICT"), nullable=False
    )
    amount: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(50))

    cocktail: Mapped["Cocktail"] = relationship(back_populates="ingredients")
    ingredient: Mapped["Ingredient"] = relationship()
