"""칵테일/재료 ORM 모델.

운영 DB 스키마와 1:1 대응한다 (인덱스·제약 이름까지 동일 — autogenerate 드리프트 방지).
응답(View) 필드명은 camelCase 이지만 DB/ORM 컬럼은 snake_case 이며,
매핑은 api-server의 service/schemas 계층에서 처리한다.

[기능 1] 칵테일 유사도 임베딩 + ANN(HNSW cosine) 범위. 기능 2(맛/향 패싯)는 별도 테이블로 추후 추가.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cocktail_mate_db.base import Base

# [기능 1] 유사도 임베딩 차원 — 자체 학습(삼중항) 모델의 출력 차원.
# 1위 모델 G0_ground_linear(OUT_DIM=32) 확정값. 변경 시 cocktails/ingredients의 embedding 컬럼 타입
# + HNSW 인덱스를 재생성하는 마이그레이션 필요.
EMBEDDING_DIM = 32

# base_tag 허용값 (text + CHECK; 종류가 늘 수 있어 enum 대신 CHECK).
BASE_TAGS = (
    "vodka",
    "gin",
    "whiskey",
    "tequila",
    "brandy",
    "rum",
    "liqueur",
    "non_alcoholic",
    "other",
)


class Cocktail(Base):
    __tablename__ = "cocktails"
    __table_args__ = (
        # abv는 % (0~100) 또는 무알콜 NULL
        CheckConstraint(
            "abv IS NULL OR (abv >= 0 AND abv <= 100)",
            name="chk_cocktails_abv",
        ),
        CheckConstraint(
            "base_tag IS NULL OR base_tag IN "
            "('vodka','gin','whiskey','tequila','brandy','rum','liqueur','non_alcoholic','other')",
            name="chk_cocktails_base_tag",
        ),
        Index("idx_cocktails_base_tag", "base_tag"),
        # 검색 인덱스: 영문(name_en, 대소문자 무시) / 한글(name) 정확 검색.
        Index("idx_cocktails_name_en", "name_en"),
        Index("idx_cocktails_name", "name"),
        # [기능 1] ANN — HNSW cosine. embedding IS NULL 행은 인덱스에서 자연히 제외.
        Index(
            "idx_cocktails_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    # name은 UNIQUE 아님: 동명(同名) 변형 칵테일 허용. 식별은 surrogate id PK로만.
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # 외부 출처 정식 영문명. citext(대소문자 무시) → 영문 정확 검색 + 외부 매칭.
    # 출처별 중복 가능성이 있어 UNIQUE 미설정(기본). nullable(수동 추가 칵테일).
    name_en: Mapped[str | None] = mapped_column(CITEXT)
    image_url: Mapped[str | None] = mapped_column(Text)
    glass: Mapped[str | None] = mapped_column(String(255))
    # numeric(4,1)이지만 Python에서는 float로 (asdecimal=False) — api-server 호환 유지.
    abv: Mapped[float | None] = mapped_column(Numeric(4, 1, asdecimal=False))
    # 레시피 단계 배열. 예: ['얼음과 함께 셰이커에','15초 셰이크','더블 스트레인']
    recipe: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    description: Mapped[str | None] = mapped_column(Text)  # LLM 생성 맛 설명
    base_tag: Mapped[str | None] = mapped_column(String(50))  # 주 베이스 (CHECK 참조)
    # [기능 1] 유사도 임베딩. 학습 전/신규 칵테일은 NULL.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    # 임베딩을 3D로 축소한 시각화용 좌표(구 배치). 풀 임베딩에서 파생, 학습 전 NULL.
    embedding_3d: Mapped[list[float] | None] = mapped_column(Vector(3))
    # 임베딩 마지막 갱신 시각(재학습 추적). 학습 전 NULL.
    embedding_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # num_like 제거: likes에서 COUNT로 유도.

    ingredients: Mapped[list["CocktailIngredient"]] = relationship(
        back_populates="cocktail", cascade="all, delete-orphan"
    )


class Ingredient(Base):
    __tablename__ = "ingredients"
    __table_args__ = (
        # 도수 % (0~100) 또는 무알콜/비주류 NULL
        CheckConstraint(
            "abv IS NULL OR (abv >= 0 AND abv <= 100)",
            name="chk_ingredients_abv",
        ),
        # 재료 영문명은 유일(대소문자 무시) — 외부 데이터 dedup 키.
        UniqueConstraint("name_en", name="uq_ingredients_name_en"),
        # 한글(일반 이름) 검색 인덱스 (name UNIQUE는 별도 검색 인덱스로 직접 못 쓰는 경우 대비).
        Index("idx_ingredients_name", "name"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    # name UNIQUE 유지: 같은 재료가 여러 id로 갈라지면 임베딩 학습이 깨지므로 필수.
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    # 선택: 'spirit'|'liqueur'|'juice'|'syrup'|'bitters'|'garnish'|'other' (자유 텍스트)
    category: Mapped[str | None] = mapped_column(String(50))
    # 알콜 도수(%). 무알콜 재료는 NULL. numeric(4,1)이지만 Python float.
    abv: Mapped[float | None] = mapped_column(Numeric(4, 1, asdecimal=False))
    # 외부 출처 정식 영문명. citext(대소문자 무시) UNIQUE → 영문 검색 + 외부 dedup 키.
    name_en: Mapped[str | None] = mapped_column(CITEXT)
    image_url: Mapped[str | None] = mapped_column(Text)  # 재료 이미지 URL
    description: Mapped[str | None] = mapped_column(Text)  # 재료 설명(맛/특징 등)
    # [기능 1] 재료 맛 임베딩 (G0_ground_linear: E_free + Wg·flavor_feat).
    # 칵테일 임베딩의 구성 단위. 비-맛 재료(수량 없음/얼음/가니시 등)·학습 전은 NULL.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    # [기능 1] 재료 강도(potency) — 칵테일 풀링 가중치 softplus(potency + γ·log amount)의 학습 bias. 학습 전 NULL.
    potency: Mapped[float | None] = mapped_column(Float)


class CocktailIngredient(Base):
    """칵테일-재료 조인 (재료당 1 row). amount=용량(실수), role=특수목적 구분."""

    __tablename__ = "cocktail_ingredients"
    __table_args__ = (
        # 제약 이름은 운영 DB(inline UNIQUE의 PG 기본 이름)와 동일하게 유지
        UniqueConstraint(
            "cocktail_id",
            "ingredient_id",
            name="cocktail_ingredients_cocktail_id_ingredient_id_key",
        ),
        Index("idx_cocktail_ingredients_cocktail", "cocktail_id"),
        Index("idx_cocktail_ingredients_ingr", "ingredient_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    cocktail_id: Mapped[int] = mapped_column(
        ForeignKey("cocktails.id", ondelete="CASCADE"), nullable=False
    )
    ingredient_id: Mapped[int] = mapped_column(
        ForeignKey("ingredients.id", ondelete="RESTRICT"), nullable=False
    )
    # numeric(8,3) (0.5/1.5/2 dash 등). "to taste"/presence-only는 NULL(학습 시 수량 1). Python float.
    amount: Mapped[float | None] = mapped_column(Numeric(8, 3, asdecimal=False))
    unit: Mapped[str | None] = mapped_column(
        String(50)
    )  # 'ml','oz','dash'... presence-only는 NULL

    cocktail: Mapped["Cocktail"] = relationship(back_populates="ingredients")
    ingredient: Mapped["Ingredient"] = relationship()
