"""feature1: embedding(64) + ANN, recipe[], abv/amount numeric, role/category/ingredient abv

Revision ID: 1b066c2c61df
Revises: 4e3bb0bb2d9f
Create Date: 2026-06-20

기능 1(칵테일 유사도 임베딩 + HNSW ANN) 범위 스키마 변경. 데이터가 없어 USING 캐스트는
안전(빈 테이블). 기능 2(맛/향 패싯)는 별도 추가 마이그레이션으로 추후.

대상: ingredients / cocktails / cocktail_ingredients.
변경 제외: users, likes, alembic_version.
"""
from typing import Sequence, Union

import pgvector.sqlalchemy  # noqa: F401 — Vector 타입 렌더링에 필요
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "1b066c2c61df"
down_revision: Union[str, None] = "4e3bb0bb2d9f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# base_tag 허용값 (모델 BASE_TAGS와 동일하게 유지)
_BASE_TAG_CHECK = (
    "base_tag IS NULL OR base_tag IN "
    "('vodka','gin','whiskey','tequila','brandy','rum','liqueur','non_alcoholic','other')"
)
_ABV_CHECK = "abv IS NULL OR (abv >= 0 AND abv <= 100)"


def upgrade() -> None:
    # --- ingredients: category + abv(도수 %) 추가 ---
    op.add_column("ingredients", sa.Column("category", sa.String(length=50), nullable=True))
    op.add_column("ingredients", sa.Column("abv", sa.Numeric(precision=4, scale=1), nullable=True))
    op.create_check_constraint("chk_ingredients_abv", "ingredients", _ABV_CHECK)

    # --- cocktails ---
    # num_like 제거 (likes의 COUNT로 유도)
    op.drop_column("cocktails", "num_like")
    # name UNIQUE 제거 (surrogate id PK만 식별자로 유지; 동명 변형 허용)
    op.drop_constraint("cocktails_name_key", "cocktails", type_="unique")
    # abv: REAL -> numeric(4,1) + CHECK(0..100)
    op.alter_column(
        "cocktails",
        "abv",
        existing_type=sa.REAL(),
        type_=sa.Numeric(precision=4, scale=1),
        existing_nullable=True,
        postgresql_using="abv::numeric(4,1)",
    )
    op.create_check_constraint("chk_cocktails_abv", "cocktails", _ABV_CHECK)
    # recipe: text -> text[] (단계 배열). 빈 테이블이라 기존 값 캐스트 영향 없음.
    op.alter_column(
        "cocktails",
        "recipe",
        existing_type=sa.Text(),
        type_=postgresql.ARRAY(sa.Text()),
        existing_nullable=True,
        postgresql_using="CASE WHEN recipe IS NULL THEN NULL ELSE ARRAY[recipe] END",
    )
    # base_tag 값 제약
    op.create_check_constraint("chk_cocktails_base_tag", "cocktails", _BASE_TAG_CHECK)
    # embedding: vector(1536) -> vector(64). 재학습 전제 → 전부 NULL로 재설정.
    op.alter_column(
        "cocktails",
        "embedding",
        existing_type=pgvector.sqlalchemy.Vector(1536),
        type_=pgvector.sqlalchemy.Vector(64),
        existing_nullable=True,
        postgresql_using="NULL",
    )
    # 임베딩 갱신 시각
    op.add_column(
        "cocktails",
        sa.Column("embedding_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    # [기능 1] ANN — HNSW cosine
    op.create_index(
        "idx_cocktails_embedding_hnsw",
        "cocktails",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    # --- cocktail_ingredients: amount 정밀화 + role 추가 ---
    op.alter_column(
        "cocktail_ingredients",
        "amount",
        existing_type=sa.REAL(),
        type_=sa.Numeric(precision=8, scale=3),
        existing_nullable=True,
        postgresql_using="amount::numeric(8,3)",
    )
    op.add_column(
        "cocktail_ingredients",
        sa.Column("role", sa.String(length=20), nullable=True),
    )


def downgrade() -> None:
    # --- cocktail_ingredients ---
    op.drop_column("cocktail_ingredients", "role")
    op.alter_column(
        "cocktail_ingredients",
        "amount",
        existing_type=sa.Numeric(precision=8, scale=3),
        type_=sa.REAL(),
        existing_nullable=True,
        postgresql_using="amount::real",
    )

    # --- cocktails ---
    op.drop_index("idx_cocktails_embedding_hnsw", table_name="cocktails")
    op.drop_column("cocktails", "embedding_updated_at")
    op.alter_column(
        "cocktails",
        "embedding",
        existing_type=pgvector.sqlalchemy.Vector(64),
        type_=pgvector.sqlalchemy.Vector(1536),
        existing_nullable=True,
        postgresql_using="NULL",
    )
    op.drop_constraint("chk_cocktails_base_tag", "cocktails", type_="check")
    op.alter_column(
        "cocktails",
        "recipe",
        existing_type=postgresql.ARRAY(sa.Text()),
        type_=sa.Text(),
        existing_nullable=True,
        postgresql_using="array_to_string(recipe, '\n')",
    )
    op.drop_constraint("chk_cocktails_abv", "cocktails", type_="check")
    op.alter_column(
        "cocktails",
        "abv",
        existing_type=sa.Numeric(precision=4, scale=1),
        type_=sa.REAL(),
        existing_nullable=True,
        postgresql_using="abv::real",
    )
    op.create_unique_constraint("cocktails_name_key", "cocktails", ["name"])
    op.add_column(
        "cocktails",
        sa.Column("num_like", sa.Integer(), server_default="0", nullable=False),
    )

    # --- ingredients ---
    op.drop_constraint("chk_ingredients_abv", "ingredients", type_="check")
    op.drop_column("ingredients", "abv")
    op.drop_column("ingredients", "category")
