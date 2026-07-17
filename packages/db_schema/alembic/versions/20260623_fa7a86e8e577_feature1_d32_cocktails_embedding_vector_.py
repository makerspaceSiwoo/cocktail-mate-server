"""feature1 D32: cocktails.embedding vector(32) + ingredients embedding/potency

Revision ID: fa7a86e8e577
Revises: f69f4c69e99d
Create Date: 2026-06-23

기능 1 임베딩 차원을 placeholder 64 → 확정 **32** 로 맞춘다(1위 모델 G0_ground_linear, OUT_DIM=32).
- cocktails.embedding: vector(64) → vector(32). HNSW 인덱스가 컬럼에 의존하므로 drop → ALTER → 재생성.
  데이터가 없는 빈 테이블이라 USING NULL 캐스트는 안전(기존 임베딩 전무).
- ingredients: embedding vector(32) + potency(float) 추가(앞서 f69f4c69e99d 에서 drop했던 것을 32D로 재도입).
  재료 임베딩은 칵테일 임베딩의 구성 단위(저장 용도, ANN 인덱스 없음). 둘 다 학습 전/비-맛 재료는 NULL.
"""
from typing import Sequence, Union

import pgvector.sqlalchemy  # noqa: F401 — Vector 타입 렌더링에 필요
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fa7a86e8e577"
down_revision: Union[str, None] = "f69f4c69e99d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_HNSW_INDEX = "idx_cocktails_embedding_hnsw"


def upgrade() -> None:
    # --- cocktails.embedding: vector(64) -> vector(32) (HNSW 인덱스 재생성 동반) ---
    op.drop_index(_HNSW_INDEX, table_name="cocktails")
    op.alter_column(
        "cocktails",
        "embedding",
        existing_type=pgvector.sqlalchemy.Vector(64),
        type_=pgvector.sqlalchemy.Vector(32),
        existing_nullable=True,
        postgresql_using="NULL",  # 빈 테이블 + 재학습 전제 → 전부 NULL로 재설정
    )
    op.create_index(
        _HNSW_INDEX,
        "cocktails",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    # --- ingredients: 맛 임베딩(32) + 강도(potency) 추가 ---
    op.add_column(
        "ingredients",
        sa.Column("embedding", pgvector.sqlalchemy.Vector(32), nullable=True),
    )
    op.add_column("ingredients", sa.Column("potency", sa.Float(), nullable=True))


def downgrade() -> None:
    # --- ingredients ---
    op.drop_column("ingredients", "potency")
    op.drop_column("ingredients", "embedding")

    # --- cocktails.embedding: vector(32) -> vector(64) ---
    op.drop_index(_HNSW_INDEX, table_name="cocktails")
    op.alter_column(
        "cocktails",
        "embedding",
        existing_type=pgvector.sqlalchemy.Vector(32),
        type_=pgvector.sqlalchemy.Vector(64),
        existing_nullable=True,
        postgresql_using="NULL",
    )
    op.create_index(
        _HNSW_INDEX,
        "cocktails",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
