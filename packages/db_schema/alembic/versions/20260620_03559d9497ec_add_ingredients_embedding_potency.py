"""add ingredients.embedding (vector(64)) + potency (float), nullable

Revision ID: 03559d9497ec
Revises: d28c7ffc95e4
Create Date: 2026-06-20

모델 학습에서 산출되는 재료 임베딩/강도 저장(순수 추가). cocktails와 동일 차원(64).
인덱스 없음(저장 용도). 둘 다 학습 전 NULL.
"""

from typing import Sequence, Union

import pgvector.sqlalchemy  # noqa: F401 — Vector 타입 렌더링에 필요
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "03559d9497ec"
down_revision: Union[str, None] = "d28c7ffc95e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# cocktails.embedding과 동일 차원 (모델 cocktail_mate_db.EMBEDDING_DIM = 64 스냅샷)
_EMBEDDING_DIM = 64


def upgrade() -> None:
    op.add_column(
        "ingredients",
        sa.Column(
            "embedding", pgvector.sqlalchemy.Vector(_EMBEDDING_DIM), nullable=True
        ),
    )
    op.add_column("ingredients", sa.Column("potency", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("ingredients", "potency")
    op.drop_column("ingredients", "embedding")
