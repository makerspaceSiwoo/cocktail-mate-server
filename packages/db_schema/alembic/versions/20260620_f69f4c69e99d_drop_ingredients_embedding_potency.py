"""drop ingredients.embedding + potency (revert 03559d9497ec)

Revision ID: f69f4c69e99d
Revises: 03559d9497ec
Create Date: 2026-06-20

ingredients.embedding/potency 추가를 취소. (앞으로 가는 drop 마이그레이션으로 되돌림)
"""

from typing import Sequence, Union

import pgvector.sqlalchemy  # noqa: F401 — downgrade의 Vector 타입 렌더링에 필요
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f69f4c69e99d"
down_revision: Union[str, None] = "03559d9497ec"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_EMBEDDING_DIM = 64


def upgrade() -> None:
    op.drop_column("ingredients", "potency")
    op.drop_column("ingredients", "embedding")


def downgrade() -> None:
    op.add_column(
        "ingredients",
        sa.Column(
            "embedding", pgvector.sqlalchemy.Vector(_EMBEDDING_DIM), nullable=True
        ),
    )
    op.add_column("ingredients", sa.Column("potency", sa.Float(), nullable=True))
