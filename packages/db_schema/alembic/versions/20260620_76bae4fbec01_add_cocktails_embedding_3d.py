"""add cocktails.embedding_3d (vector(3), nullable) — 시각화용 축소 좌표

Revision ID: 76bae4fbec01
Revises: 36bf010451dc
Create Date: 2026-06-20

64D 유사도 임베딩을 3D로 축소한 시각화용 좌표(순수 추가). 풀 임베딩에서 파생.
인덱스 없음(전체 조회해 그리는 용도라 ANN 불필요).
"""
from typing import Sequence, Union

import pgvector.sqlalchemy  # noqa: F401 — Vector 타입 렌더링에 필요
import sqlalchemy as sa  # noqa: F401
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "76bae4fbec01"
down_revision: Union[str, None] = "36bf010451dc"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "cocktails",
        sa.Column("embedding_3d", pgvector.sqlalchemy.Vector(3), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("cocktails", "embedding_3d")
