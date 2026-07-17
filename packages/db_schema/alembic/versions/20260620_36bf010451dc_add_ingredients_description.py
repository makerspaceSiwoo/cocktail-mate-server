"""add ingredients.description (text, nullable)

Revision ID: 36bf010451dc
Revises: a371d4283542
Create Date: 2026-06-20

재료 설명 컬럼 추가(순수 추가). cocktails.description과 동일하게 text/nullable.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "36bf010451dc"
down_revision: Union[str, None] = "a371d4283542"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("ingredients", sa.Column("description", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("ingredients", "description")
