"""drop cocktail_ingredients.role

Revision ID: d28c7ffc95e4
Revises: 76bae4fbec01
Create Date: 2026-06-20

cocktail_ingredients.role 제거. (가니시/데코 구분 컬럼 — 사용 중단)
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d28c7ffc95e4"
down_revision: Union[str, None] = "76bae4fbec01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("cocktail_ingredients", "role")


def downgrade() -> None:
    op.add_column(
        "cocktail_ingredients",
        sa.Column("role", sa.String(length=20), nullable=True),
    )
