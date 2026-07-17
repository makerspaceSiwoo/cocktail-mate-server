"""add ingredients.image_url (text, nullable)

Revision ID: a371d4283542
Revises: fcbddb317195
Create Date: 2026-06-20

재료 이미지 URL 추가(순수 추가). cocktails.image_url과 동일하게 text/nullable.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a371d4283542"
down_revision: Union[str, None] = "fcbddb317195"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("ingredients", sa.Column("image_url", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("ingredients", "image_url")
