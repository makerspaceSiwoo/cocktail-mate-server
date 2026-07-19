"""add name_en (citext) to cocktails/ingredients for English-name search

Revision ID: fcbddb317195
Revises: 1b066c2c61df
Create Date: 2026-06-20

기능 1 스키마 위에 얹는 순수 추가(additive) 변경. 기존 컬럼/제약은 건드리지 않는다.
- cocktails.name_en   citext NULL          (영문 검색; 출처별 중복 가능 → UNIQUE 미설정)
- ingredients.name_en citext NULL UNIQUE   (영문 검색 + 외부 dedup 키)
대소문자 무시 정확 매칭을 위해 citext 확장 사용.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "fcbddb317195"
down_revision: Union[str, None] = "1b066c2c61df"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS citext")

    # --- cocktails ---
    op.add_column("cocktails", sa.Column("name_en", postgresql.CITEXT(), nullable=True))
    op.create_index("idx_cocktails_name_en", "cocktails", ["name_en"], unique=False)
    op.create_index("idx_cocktails_name", "cocktails", ["name"], unique=False)

    # --- ingredients ---
    op.add_column(
        "ingredients", sa.Column("name_en", postgresql.CITEXT(), nullable=True)
    )
    op.create_unique_constraint("uq_ingredients_name_en", "ingredients", ["name_en"])
    op.create_index("idx_ingredients_name", "ingredients", ["name"], unique=False)


def downgrade() -> None:
    # --- ingredients ---
    op.drop_index("idx_ingredients_name", table_name="ingredients")
    op.drop_constraint("uq_ingredients_name_en", "ingredients", type_="unique")
    op.drop_column("ingredients", "name_en")

    # --- cocktails ---
    op.drop_index("idx_cocktails_name", table_name="cocktails")
    op.drop_index("idx_cocktails_name_en", table_name="cocktails")
    op.drop_column("cocktails", "name_en")

    # citext 확장은 다른 객체가 쓸 수 있어 DROP하지 않는다.
