"""모든 ORM 모델의 공통 베이스 (canonical).

api-server는 `from cocktail_mate_db.base import Base`로 re-export해서 사용한다.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """모든 ORM 모델의 공통 베이스."""
