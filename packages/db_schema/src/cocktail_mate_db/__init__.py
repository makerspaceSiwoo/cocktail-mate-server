"""cocktail-mate canonical DB 패키지.

이 모듈을 import하면 모든 모델이 Base.metadata에 등록된다
(alembic env.py는 이것만 import하면 됨).
"""

from cocktail_mate_db.base import Base
from cocktail_mate_db.models import (
    EMBEDDING_DIM,
    Cocktail,
    CocktailIngredient,
    Ingredient,
    Like,
    RefreshToken,
    User,
)

__all__ = [
    "Base",
    "EMBEDDING_DIM",
    "Cocktail",
    "CocktailIngredient",
    "Ingredient",
    "Like",
    "RefreshToken",
    "User",
]
