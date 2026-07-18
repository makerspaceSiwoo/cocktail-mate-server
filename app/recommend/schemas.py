"""추천(유사 칵테일) 응답 스키마."""

from pydantic import BaseModel


class RecommendItem(BaseModel):
    id: int
    name: str
    similarity: float
