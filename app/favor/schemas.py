"""취향 추천 응답 스키마."""

from pydantic import BaseModel


class FavorItem(BaseModel):
    id: int
    name: str
    similarity: float
