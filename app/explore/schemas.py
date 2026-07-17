"""탐색(3D 벡터 맵 시각화)용 응답 스키마."""

from pydantic import BaseModel


class ExploreItem(BaseModel):
    id: int
    name: str
    abv: float | None = None
    embedding3d: list[float]
