"""추천(유사 칵테일 및 맛 선택) 요청/응답 스키마."""

from pydantic import BaseModel, Field, field_validator


class RecommendItem(BaseModel):
    id: int
    name: str
    nameEn: str | None = None
    description: str | None = None
    imageUrl: str | None = None
    abv: float | None = None
    similarity: float


class TasteDescriptorItem(BaseModel):
    id: int
    code: str
    labelKo: str
    category: str


class TasteDescriptorCatalogResponse(BaseModel):
    items: list[TasteDescriptorItem]
    maxSelectionsPerCategory: int


class TasteRecommendRequest(BaseModel):
    descriptorIds: list[int] = Field(max_length=7)

    @field_validator("descriptorIds")
    @classmethod
    def descriptor_ids_must_be_unique(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("descriptorIds must not contain duplicates")
        return value
