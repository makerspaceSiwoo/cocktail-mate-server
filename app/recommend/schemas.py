"""추천(유사 칵테일 및 맛 선택) 요청/응답 스키마."""

from pydantic import BaseModel, Field, field_validator

from app.sensory_embedding import SENSORY_V2_REGISTRY

# 48축 감각 레지스트리의 카테고리 수. taste_descriptors.category 와 1:1 대응한다.
# 하드코딩하지 않는 이유: 축이 늘거나 카테고리가 쪼개지면 상한이 같이 따라와야 한다.
TASTE_CATEGORY_COUNT = len(SENSORY_V2_REGISTRY.category_counts)


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
    # 상한은 감각 축 카테고리 수와 같다. 서비스가 카테고리당 1개만 허용하므로
    # 이 값보다 작으면 "카테고리마다 하나씩" 고른 정상 요청이 422로 거부된다.
    # 48축 전환으로 taste_chemosensory 가 추가되며 7 -> 8 이 되었다.
    descriptorIds: list[int] = Field(max_length=TASTE_CATEGORY_COUNT)

    @field_validator("descriptorIds")
    @classmethod
    def descriptor_ids_must_be_unique(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("descriptorIds must not contain duplicates")
        return value
