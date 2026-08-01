from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.recommend.schemas import RecommendItem, TasteRecommendRequest
from app.recommend.router import router as recommend_router
from app.recommend.service import RecommendService
from app.taste_query.model import get_taste_query_model
from app.taste_query.vocabulary import (
    CATEGORY_ORDER,
    DESCRIPTORS,
    extract_descriptor_codes,
)


def test_controlled_taste_vocabulary_is_unique_and_extracts_prompt_terms() -> None:
    assert len(DESCRIPTORS) == 43
    assert len({descriptor.code for descriptor in DESCRIPTORS}) == len(DESCRIPTORS)
    assert len({descriptor.label_ko for descriptor in DESCRIPTORS}) == len(DESCRIPTORS)
    codes = extract_descriptor_codes(
        "선명한 레몬 산미와 드라이한 질감 뒤로 부드러운 온기가 이어진다."
    )
    assert "fruit.citrus" in codes
    assert "mouthfeel.dry" in codes
    assert "alcohol.soft" in codes
    categories = [
        next(item.category for item in DESCRIPTORS if item.code == code)
        for code in codes
    ]
    assert len(categories) == len(set(categories))
    assert set(categories).issubset(CATEGORY_ORDER)
    assert not any("흙" in alias for item in DESCRIPTORS for alias in item.aliases)


def test_trained_taste_query_model_outputs_unit_32d_vector() -> None:
    model = get_taste_query_model()
    vector = model.encode(["fruit.citrus", "mouthfeel.dry", "alcohol.strong"])

    assert len(model.supported_codes) == 43
    assert vector.shape == (32,)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)


def test_taste_recommend_request_rejects_duplicate_ids() -> None:
    assert TasteRecommendRequest(descriptorIds=[]).descriptorIds == []
    with pytest.raises(ValidationError):
        TasteRecommendRequest(descriptorIds=[3, 3])
    with pytest.raises(ValidationError):
        TasteRecommendRequest(descriptorIds=list(range(1, 9)))


def test_recommend_item_includes_description_and_image_url() -> None:
    item = RecommendItem.model_validate(
        {
            "id": 7,
            "name": "테스트",
            "description": "테스트 칵테일 설명",
            "imageUrl": "https://example.com/test.jpg",
            "similarity": 0.9,
        }
    )

    assert item.description == "테스트 칵테일 설명"
    assert item.imageUrl == "https://example.com/test.jpg"


def test_flavor_recommend_route_is_registered() -> None:
    routes = {route.path: route.methods for route in recommend_router.routes}

    assert routes["/flavor/recommend"] == {"POST"}
    assert "/cocktail/recommend/by-taste" not in routes


class _FakeModel:
    supported_codes = frozenset({"fruit.citrus", "mouthfeel.dry"})

    @staticmethod
    def encode(codes):
        assert codes == ["fruit.citrus", "mouthfeel.dry"]
        vector = np.zeros(32, dtype=np.float32)
        vector[0] = 1.0
        return vector


class _FakeRepository:
    def __init__(self, *, missing=False, same_category=False):
        self.missing = missing
        self.same_category = same_category
        self.query = None

    def active_taste_descriptors_by_ids(self, _db, _ids):
        values = [
            SimpleNamespace(id=1, code="fruit.citrus", category="fruit"),
            SimpleNamespace(
                id=2,
                code="mouthfeel.dry",
                category="fruit" if self.same_category else "mouthfeel",
            ),
        ]
        return values[:1] if self.missing else values

    def nearest_for_virtual_taste(self, _db, embedding, limit):
        self.query = embedding
        assert limit == 5
        return [
            {
                "id": 7,
                "name": "테스트",
                "description": "테스트 칵테일 설명",
                "imageUrl": "https://example.com/test.jpg",
                "similarity": 0.9,
            }
        ]

    def random_cocktails(self, _db, limit):
        assert limit == 5
        return [
            {
                "id": index,
                "name": f"랜덤 {index}",
                "description": f"랜덤 칵테일 {index} 설명",
                "imageUrl": f"https://example.com/random-{index}.jpg",
                "similarity": 0.0,
            }
            for index in range(1, limit + 1)
        ]


def test_service_encodes_taste_codes_and_runs_ann() -> None:
    repository = _FakeRepository()
    service = RecommendService(repository, taste_model_loader=lambda: _FakeModel())

    result = service.recommend_by_taste(object(), [1, 2])

    assert result[0]["id"] == 7
    assert result[0]["description"] == "테스트 칵테일 설명"
    assert result[0]["imageUrl"] == "https://example.com/test.jpg"
    assert len(repository.query) == 32


def test_service_returns_random_cocktails_for_empty_taste_selection() -> None:
    repository = _FakeRepository()
    service = RecommendService(
        repository,
        taste_model_loader=lambda: pytest.fail("empty selection must not load model"),
    )

    result = service.recommend_by_taste(object(), [])

    assert len(result) == 5
    assert all(item["similarity"] == 0.0 for item in result)
    assert all(item["description"] for item in result)
    assert all(item["imageUrl"] for item in result)


def test_service_rejects_unknown_or_same_category_descriptors() -> None:
    missing_service = RecommendService(
        _FakeRepository(missing=True),
        taste_model_loader=lambda: _FakeModel(),
    )
    with pytest.raises(HTTPException) as missing_error:
        missing_service.recommend_by_taste(object(), [1, 2])
    assert missing_error.value.status_code == 422
    assert missing_error.value.detail["code"] == "unknown_taste_descriptors"

    same_category_service = RecommendService(
        _FakeRepository(same_category=True),
        taste_model_loader=lambda: _FakeModel(),
    )
    with pytest.raises(HTTPException) as category_error:
        same_category_service.recommend_by_taste(object(), [1, 2])
    assert category_error.value.status_code == 422
    assert category_error.value.detail["code"] == "multiple_descriptors_in_category"
