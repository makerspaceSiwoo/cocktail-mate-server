from __future__ import annotations

import json
from types import SimpleNamespace

from app.taste_generation.core import ModelRateLimiter, TasteGenerationSettings
from app.taste_generation.gemini import GeminiTasteGateway


class FakeModels:
    def __init__(self, output_text: str | list[str]) -> None:
        self.output_texts = (
            list(output_text) if isinstance(output_text, list) else [output_text]
        )
        self.requests: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return SimpleNamespace(text=self.output_texts.pop(0))


class FakeClient:
    def __init__(self, output_text: str | list[str]) -> None:
        self.models = FakeModels(output_text)


def _profile_json() -> str:
    return json.dumps(
        {
            "sweetness": "LOW",
            "acidity": "HIGH",
            "bitterness": "MEDIUM",
            "salinity": "NONE",
            "umami": "NONE",
            "aroma_intensity": "HIGH",
            "fruit_aromas": ["자몽"],
            "other_aromas": ["주니퍼"],
            "palate_fruit_notes": ["자몽"],
            "palate_other_notes": ["주니퍼"],
            "body": "LIGHT",
            "carbonation": "NONE",
            "creaminess": "NONE",
            "mouthfeel": ["산뜻함", "드라이함"],
            "serving_temperature": "COLD",
            "alcohol_presence": "HIGH",
            "alcohol_character": "DISTINCT",
            "finish_length": "LONG",
            "finish_character": ["쌉쌀함", "드라이함"],
            "embedding_text": (
                "주니퍼 향과 자몽의 선명한 산미, 허브의 씁쓸한 풍미가 "
                "또렷하게 펼쳐진다. 라이트하고 산뜻한 목넘김 뒤로 깔끔한 "
                "드라이함과 선명한 스피릿 열감이 길게 이어진다."
            ),
        },
        ensure_ascii=False,
    )


def test_gateway_requests_strict_json_schema() -> None:
    client = FakeClient(_profile_json())
    settings = TasteGenerationSettings(
        gemini_api_key="test",
        gemini_min_request_interval_seconds=0,
    )
    gateway = GeminiTasteGateway(
        settings,
        client=client,
        limiter=ModelRateLimiter(0),
    )

    profile = gateway.generate_profile("facts")

    assert profile.fruit_aromas[0].value == "자몽"
    request = client.models.requests[0]
    assert request["model"] == "gemini-3.5-flash-lite"
    config = request["config"]
    assert config.response_mime_type == "application/json"  # type: ignore[union-attr]
    schema = config.response_json_schema  # type: ignore[union-attr]
    assert schema["additionalProperties"] is False
    assert "fruit_aromas" in schema["properties"]
    assert "embedding_text" in schema["properties"]


def test_gateway_requests_one_correction_after_embedding_validation_failure() -> None:
    invalid = json.loads(_profile_json())
    invalid["embedding_text"] = (
        "주니퍼와 자몽 향이 또렷하게 펼쳐지고 은은한 단맛과 허브의 풍미가 "
        "중심을 잡는다. 라이트하고 산뜻한 목넘김 뒤로 깔끔한 드라이함과 "
        "선명한 스피릿 열감이 길게 이어진다."
    )
    client = FakeClient(
        [
            json.dumps(invalid, ensure_ascii=False),
            _profile_json(),
        ]
    )
    gateway = GeminiTasteGateway(
        TasteGenerationSettings(
            gemini_api_key="test",
            gemini_min_request_interval_seconds=0,
            gemini_max_validation_retries=1,
        ),
        client=client,
        limiter=ModelRateLimiter(0),
    )

    profile = gateway.generate_profile("facts")

    assert "드라이함" in profile.embedding_text
    assert len(client.models.requests) == 2
    correction_prompt = client.models.requests[1]["contents"]
    assert "previous JSON response was rejected" in correction_prompt
    assert "low sweetness text" in correction_prompt
    assert "Remove 단맛, 달콤함, 당도" in correction_prompt


def test_correction_prompt_names_selected_flavor_notes() -> None:
    invalid = json.loads(_profile_json())
    invalid["embedding_text"] = (
        "선명한 시트러스 풍미와 허브의 쌉쌀한 인상이 또렷하게 펼쳐진다. "
        "라이트하고 산뜻한 목넘김 뒤로 깔끔한 드라이함과 선명한 스피릿 "
        "열감이 길고 경쾌하게 이어진다."
    )
    client = FakeClient(
        [
            json.dumps(invalid, ensure_ascii=False),
            _profile_json(),
        ]
    )
    gateway = GeminiTasteGateway(
        TasteGenerationSettings(
            gemini_api_key="test",
            gemini_min_request_interval_seconds=0,
            gemini_max_validation_retries=1,
        ),
        client=client,
        limiter=ModelRateLimiter(0),
    )

    gateway.generate_profile("facts")

    correction_prompt = client.models.requests[1]["contents"]
    assert "exact selected sensory notes" in correction_prompt
    assert "자몽, 주니퍼" in correction_prompt


def test_correction_prompt_keeps_feedback_from_previous_attempts() -> None:
    missing_light_body = json.loads(_profile_json())
    missing_light_body["embedding_text"] = missing_light_body[
        "embedding_text"
    ].replace("라이트하고 산뜻한", "매끄러운")
    missing_alcohol = json.loads(_profile_json())
    missing_alcohol["embedding_text"] = missing_alcohol[
        "embedding_text"
    ].replace("선명한 스피릿 열감", "깔끔한 풍미")
    client = FakeClient(
        [
            json.dumps(missing_light_body, ensure_ascii=False),
            json.dumps(missing_alcohol, ensure_ascii=False),
            _profile_json(),
        ]
    )
    gateway = GeminiTasteGateway(
        TasteGenerationSettings(
            gemini_api_key="test",
            gemini_min_request_interval_seconds=0,
            gemini_max_validation_retries=2,
        ),
        client=client,
        limiter=ModelRateLimiter(0),
    )

    gateway.generate_profile("facts")

    final_correction_prompt = client.models.requests[2]["contents"]
    assert "light body requires" in final_correction_prompt
    assert "dominant alcohol perception" in final_correction_prompt
    assert "Include at least one exact positive concept using 가벼운" in (
        final_correction_prompt
    )


def test_repeated_feedback_changes_correction_attempt_prompt() -> None:
    invalid = json.loads(_profile_json())
    invalid["aroma_intensity"] = "NONE"
    client = FakeClient(
        [
            json.dumps(invalid, ensure_ascii=False),
            json.dumps(invalid, ensure_ascii=False),
            _profile_json(),
        ]
    )
    gateway = GeminiTasteGateway(
        TasteGenerationSettings(
            gemini_api_key="test",
            gemini_min_request_interval_seconds=0,
            gemini_max_validation_retries=2,
        ),
        client=client,
        limiter=ModelRateLimiter(0),
    )

    gateway.generate_profile("facts")

    first_correction = client.models.requests[1]["contents"]
    second_correction = client.models.requests[2]["contents"]
    assert "correction attempt 2 of 3" in first_correction
    assert "correction attempt 3 of 3" in second_correction
    assert "aroma_intensity must be VERY_LOW" in first_correction


def test_negative_wording_guidance_bans_positive_phrases_containing_eops() -> None:
    invalid = json.loads(_profile_json())
    invalid["embedding_text"] = invalid["embedding_text"].replace(
        "라이트하고 산뜻한 목넘김",
        "부담 없이 산뜻한 목넘김",
    )
    client = FakeClient([json.dumps(invalid, ensure_ascii=False), _profile_json()])
    gateway = GeminiTasteGateway(
        TasteGenerationSettings(
            gemini_api_key="test",
            gemini_min_request_interval_seconds=0,
            gemini_max_validation_retries=1,
        ),
        client=client,
        limiter=ModelRateLimiter(0),
    )

    gateway.generate_profile("facts")

    correction_prompt = client.models.requests[1]["contents"]
    assert "substring 없 is forbidden" in correction_prompt
    assert "부담 없이" in correction_prompt
