from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.image_generation.core import ImageGenerationSettings, ModelRateLimiter
from app.image_generation.errors import (
    GenerationFatalError,
    GenerationQuotaError,
)
from app.image_generation.gemini import GeminiGateway


class FakeInteractions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.interactions = FakeInteractions(responses)


class FakeApiError(Exception):
    def __init__(self, code: int, message: str | None = None) -> None:
        super().__init__(message or f"api error {code}")
        self.code = code


def _gateway(
    responses: list[object],
    *,
    retries: int = 0,
    sleeps: list[float] | None = None,
) -> tuple[GeminiGateway, FakeClient]:
    client = FakeClient(responses)
    settings = ImageGenerationSettings(
        gemini_api_key="test-key",
        gemini_min_request_interval_seconds=0,
        gemini_max_transient_retries=retries,
    )
    sleep_calls = sleeps if sleeps is not None else []
    gateway = GeminiGateway(
        settings,
        client=client,
        limiter=ModelRateLimiter(0),
        sleep=sleep_calls.append,
        random_uniform=lambda _start, _end: 0,
    )
    return gateway, client


def test_expand_prompt_uses_structured_json_response_format() -> None:
    cocktail_contents = (
        "A clear ruby cocktail with a bright lemon twist, large transparent ice "
        "cubes, fine condensation droplets, and a subtle lighter layer near the "
        "surface with no cream or foam."
    )
    gateway, client = _gateway(
        [SimpleNamespace(output_text=f'{{"cocktail_contents": "{cocktail_contents}"}}')]
    )

    assert gateway.expand_prompt("facts") == cocktail_contents
    request = client.interactions.requests[0]
    assert request["model"] == "gemini-3.5-flash-lite"
    assert request["response_format"]["mime_type"] == "application/json"  # type: ignore[index]


def test_quota_error_is_exposed_to_the_pipeline() -> None:
    gateway, _client = _gateway([FakeApiError(429)])

    with pytest.raises(GenerationQuotaError):
        gateway.expand_prompt("facts")


def test_transient_errors_use_bounded_backoff() -> None:
    sleeps: list[float] = []
    gateway, _client = _gateway(
        [
            FakeApiError(503),
            SimpleNamespace(
                output_text='{"cocktail_contents": "A sufficiently detailed ruby '
                "cocktail with clear ice cubes, a citrus layer, realistic "
                'condensation, and a bright orange garnish beside the rim."}'
            ),
        ],
        retries=1,
        sleeps=sleeps,
    )

    assert "ruby cocktail" in gateway.expand_prompt("facts")
    assert sleeps == [1]


def test_auth_error_is_fatal() -> None:
    gateway, client = _gateway([FakeApiError(401)], retries=3)

    with pytest.raises(GenerationFatalError, match="rejected"):
        gateway.expand_prompt("facts")

    assert len(client.interactions.requests) == 1
