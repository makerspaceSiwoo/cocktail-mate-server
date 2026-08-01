from __future__ import annotations

import base64
from typing import Any

import pytest

from app.image_generation.core import ImageGenerationSettings, ModelRateLimiter
from app.image_generation.errors import (
    GenerationFatalError,
    GenerationItemError,
    GenerationQuotaError,
)
from app.image_generation.nvidia import NvidiaImageGateway


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        body: object,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = str(body)

    def json(self) -> object:
        return self._body


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"url": url, **kwargs})
        return self.responses.pop(0)


def _gateway(
    responses: list[FakeResponse],
    *,
    retries: int = 0,
    sleeps: list[float] | None = None,
) -> tuple[NvidiaImageGateway, FakeSession]:
    settings = ImageGenerationSettings(
        gemini_api_key="gemini-test",
        nvidia_api_key="nvidia-test",
        nvidia_min_request_interval_seconds=0,
        nvidia_max_transient_retries=retries,
    )
    session = FakeSession(responses)
    sleep_calls = sleeps if sleeps is not None else []
    gateway = NvidiaImageGateway(
        settings,
        session=session,
        limiter=ModelRateLimiter(0),
        sleep=sleep_calls.append,
        random_uniform=lambda _start, _end: 0,
    )
    return gateway, session


def test_flux2_klein_request_and_artifact_decode() -> None:
    raw_image = b"\x89PNG\r\n\x1a\nmock"
    gateway, session = _gateway(
        [
            FakeResponse(
                200,
                {
                    "artifacts": [
                        {"base64": base64.b64encode(raw_image).decode("ascii")}
                    ]
                },
            )
        ]
    )

    assert gateway.generate_image("four-part cocktail prompt") == raw_image
    request = session.requests[0]
    assert request["url"].endswith("/black-forest-labs/flux.2-klein-4b")
    assert request["headers"]["Authorization"] == "Bearer nvidia-test"
    assert request["headers"]["Accept"] == "application/json"
    assert request["json"] == {
        "prompt": "four-part cocktail prompt",
        "cfg_scale": 1.0,
        "width": 1184,
        "height": 880,
        "seed": 42,
        "steps": 4,
    }
    assert request["timeout"] == 300


def test_rate_limit_honors_retry_after_then_succeeds() -> None:
    sleeps: list[float] = []
    gateway, session = _gateway(
        [
            FakeResponse(429, {"detail": "slow down"}, headers={"Retry-After": "7"}),
            FakeResponse(
                200,
                {"artifacts": [{"base64": base64.b64encode(b"image").decode("ascii")}]},
            ),
        ],
        retries=1,
        sleeps=sleeps,
    )

    assert gateway.generate_image("prompt") == b"image"
    assert sleeps == [7]
    assert len(session.requests) == 2


def test_persistent_rate_limit_is_checkpointable_quota_error() -> None:
    gateway, _session = _gateway([FakeResponse(429, {"detail": "credits exhausted"})])

    with pytest.raises(GenerationQuotaError, match="credits exhausted"):
        gateway.generate_image("prompt")


def test_auth_error_is_fatal_without_retry() -> None:
    gateway, session = _gateway(
        [FakeResponse(401, {"detail": "invalid token"})],
        retries=3,
    )

    with pytest.raises(GenerationFatalError, match="401"):
        gateway.generate_image("prompt")

    assert len(session.requests) == 1


def test_prompt_over_800_chars_fails_before_request() -> None:
    gateway, session = _gateway([])

    with pytest.raises(GenerationItemError, match="exceeds 800"):
        gateway.generate_image("x" * 801)

    assert session.requests == []


def test_content_filtered_artifact_is_an_item_failure() -> None:
    gateway, _session = _gateway(
        [
            FakeResponse(
                200,
                {"artifacts": [{"finish_reason": "CONTENT_FILTERED", "base64": ""}]},
            )
        ]
    )

    with pytest.raises(GenerationItemError, match="safety filter"):
        gateway.generate_image("prompt")


def test_flux2_klein_rejects_wrong_generation_size() -> None:
    settings = ImageGenerationSettings(
        gemini_api_key="gemini-test",
        nvidia_api_key="nvidia-test",
        nvidia_image_height=1024,
    )
    gateway = NvidiaImageGateway(settings, session=FakeSession([]))

    with pytest.raises(GenerationFatalError, match="1184x880"):
        gateway.verify_model()
