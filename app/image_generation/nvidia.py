"""NVIDIA FLUX.2 Klein image adapter for the cocktail image batch."""

from __future__ import annotations

import base64
import binascii
import logging
import random
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import requests

from app.image_generation.core import (
    NVIDIA_IMAGE_PROMPT_MAX_CHARS,
    ImageGenerationSettings,
    ModelRateLimiter,
)
from app.image_generation.errors import (
    GenerationFatalError,
    GenerationItemError,
    GenerationQuotaError,
)

logger = logging.getLogger("app.image_generation.nvidia")


def _response_detail(response: Any) -> str:
    try:
        body = response.json()
    except (TypeError, ValueError):
        body = getattr(response, "text", "")
    detail = str(body).replace("\n", " ").strip()
    return detail[:500] or "empty response"


def _decode_artifact(value: object) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise GenerationItemError("NVIDIA returned no base64 image artifact")
    encoded = value.strip()
    if encoded.startswith("data:"):
        _, separator, encoded = encoded.partition(",")
        if not separator:
            raise GenerationItemError("NVIDIA returned an invalid image data URI")
    try:
        return base64.b64decode(encoded, validate=True)
    except binascii.Error as error:
        raise GenerationItemError(
            "NVIDIA returned invalid base64 image data"
        ) from error


class NvidiaImageGateway:
    def __init__(
        self,
        settings: ImageGenerationSettings,
        *,
        session: Any | None = None,
        limiter: ModelRateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.settings = settings
        self._session = session or requests.Session()
        self._limiter = limiter or ModelRateLimiter(
            settings.nvidia_min_request_interval_seconds
        )
        self._sleep = sleep
        self._random_uniform = random_uniform

    @property
    def image_config(self) -> dict[str, object]:
        return {
            "provider": "nvidia",
            "endpoint": self.settings.nvidia_image_invoke_url,
            "cfg_scale": self.settings.nvidia_image_cfg_scale,
            "width": self.settings.nvidia_image_width,
            "height": self.settings.nvidia_image_height,
            "crop_width": self.settings.nvidia_image_crop_width,
            "crop_height": self.settings.nvidia_image_crop_height,
            "seed": self.settings.nvidia_image_seed,
            "steps": self.settings.nvidia_image_steps,
        }

    def verify_model(self) -> None:
        if not self.settings.nvidia_api_key.get_secret_value():
            raise GenerationFatalError("NVIDIA_API_KEY is required")
        endpoint = urlparse(self.settings.nvidia_image_invoke_url)
        if endpoint.scheme != "https" or not endpoint.netloc:
            raise GenerationFatalError(
                "NVIDIA_IMAGE_INVOKE_URL must be an absolute HTTPS URL"
            )
        if self.settings.nvidia_image_model != "black-forest-labs/flux.2-klein-4b":
            raise GenerationFatalError(
                "Only NVIDIA_IMAGE_MODEL=black-forest-labs/flux.2-klein-4b is supported"
            )
        if (
            self.settings.nvidia_image_width != 1184
            or self.settings.nvidia_image_height != 880
        ):
            raise GenerationFatalError(
                "Cocktail Mate FLUX.2 Klein generation must use 1184x880"
            )
        if (
            self.settings.nvidia_image_crop_width != 1172
            or self.settings.nvidia_image_crop_height != 879
        ):
            raise GenerationFatalError(
                "Cocktail Mate FLUX.2 Klein crop must use 1172x879"
            )
        if self.settings.nvidia_image_cfg_scale < 1:
            raise GenerationFatalError("NVIDIA_IMAGE_CFG_SCALE must be at least 1")
        if not 1 <= self.settings.nvidia_image_steps <= 4:
            raise GenerationFatalError("NVIDIA_IMAGE_STEPS must be between 1 and 4")
        if self.settings.nvidia_image_seed < 0:
            raise GenerationFatalError("NVIDIA_IMAGE_SEED must be non-negative")

    def generate_image(self, prompt: str) -> bytes:
        self.verify_model()
        if len(prompt) > NVIDIA_IMAGE_PROMPT_MAX_CHARS:
            raise GenerationItemError(
                "NVIDIA FLUX.2 Klein prompt exceeds 800 characters"
            )
        payload = {
            "prompt": prompt,
            "cfg_scale": self.settings.nvidia_image_cfg_scale,
            "width": self.settings.nvidia_image_width,
            "height": self.settings.nvidia_image_height,
            "seed": self.settings.nvidia_image_seed,
            "steps": self.settings.nvidia_image_steps,
        }
        max_attempts = self.settings.nvidia_max_transient_retries + 1

        for attempt in range(1, max_attempts + 1):
            waited = self._limiter.wait(self.settings.nvidia_image_model)
            if waited:
                logger.info("NVIDIA FLUX.2 Klein rate limit wait: %.1fs", waited)
            try:
                response = self._session.post(
                    self.settings.nvidia_image_invoke_url,
                    headers={
                        "Authorization": (
                            f"Bearer {self.settings.nvidia_api_key.get_secret_value()}"
                        ),
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=300,
                )
            except requests.RequestException as error:
                if attempt == max_attempts:
                    raise GenerationItemError(
                        f"NVIDIA network error persisted: {error}"
                    ) from error
                self._sleep(self._retry_delay(attempt))
                continue

            code = int(response.status_code)
            if code == 200:
                return self._image_from_response(response)
            detail = _response_detail(response)
            if code in {400, 401, 403, 404}:
                raise GenerationFatalError(
                    f"NVIDIA request rejected ({code}): {detail}"
                )
            if code == 422:
                if "content_filtered" in detail.lower():
                    raise GenerationItemError(
                        "NVIDIA safety filter blocked this prompt"
                    )
                raise GenerationFatalError(
                    f"NVIDIA request validation failed (422): {detail}"
                )
            if code == 429:
                if attempt == max_attempts:
                    raise GenerationQuotaError(
                        f"NVIDIA quota or rate limit persisted: {detail}"
                    )
                self._sleep(self._retry_delay(attempt, response))
                continue
            if code not in {408, 500, 502, 503, 504}:
                raise GenerationItemError(
                    f"NVIDIA image request failed ({code}): {detail}"
                )
            if attempt == max_attempts:
                raise GenerationItemError(
                    f"NVIDIA transient error persisted ({code}): {detail}"
                )
            self._sleep(self._retry_delay(attempt, response))

        raise AssertionError("unreachable")

    def _retry_delay(self, attempt: int, response: Any | None = None) -> float:
        retry_after = None
        if response is not None:
            value = getattr(response, "headers", {}).get("Retry-After")
            try:
                retry_after = float(value) if value is not None else None
            except (TypeError, ValueError):
                retry_after = None
        backoff = min(2 ** (attempt - 1), 30) + self._random_uniform(0, 1)
        return max(backoff, retry_after or 0)

    @staticmethod
    def _image_from_response(response: Any) -> bytes:
        try:
            body = response.json()
        except (TypeError, ValueError) as error:
            raise GenerationItemError("NVIDIA returned invalid JSON") from error
        artifacts = body.get("artifacts") if isinstance(body, dict) else None
        if not isinstance(artifacts, list) or not artifacts:
            raise GenerationItemError("NVIDIA returned no image artifacts")
        artifact = artifacts[0]
        if not isinstance(artifact, dict):
            raise GenerationItemError("NVIDIA returned an invalid image artifact")
        finish_reason = str(artifact.get("finish_reason") or "").upper()
        if finish_reason == "CONTENT_FILTERED":
            raise GenerationItemError("NVIDIA safety filter blocked this prompt")
        return _decode_artifact(artifact.get("base64"))
