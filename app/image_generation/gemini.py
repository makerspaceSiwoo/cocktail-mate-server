"""Gemini text-prompt adapter for the cocktail image batch."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import Any

from app.image_generation.core import (
    ImageGenerationSettings,
    ModelRateLimiter,
    PromptExpansion,
)
from app.image_generation.errors import (
    GenerationFatalError,
    GenerationItemError,
    GenerationQuotaError,
    GenerationRequestError,
)

logger = logging.getLogger("app.image_generation.gemini")


def _status_code(error: BaseException) -> int | None:
    for attribute in ("code", "status_code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


class GeminiGateway:
    def __init__(
        self,
        settings: ImageGenerationSettings,
        *,
        client: Any | None = None,
        limiter: ModelRateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.settings = settings
        self._client = client
        self._limiter = limiter or ModelRateLimiter(
            settings.gemini_min_request_interval_seconds
        )
        self._sleep = sleep
        self._random_uniform = random_uniform

    @property
    def client(self) -> Any:
        if self._client is None:
            from google import genai  # noqa: PLC0415
            from google.genai import types  # noqa: PLC0415

            self._client = genai.Client(
                api_key=self.settings.gemini_api_key.get_secret_value(),
                http_options=types.HttpOptions(
                    timeout=300_000,
                    retry_options=types.HttpRetryOptions(attempts=0),
                ),
            )
        return self._client

    def verify_model(self) -> None:
        model = self.settings.gemini_text_model
        try:
            self.client.models.get(model=model)
        except Exception as error:  # noqa: BLE001
            code = _status_code(error)
            if code in {400, 401, 403, 404}:
                raise GenerationFatalError(
                    f"Cannot access Gemini text model {model}: {error}"
                ) from error
            raise GenerationRequestError(
                f"Could not verify Gemini text model {model}: {error}"
            ) from error

    def expand_prompt(self, prompt: str) -> str:
        interaction = self._request(
            self.settings.gemini_text_model,
            lambda: self.client.interactions.create(
                model=self.settings.gemini_text_model,
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": PromptExpansion.model_json_schema(),
                },
            ),
        )
        output_text = getattr(interaction, "output_text", None)
        if not output_text:
            raise GenerationItemError("Gemini returned no expanded prompt")
        try:
            expansion = PromptExpansion.model_validate_json(output_text)
        except ValueError as error:
            raise GenerationItemError(
                f"Gemini returned an invalid structured prompt: {error}"
            ) from error
        cocktail_contents = expansion.cocktail_contents.strip()
        if len(cocktail_contents) < 80:
            raise GenerationItemError(
                "Gemini returned an underspecified cocktail contents prompt"
            )
        return cocktail_contents

    def _request(self, model: str, request: Callable[[], Any]) -> Any:
        max_attempts = self.settings.gemini_max_transient_retries + 1
        for attempt in range(1, max_attempts + 1):
            waited = self._limiter.wait(model)
            if waited:
                logger.info("Gemini %s rate limit wait: %.1fs", model, waited)
            try:
                return request()
            except Exception as error:  # noqa: BLE001
                code = _status_code(error)
                if code in {400, 401, 403, 404}:
                    raise GenerationFatalError(
                        f"Gemini request rejected ({code}): {error}"
                    ) from error
                if code == 429:
                    if attempt < max_attempts:
                        logger.warning(
                            "Gemini %s quota/rate response; retrying after the "
                            "model interval (%s/%s)",
                            model,
                            attempt,
                            max_attempts,
                        )
                        continue
                    raise GenerationQuotaError(
                        f"Gemini quota exhausted for {model}: {error}"
                    ) from error
                if code not in {408, 500, 502, 503, 504}:
                    raise GenerationItemError(
                        f"Gemini request failed for {model}: {error}"
                    ) from error
                if attempt == max_attempts:
                    raise GenerationItemError(
                        f"Gemini transient error persisted for {model}: {error}"
                    ) from error
                delay = min(2 ** (attempt - 1), 30) + self._random_uniform(0, 1)
                logger.warning(
                    "Gemini %s transient error (%s); retrying in %.1fs",
                    model,
                    code,
                    delay,
                )
                self._sleep(delay)

        raise AssertionError("unreachable")
