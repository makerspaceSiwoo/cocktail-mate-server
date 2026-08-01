"""Gemini structured-output adapter for cocktail taste profiles."""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from typing import Any

from app.taste_generation.core import (
    FlavorProfile,
    ModelRateLimiter,
    TasteGenerationSettings,
)
from app.taste_generation.errors import (
    TasteGenerationFatalError,
    TasteGenerationItemError,
    TasteGenerationQuotaError,
)

logger = logging.getLogger("app.taste_generation.gemini")


def _status_code(error: BaseException) -> int | None:
    for attribute in ("code", "status_code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


class GeminiTasteGateway:
    def __init__(
        self,
        settings: TasteGenerationSettings,
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
                api_key=self.settings.gemini_api_key,
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
                raise TasteGenerationFatalError(
                    f"Cannot access Gemini text model {model}: {error}"
                ) from error
            raise TasteGenerationItemError(
                f"Could not verify Gemini text model {model}: {error}"
            ) from error

    def generate_profile(self, prompt: str) -> FlavorProfile:
        from google.genai import types  # noqa: PLC0415

        request_prompt = prompt
        max_attempts = self.settings.gemini_max_validation_retries + 1
        feedback_history: list[str] = []
        for attempt in range(1, max_attempts + 1):
            response = self._request(
                self.settings.gemini_text_model,
                lambda: self.client.models.generate_content(
                    model=self.settings.gemini_text_model,
                    contents=request_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        seed=42,
                        max_output_tokens=2048,
                        response_mime_type="application/json",
                        response_json_schema=FlavorProfile.model_json_schema(),
                    ),
                ),
            )
            output_text = getattr(response, "text", None)
            if not output_text:
                validation_error: ValueError = ValueError(
                    "Gemini returned no taste profile"
                )
            else:
                try:
                    return FlavorProfile.model_validate_json(output_text)
                except ValueError as error:
                    validation_error = error

            if attempt == max_attempts:
                raise TasteGenerationItemError(
                    f"Gemini returned an invalid taste profile: {validation_error}"
                ) from validation_error

            feedback = _validation_feedback(validation_error)
            if feedback not in feedback_history:
                feedback_history.append(feedback)
            cumulative_feedback = "; ".join(feedback_history)
            correction_guidance = _correction_guidance(cumulative_feedback, output_text)
            logger.warning(
                "Gemini taste profile failed validation; requesting correction "
                "(%s/%s): %s",
                attempt,
                max_attempts,
                feedback,
            )
            request_prompt = (
                f"{prompt}\n\nCorrection required: the previous JSON response was "
                f"rejected because {cumulative_feedback}. Return a completely revised "
                f"JSON response for correction attempt {attempt + 1} of "
                f"{max_attempts}; do not repeat the previous rejected JSON. "
                "Re-check every embedding_text prohibition and keep the "
                f"machine facets truthful. {correction_guidance}"
            )

        raise AssertionError("unreachable")

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
                    raise TasteGenerationFatalError(
                        f"Gemini request rejected ({code}): {error}"
                    ) from error
                if code == 429:
                    if attempt < max_attempts:
                        logger.warning(
                            "Gemini %s rate/quota response; retrying (%s/%s)",
                            model,
                            attempt,
                            max_attempts,
                        )
                        continue
                    raise TasteGenerationQuotaError(
                        f"Gemini quota exhausted for {model}: {error}"
                    ) from error
                if code not in {408, 500, 502, 503, 504}:
                    raise TasteGenerationItemError(
                        f"Gemini request failed for {model}: {error}"
                    ) from error
                if attempt == max_attempts:
                    raise TasteGenerationItemError(
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


def _validation_feedback(error: ValueError) -> str:
    errors_method = getattr(error, "errors", None)
    if callable(errors_method):
        details = errors_method(include_url=False, include_input=False)
        messages = [str(detail.get("msg", "validation failed")) for detail in details]
        return "; ".join(messages[:3])
    return str(error).splitlines()[0]


def _correction_guidance(feedback: str, output_text: str | None) -> str:
    guidance_by_error = (
        (
            "at least 80 characters",
            "Expand embedding_text to at least 80 Korean characters with another "
            "truthful sentence about aroma, palate, mouthfeel, or finish.",
        ),
        (
            "two to four sentences",
            "Rewrite embedding_text as exactly two to four complete Korean sentences.",
        ),
        (
            "negative or level wording",
            "Remove every negative or absence expression, including 없음, 없다, "
            "없는, 없이, 않다, 낮은, and 약한; omit that axis or replace it with "
            "a truthful positive sensory descriptor. The Korean substring 없 is "
            "forbidden even inside otherwise positive phrases such as 부담 없이, "
            "군더더기 없이, 막힘없이, or 거침없이. Search embedding_text for "
            "every occurrence of 없 and rewrite the entire surrounding phrase.",
        ),
        (
            "aroma_intensity must agree",
            "Make the aroma facets internally consistent: when fruit_aromas or "
            "other_aromas contains any note, aroma_intensity must be VERY_LOW, LOW, "
            "MEDIUM, HIGH, or VERY_HIGH and must never be NONE. When both aroma "
            "arrays are empty, aroma_intensity must be NONE. Change either the "
            "aroma_intensity value or the aroma arrays so this exact relationship "
            "holds.",
        ),
        (
            "without labels or numbers",
            "Use plain natural sentences with no headings, separators, scores, "
            "digits, percentages, or key=value formatting.",
        ),
        (
            "non-sensory term",
            "Remove every visual, historical, popularity, occasion, and presentation "
            "reference; retain sensory taste, aroma, texture, temperature, and finish.",
        ),
        (
            "glass/presentation wording",
            "Remove every mention of a glass, serving vessel, garnish, or presentation.",
        ),
        (
            "low sweetness text",
            "Remove 단맛, 달콤함, 당도, 스위트, and 디저트 completely; include "
            "one accurate positive descriptor from 드라이함, 깔끔함, 산뜻함, "
            "씁쓸함, or 담백함.",
        ),
        (
            "low acidity text",
            "Remove 산미, 새콤함, and 시큼함 completely. Omit acidity or describe "
            "the resulting mouthfeel positively without naming the acidity axis.",
        ),
        (
            "light body requires",
            "Include at least one exact positive concept using 가벼운, 라이트한, "
            "산뜻한, 청량한, or 경쾌한.",
        ),
        (
            "full body requires",
            "Include at least one exact positive concept using 밀도감, 묵직함, "
            "리치함, 실키함, 벨벳감, or 점성감.",
        ),
        (
            "carbonation must be omitted",
            "Remove 탄산, 기포, 버블, and every carbonation-related expression.",
        ),
        (
            "creaminess must be omitted",
            "Remove 크리미 and every explicit creaminess expression.",
        ),
        (
            "omit alcohol tokens",
            "Remove 알코올, 스피릿, 에탄올, 도수, 열감, and 온기 completely.",
        ),
    )
    guidance = [
        guidance for marker, guidance in guidance_by_error if marker in feedback
    ]
    if "selected flavor note" in feedback and output_text:
        try:
            payload = json.loads(output_text)
        except (TypeError, ValueError):
            payload = {}
        selected_notes: list[str] = []
        for field_name in (
            "fruit_aromas",
            "other_aromas",
            "palate_fruit_notes",
            "palate_other_notes",
        ):
            values = payload.get(field_name, [])
            if isinstance(values, list):
                selected_notes.extend(str(value) for value in values)
        unique_notes = list(dict.fromkeys(selected_notes))
        if unique_notes:
            guidance.append(
                "Include at least one of these exact selected sensory notes in "
                f"embedding_text: {', '.join(unique_notes)}."
            )
    return " ".join(guidance)
