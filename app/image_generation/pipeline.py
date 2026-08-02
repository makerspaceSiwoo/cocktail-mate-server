"""DB-to-CSV pipeline for reusable Gemini cocktail image prompts."""

from __future__ import annotations

import csv
import io
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cocktail_mate_db.models import Cocktail
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.image_generation.core import (
    CocktailSource,
    GenerationState,
    ImageGenerationSettings,
    PromptParts,
    StateStore,
    atomic_write,
    build_contents_expansion_prompt,
    build_contents_fingerprint,
    build_final_prompt,
    build_input_fingerprint,
    load_prompt_catalog,
)
from app.image_generation.errors import (
    GenerationFatalError,
    GenerationItemError,
    GenerationQuotaError,
)

logger = logging.getLogger("app.image_generation.pipeline")
PROMPT_EXPORT_IMAGE_MODEL = "prompt-export"


class PromptGateway(Protocol):
    def expand_prompt(self, prompt: str) -> str: ...


@dataclass(slots=True)
class PromptExportSummary:
    exported: int = 0
    cache_hits: int = 0
    gemini_generated: int = 0
    skipped: int = 0
    failed: int = 0


class CocktailImagePipeline:
    """Generate image prompts; image generation is handled by Gemini Batch."""

    def __init__(
        self,
        settings: ImageGenerationSettings,
        session_factory: sessionmaker[Session],
        prompt_gateway: PromptGateway,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.prompt_gateway = prompt_gateway
        self.state_store = StateStore(settings.cocktail_image_state_dir)
        self.prompt_catalog = load_prompt_catalog(settings)

    def export_prompts(
        self,
        output_path: Path,
        *,
        limit: int | None = None,
        cocktail_ids: tuple[int, ...] = (),
        cached_only: bool = False,
    ) -> PromptExportSummary:
        if limit is not None and limit <= 0:
            raise ValueError("--limit must be greater than zero")

        rows: list[dict[str, object]] = []
        summary = PromptExportSummary()
        for source in self._load_sources(cocktail_ids):
            source_error = self._source_error(source)
            if source_error:
                logger.error("[%s] prompt export skipped: %s", source.id, source_error)
                summary.skipped += 1
                continue

            parts = self.prompt_catalog.parts_for(source)
            contents_fingerprint = build_contents_fingerprint(
                source,
                text_model=self.settings.gemini_text_model,
            )
            state = self.state_store.load(source.id)
            contents = (
                state.cocktail_contents_prompt
                if state is not None
                and state.cocktail_contents_prompt
                and state.contents_fingerprint == contents_fingerprint
                else None
            )
            if contents is not None:
                summary.cache_hits += 1
            elif cached_only:
                summary.skipped += 1
                continue
            else:
                try:
                    contents = self.prompt_gateway.expand_prompt(
                        build_contents_expansion_prompt(source)
                    )
                except (GenerationQuotaError, GenerationFatalError):
                    raise
                except GenerationItemError as error:
                    logger.error("[%s] Gemini prompt failed: %s", source.id, error)
                    summary.failed += 1
                    continue
                summary.gemini_generated += 1

            final_prompt = build_final_prompt(contents, parts)
            self.state_store.save(
                GenerationState(
                    cocktail_id=source.id,
                    input_fingerprint=self._fingerprint(source, parts),
                    contents_fingerprint=contents_fingerprint,
                    text_model=self.settings.gemini_text_model,
                    image_model=PROMPT_EXPORT_IMAGE_MODEL,
                    status="prompt_ready",
                    cocktail_contents_prompt=contents,
                    glass_prompt=parts.glass,
                    background_prompt=parts.background,
                    composition_prompt=parts.composition,
                    expanded_prompt=final_prompt,
                    attempts=state.attempts if state else 0,
                )
            )
            rows.append(
                {
                    "id": source.id,
                    "cocktail_name": source.name,
                    "cocktail_name_en": source.name_en or "",
                    "image_filename": f"cocktail-{source.id}.png",
                    "final_image_prompt": final_prompt,
                }
            )
            summary.exported += 1
            if limit is not None and summary.exported >= limit:
                break

        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "id",
                "cocktail_name",
                "cocktail_name_en",
                "image_filename",
                "final_image_prompt",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
        atomic_write(output_path, output.getvalue().encode("utf-8-sig"))
        return summary

    def _load_sources(self, cocktail_ids: Iterable[int]) -> Iterator[CocktailSource]:
        statement = select(
            Cocktail.id,
            Cocktail.name,
            Cocktail.name_en,
            Cocktail.glass,
            Cocktail.base_tag,
            Cocktail.recipe,
        ).order_by(Cocktail.id)
        ids = tuple(cocktail_ids)
        if ids:
            statement = statement.where(Cocktail.id.in_(ids))

        with self.session_factory() as session:
            for row in session.execute(statement):
                yield CocktailSource(
                    id=int(row.id),
                    name=row.name,
                    name_en=str(row.name_en) if row.name_en else None,
                    glass=row.glass,
                    base_tag=row.base_tag,
                    recipe=tuple(row.recipe or ()),
                )

    def _fingerprint(self, source: CocktailSource, parts: PromptParts) -> str:
        return build_input_fingerprint(
            source,
            parts,
            text_model=self.settings.gemini_text_model,
            image_model=PROMPT_EXPORT_IMAGE_MODEL,
        )

    def _source_error(self, source: CocktailSource) -> str | None:
        if source.validation_error:
            return source.validation_error
        try:
            self.prompt_catalog.parts_for(source)
        except ValueError as error:
            return str(error)
        return None
