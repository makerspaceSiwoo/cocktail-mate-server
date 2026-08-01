"""Resumable DB-to-provider cocktail image generation pipeline."""

from __future__ import annotations

import csv
import io
import logging
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cocktail_mate_db.models import Cocktail
from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session, sessionmaker

from app.image_generation.core import (
    MAIN_IMAGE_SIZE,
    THUMBNAIL_IMAGE_SIZE,
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
    build_main_url,
    create_image_variants,
    image_filenames,
    load_prompt_catalog,
    validate_webp,
)
from app.image_generation.errors import (
    GenerationFatalError,
    GenerationItemError,
    GenerationQuotaError,
)

logger = logging.getLogger("app.image_generation.pipeline")

ADVISORY_LOCK_KEY = 2026072901


class AlreadyRunningError(RuntimeError):
    """Another worker currently owns the DB advisory lock."""


class PromptGateway(Protocol):
    def verify_model(self) -> None: ...

    def expand_prompt(self, prompt: str) -> str: ...


class ImageGateway(Protocol):
    @property
    def image_config(self) -> Mapping[str, object]: ...

    def verify_model(self) -> None: ...

    def generate_image(self, prompt: str) -> bytes: ...


@dataclass(slots=True)
class RunOptions:
    limit: int | None = None
    cocktail_ids: tuple[int, ...] = ()
    force: bool = False
    retry_failed: bool = False
    dry_run: bool = False
    no_db_update: bool = False


@dataclass(slots=True)
class RunSummary:
    generated: int = 0
    repaired: int = 0
    skipped: int = 0
    failed: int = 0
    invalid: int = 0
    dry_run: int = 0
    staged: int = 0


@dataclass(slots=True)
class PromptExportSummary:
    exported: int = 0
    cache_hits: int = 0
    gemini_generated: int = 0
    skipped: int = 0
    failed: int = 0


@contextmanager
def advisory_lock(session_factory: sessionmaker[Session]) -> Iterator[None]:
    """Hold a PostgreSQL session-level advisory lock for the full batch."""

    bind = session_factory.kw.get("bind")
    if bind is None:
        raise RuntimeError("Session factory has no bind")
    with bind.connect() as connection:
        acquired = bool(
            connection.execute(
                text("SELECT pg_try_advisory_lock(:lock_key)"),
                {"lock_key": ADVISORY_LOCK_KEY},
            ).scalar()
        )
        if not acquired:
            raise AlreadyRunningError("Another cocktail image worker is running")
        try:
            yield
        finally:
            connection.execute(
                text("SELECT pg_advisory_unlock(:lock_key)"),
                {"lock_key": ADVISORY_LOCK_KEY},
            )


class CocktailImagePipeline:
    def __init__(
        self,
        settings: ImageGenerationSettings,
        session_factory: sessionmaker[Session],
        prompt_gateway: PromptGateway,
        image_gateway: ImageGateway,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.prompt_gateway = prompt_gateway
        self.image_gateway = image_gateway
        self.state_store = StateStore(settings.cocktail_image_state_dir)
        self.prompt_catalog = load_prompt_catalog(settings)

    def preflight(self) -> dict[str, object]:
        self._verify_directories()
        self.prompt_gateway.verify_model()
        self.image_gateway.verify_model()
        sources = list(self._load_sources(()))
        invalid = sum(self._source_error(source) is not None for source in sources)
        status_counts = Counter(state.status for state in self.state_store.all_states())
        return {
            "cocktails": len(sources),
            "valid": len(sources) - invalid,
            "invalid": invalid,
            "states": dict(status_counts),
            "text_model": self.settings.gemini_text_model,
            "image_provider": "nvidia",
            "image_model": self.settings.nvidia_image_model,
            "minimum_interval_seconds": {
                "gemini": self.settings.gemini_min_request_interval_seconds,
                "nvidia": self.settings.nvidia_min_request_interval_seconds,
            },
            "estimated_minimum_hours": round(
                (
                    len(sources)
                    * max(
                        self.settings.gemini_min_request_interval_seconds,
                        self.settings.nvidia_min_request_interval_seconds,
                    )
                )
                / 3600,
                2,
            ),
            "output_dir": str(self.settings.cocktail_image_output_dir),
            "state_dir": str(self.settings.cocktail_image_state_dir),
            "configured_backgrounds": {
                base_tag: {
                    "name": background.display_name,
                    "color": background.color_hex,
                }
                for base_tag, background in self.prompt_catalog.backgrounds.items()
            },
            "configured_glasses": len(self.prompt_catalog.glasses),
        }

    def status(self) -> dict[str, object]:
        states = self.state_store.all_states()
        status_counts = Counter(state.status for state in states)
        with self.session_factory() as session:
            total = session.scalar(select(func.count(Cocktail.id)))
        tracked = len(states)
        return {
            "cocktails": int(total or 0),
            "tracked": tracked,
            "untracked": max(0, int(total or 0) - tracked),
            "states": dict(status_counts),
            "last_errors": [
                {
                    "cocktail_id": state.cocktail_id,
                    "status": state.status,
                    "error": state.last_error,
                    "updated_at": state.updated_at.isoformat(),
                }
                for state in sorted(
                    (state for state in states if state.last_error),
                    key=lambda state: state.updated_at,
                    reverse=True,
                )[:10]
            ],
        }

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
            working_state = GenerationState(
                cocktail_id=source.id,
                input_fingerprint=self._fingerprint(source, parts),
                contents_fingerprint=contents_fingerprint,
                text_model=self.settings.gemini_text_model,
                image_model=self.settings.nvidia_image_model,
                status="prompt_ready",
                cocktail_contents_prompt=contents,
                glass_prompt=parts.glass,
                background_prompt=parts.background,
                composition_prompt=parts.composition,
                expanded_prompt=final_prompt,
                attempts=state.attempts if state else 0,
            )
            self.state_store.save(working_state)
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

    def run(self, options: RunOptions) -> RunSummary:
        if options.limit is not None and options.limit <= 0:
            raise ValueError("--limit must be greater than zero")
        if not options.dry_run:
            self._verify_directories()

        summary = RunSummary()
        handled = 0
        lock_context = (
            _null_context() if options.dry_run else advisory_lock(self.session_factory)
        )
        with lock_context:
            for source in self._load_sources(options.cocktail_ids):
                action = self._process_source(source, options, summary)
                if action in {
                    "generated",
                    "repaired",
                    "failed",
                    "invalid",
                    "dry_run",
                    "staged",
                }:
                    handled += 1
                if options.limit is not None and handled >= options.limit:
                    break
        return summary

    def _load_sources(self, cocktail_ids: Iterable[int]) -> Iterator[CocktailSource]:
        statement = select(
            Cocktail.id,
            Cocktail.name,
            Cocktail.name_en,
            Cocktail.glass,
            Cocktail.base_tag,
            Cocktail.recipe,
            Cocktail.image_url,
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
                    image_url=row.image_url,
                )

    def _process_source(
        self,
        source: CocktailSource,
        options: RunOptions,
        summary: RunSummary,
    ) -> str:
        state = self.state_store.load(source.id)

        source_error = self._source_error(source)
        if source_error:
            if not options.dry_run:
                self.state_store.save(
                    GenerationState(
                        cocktail_id=source.id,
                        input_fingerprint=self._invalid_fingerprint(source),
                        text_model=self.settings.gemini_text_model,
                        image_model=self.settings.nvidia_image_model,
                        status="failed",
                        last_error=f"input or prompt configuration: {source_error}",
                    )
                )
            logger.error(
                "[%s] invalid input/configuration: %s", source.id, source_error
            )
            summary.invalid += 1
            return "invalid"

        prompt_parts = self.prompt_catalog.parts_for(source)
        fingerprint = self._fingerprint(source, prompt_parts)

        if (
            not options.force
            and state is not None
            and state.input_fingerprint == fingerprint
            and self._state_has_valid_images(state)
        ):
            assert state is not None and state.main_filename is not None
            main_url = build_main_url(
                self.settings.cocktail_image_base_url,
                state.main_filename,
            )
            if source.image_url != main_url or state.status != "completed":
                if options.dry_run:
                    logger.info(
                        "[%s] DRY RUN: repair DB URL -> %s", source.id, main_url
                    )
                    summary.dry_run += 1
                    return "dry_run"
                if options.no_db_update:
                    logger.info("[%s] image ready; DB update disabled", source.id)
                    summary.staged += 1
                    return "staged"
                self._update_image_url(source.id, main_url)
                state.status = "completed"
                state.last_error = None
                self.state_store.save(state)
                logger.info("[%s] DB URL repaired: %s", source.id, main_url)
                summary.repaired += 1
                return "repaired"
            logger.info("[%s] already complete; skipping", source.id)
            summary.skipped += 1
            return "skipped"

        if (
            state is not None
            and state.status == "failed"
            and state.input_fingerprint == fingerprint
            and not options.retry_failed
            and not options.force
        ):
            logger.info(
                "[%s] previously failed; use --retry-failed to retry",
                source.id,
            )
            summary.skipped += 1
            return "skipped"

        if options.dry_run:
            logger.info(
                "[%s] DRY RUN: generate %s / %s",
                source.id,
                source.name,
                source.name_en or "-",
            )
            summary.dry_run += 1
            return "dry_run"

        attempts = (state.attempts if state else 0) + 1
        contents_fingerprint = build_contents_fingerprint(
            source,
            text_model=self.settings.gemini_text_model,
        )
        reusable_contents = (
            state.cocktail_contents_prompt
            if state is not None
            and state.cocktail_contents_prompt
            and (
                state.contents_fingerprint == contents_fingerprint
                or state.input_fingerprint == fingerprint
            )
            else None
        )
        expanded_prompt = (
            build_final_prompt(reusable_contents, prompt_parts)
            if reusable_contents
            else (
                state.expanded_prompt
                if state is not None
                and state.input_fingerprint == fingerprint
                and state.expanded_prompt
                else None
            )
        )
        working_state = GenerationState(
            cocktail_id=source.id,
            input_fingerprint=fingerprint,
            contents_fingerprint=contents_fingerprint,
            text_model=self.settings.gemini_text_model,
            image_model=self.settings.nvidia_image_model,
            status="prompt_ready" if expanded_prompt else "quota_wait",
            cocktail_contents_prompt=reusable_contents,
            glass_prompt=prompt_parts.glass,
            background_prompt=prompt_parts.background,
            composition_prompt=prompt_parts.composition,
            expanded_prompt=expanded_prompt,
            attempts=attempts,
        )

        while True:
            try:
                if not working_state.expanded_prompt:
                    expansion_input = build_contents_expansion_prompt(source)
                    working_state.cocktail_contents_prompt = (
                        self.prompt_gateway.expand_prompt(expansion_input)
                    )
                    working_state.expanded_prompt = build_final_prompt(
                        working_state.cocktail_contents_prompt,
                        prompt_parts,
                    )
                    working_state.status = "prompt_ready"
                    working_state.last_error = None
                    self.state_store.save(working_state)

                raw_image: bytes | None = None
                # NVIDIA 직접 생성은 수동 프롬프트 생성 방식으로 전환해 비활성화했다.
                # raw_image = self.image_gateway.generate_image(
                #     working_state.expanded_prompt
                # )
                if raw_image is None:
                    raise GenerationFatalError(
                        "NVIDIA image generation is disabled; export the prompt CSV "
                        "and upload manually generated files instead"
                    )
                main, thumbnail, digest = create_image_variants(
                    raw_image,
                    crop_size=(
                        self.settings.nvidia_image_crop_width,
                        self.settings.nvidia_image_crop_height,
                    ),
                )
                main_filename, thumbnail_filename = image_filenames(
                    source.id,
                    digest,
                )
                main_path = self.settings.cocktail_image_output_dir / main_filename
                thumbnail_path = (
                    self.settings.cocktail_image_output_dir / thumbnail_filename
                )
                atomic_write(main_path, main)
                atomic_write(thumbnail_path, thumbnail)
                self._validate_files(main_path, thumbnail_path)

                working_state.status = "image_ready"
                working_state.main_filename = main_filename
                working_state.thumbnail_filename = thumbnail_filename
                working_state.image_sha256 = digest
                working_state.last_error = None
                self.state_store.save(working_state)

                main_url = build_main_url(
                    self.settings.cocktail_image_base_url,
                    main_filename,
                )
                if options.no_db_update:
                    logger.info(
                        "[%s] staged without DB update: %s",
                        source.id,
                        main_url,
                    )
                    summary.staged += 1
                    return "staged"
                try:
                    self._update_image_url(source.id, main_url)
                except Exception as error:  # noqa: BLE001
                    working_state.last_error = f"database update: {error}"
                    self.state_store.save(working_state)
                    raise GenerationItemError(
                        f"Image files are ready but DB update failed: {error}"
                    ) from error

                working_state.status = "completed"
                working_state.last_error = None
                self.state_store.save(working_state)
                logger.info("[%s] generated: %s", source.id, main_url)
                summary.generated += 1
                return "generated"
            except GenerationQuotaError as error:
                working_state.status = "quota_wait"
                working_state.last_error = str(error)
                self.state_store.save(working_state)
                raise
            except GenerationFatalError as error:
                working_state.status = "failed"
                working_state.last_error = str(error)
                self.state_store.save(working_state)
                raise
            except (GenerationItemError, OSError, ValueError) as error:
                if working_state.main_filename and working_state.thumbnail_filename:
                    working_state.status = "image_ready"
                else:
                    working_state.status = "failed"
                working_state.last_error = str(error)
                self.state_store.save(working_state)
                logger.error("[%s] generation failed: %s", source.id, error)
                summary.failed += 1
                return "failed"

    def _fingerprint(self, source: CocktailSource, parts: PromptParts) -> str:
        return build_input_fingerprint(
            source,
            parts,
            text_model=self.settings.gemini_text_model,
            image_model=self.settings.nvidia_image_model,
            image_config=self.image_gateway.image_config,
        )

    def _source_error(self, source: CocktailSource) -> str | None:
        if source.validation_error:
            return source.validation_error
        try:
            self.prompt_catalog.parts_for(source)
        except ValueError as error:
            return str(error)
        return None

    @staticmethod
    def _invalid_fingerprint(source: CocktailSource) -> str:
        return f"invalid:{source.id}:{source.glass}:{source.base_tag}"

    def _state_has_valid_images(self, state: GenerationState | None) -> bool:
        if state is None or not state.main_filename or not state.thumbnail_filename:
            return False
        main_path = self.settings.cocktail_image_output_dir / state.main_filename
        thumbnail_path = (
            self.settings.cocktail_image_output_dir / state.thumbnail_filename
        )
        try:
            self._validate_files(main_path, thumbnail_path)
        except (OSError, ValueError):
            return False
        return True

    @staticmethod
    def _validate_files(main_path: Path, thumbnail_path: Path) -> None:
        validate_webp(main_path.read_bytes(), MAIN_IMAGE_SIZE)
        validate_webp(thumbnail_path.read_bytes(), THUMBNAIL_IMAGE_SIZE)

    def _update_image_url(self, cocktail_id: int, image_url: str) -> None:
        with self.session_factory() as session:
            try:
                result = session.execute(
                    update(Cocktail)
                    .where(Cocktail.id == cocktail_id)
                    .values(image_url=image_url)
                )
                if result.rowcount != 1:
                    raise RuntimeError(f"Cocktail {cocktail_id} no longer exists")
                session.commit()
            except BaseException:
                session.rollback()
                raise

    def _verify_directories(self) -> None:
        for directory in (
            self.settings.cocktail_image_output_dir,
            self.settings.cocktail_image_state_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=directory):
                pass


@contextmanager
def _null_context() -> Iterator[None]:
    yield
