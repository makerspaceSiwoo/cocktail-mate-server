"""Resumable DB-to-CSV cocktail taste-generation pipeline."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cocktail_mate_db.models import Cocktail, CocktailIngredient, Ingredient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.taste_generation.core import (
    CocktailTasteSource,
    FlavorProfile,
    IngredientSource,
    Intensity,
    build_taste_prompt,
    csv_row_for,
    read_existing_rows,
    render_embedding_text,
    row_matches_source,
    write_csv_rows,
)
from app.taste_generation.errors import (
    TasteGenerationFatalError,
    TasteGenerationItemError,
    TasteGenerationQuotaError,
)

logger = logging.getLogger("app.taste_generation.pipeline")


class TasteGateway(Protocol):
    def verify_model(self) -> None: ...

    def generate_profile(self, prompt: str) -> FlavorProfile: ...


@dataclass(frozen=True, slots=True)
class ExportOptions:
    limit: int | None = None
    cocktail_ids: tuple[int, ...] = ()
    force: bool = False
    dry_run: bool = False


@dataclass(slots=True)
class ExportSummary:
    generated: int = 0
    skipped: int = 0
    failed: int = 0
    invalid: int = 0
    dry_run: int = 0


class CocktailTastePipeline:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        gateway: TasteGateway,
    ) -> None:
        self.session_factory = session_factory
        self.gateway = gateway

    def preflight(
        self,
        output_path: Path,
        *,
        cocktail_ids: tuple[int, ...] = (),
    ) -> dict[str, object]:
        self.gateway.verify_model()
        existing_rows = read_existing_rows(output_path)
        sources = list(self._load_sources(cocktail_ids))
        invalid = sum(source.validation_error is not None for source in sources)
        current = sum(
            row_matches_source(existing_rows[source.id], source)
            for source in sources
            if source.id in existing_rows
        )
        return {
            "cocktails": len(sources),
            "valid": len(sources) - invalid,
            "invalid": invalid,
            "already_current": current,
            "pending": len(sources) - invalid - current,
            "output": str(output_path),
        }

    def export(self, output_path: Path, options: ExportOptions) -> ExportSummary:
        if options.limit is not None and options.limit <= 0:
            raise ValueError("--limit must be greater than zero")

        rows = read_existing_rows(output_path)
        summary = ExportSummary()
        handled = 0
        for source in self._load_sources(options.cocktail_ids):
            source_error = source.validation_error
            if source_error:
                logger.error("[%s] invalid source: %s", source.id, source_error)
                summary.invalid += 1
                handled += 1
            elif (
                not options.force
                and source.id in rows
                and row_matches_source(rows[source.id], source)
            ):
                logger.info("[%s] already current; skipping", source.id)
                summary.skipped += 1
                continue
            elif options.dry_run:
                logger.info(
                    "[%s] DRY RUN: generate %s / %s",
                    source.id,
                    source.name,
                    source.name_en or "-",
                )
                summary.dry_run += 1
                handled += 1
            else:
                try:
                    profile = self.gateway.generate_profile(build_taste_prompt(source))
                    self._validate_profile_for_source(source, profile)
                except (TasteGenerationFatalError, TasteGenerationQuotaError):
                    raise
                except TasteGenerationItemError as error:
                    logger.error("[%s] Gemini failed: %s", source.id, error)
                    summary.failed += 1
                    handled += 1
                else:
                    rows[source.id] = csv_row_for(
                        source,
                        render_embedding_text(profile),
                    )
                    write_csv_rows(output_path, rows)
                    logger.info("[%s] generated and checkpointed", source.id)
                    summary.generated += 1
                    handled += 1

            if options.limit is not None and handled >= options.limit:
                break
        return summary

    def _load_sources(
        self,
        cocktail_ids: Iterable[int],
    ) -> Iterator[CocktailTasteSource]:
        statement = (
            select(
                Cocktail.id.label("cocktail_id"),
                Cocktail.name.label("cocktail_name"),
                Cocktail.name_en.label("cocktail_name_en"),
                Cocktail.recipe,
                Cocktail.abv.label("cocktail_abv"),
                Cocktail.base_tag,
                CocktailIngredient.id.label("link_id"),
                CocktailIngredient.amount,
                CocktailIngredient.unit,
                Ingredient.name.label("ingredient_name"),
                Ingredient.name_en.label("ingredient_name_en"),
                Ingredient.category,
                Ingredient.description.label("ingredient_description"),
                Ingredient.abv.label("ingredient_abv"),
            )
            .select_from(Cocktail)
            .outerjoin(
                CocktailIngredient,
                CocktailIngredient.cocktail_id == Cocktail.id,
            )
            .outerjoin(
                Ingredient,
                Ingredient.id == CocktailIngredient.ingredient_id,
            )
            .order_by(Cocktail.id, CocktailIngredient.id)
        )
        ids = tuple(cocktail_ids)
        if ids:
            statement = statement.where(Cocktail.id.in_(ids))

        with self.session_factory() as session:
            current_id: int | None = None
            current: dict[str, object] | None = None
            ingredients: list[IngredientSource] = []
            for row in session.execute(statement):
                row_id = int(row.cocktail_id)
                if current_id is not None and row_id != current_id:
                    assert current is not None
                    yield self._source_from_group(current, ingredients)
                    ingredients = []
                if row_id != current_id:
                    current_id = row_id
                    current = {
                        "id": row_id,
                        "name": row.cocktail_name,
                        "name_en": (
                            str(row.cocktail_name_en) if row.cocktail_name_en else None
                        ),
                        "recipe": tuple(row.recipe or ()),
                        "abv": (
                            float(row.cocktail_abv)
                            if row.cocktail_abv is not None
                            else None
                        ),
                        "base_tag": row.base_tag,
                    }
                if row.link_id is not None:
                    ingredients.append(
                        IngredientSource(
                            name=row.ingredient_name,
                            name_en=(
                                str(row.ingredient_name_en)
                                if row.ingredient_name_en
                                else None
                            ),
                            category=row.category,
                            amount=(
                                float(row.amount) if row.amount is not None else None
                            ),
                            unit=row.unit,
                            description=row.ingredient_description,
                            abv=(
                                float(row.ingredient_abv)
                                if row.ingredient_abv is not None
                                else None
                            ),
                        )
                    )
            if current is not None:
                yield self._source_from_group(current, ingredients)

    @staticmethod
    def _source_from_group(
        current: dict[str, object],
        ingredients: list[IngredientSource],
    ) -> CocktailTasteSource:
        return CocktailTasteSource(
            id=int(current["id"]),
            name=str(current["name"]),
            name_en=(
                str(current["name_en"]) if current["name_en"] is not None else None
            ),
            recipe=tuple(str(step) for step in current["recipe"]),  # type: ignore[union-attr]
            abv=float(current["abv"]) if current["abv"] is not None else None,
            base_tag=(
                str(current["base_tag"]) if current["base_tag"] is not None else None
            ),
            ingredients=tuple(ingredients),
        )

    @staticmethod
    def _validate_profile_for_source(
        source: CocktailTasteSource,
        profile: FlavorProfile,
    ) -> None:
        is_non_alcoholic = source.abv == 0 or source.base_tag == "non_alcoholic"
        has_no_alcohol = profile.alcohol_presence == Intensity.NONE
        if is_non_alcoholic and not has_no_alcohol:
            raise TasteGenerationItemError(
                "Gemini assigned alcohol presence to a non-alcoholic cocktail"
            )
        if source.abv is not None and source.abv > 0 and has_no_alcohol:
            raise TasteGenerationItemError(
                "Gemini omitted alcohol presence from an alcoholic cocktail"
            )
