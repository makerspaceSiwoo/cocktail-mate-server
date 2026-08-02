"""Generate resumable, embedding-ready cocktail taste descriptions with Gemini."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from app.taste_generation.core import TasteGenerationSettings
from app.taste_generation.errors import TasteGenerationError
from app.taste_generation.gemini import GeminiTasteGateway
from app.taste_generation.pipeline import CocktailTastePipeline, ExportOptions

DEFAULT_OUTPUT_PATH = Path("taste-data/cocktail-taste-descriptions.csv")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate normalized Korean cocktail taste descriptions from the DB "
            "and checkpoint them to CSV."
        )
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    preflight = subcommands.add_parser(
        "preflight",
        help="Verify Gemini, DB input, and any existing output CSV.",
    )
    _add_shared_arguments(preflight)

    run = subcommands.add_parser(
        "run",
        help="Generate missing or stale CSV rows.",
    )
    _add_shared_arguments(run)
    run.add_argument(
        "--limit",
        type=int,
        help="Maximum number of actionable rows (skipped rows do not count).",
    )
    run.add_argument(
        "--force",
        action="store_true",
        help="Regenerate selected rows even when their CSV inputs match.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="List actionable rows without calling Gemini or writing the CSV.",
    )
    return parser


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--cocktail-id",
        type=int,
        action="append",
        default=[],
        help="Only process this cocktail ID; may be repeated.",
    )


def _load_environment() -> Path:
    env_file = Path(".env.local") if Path(".env.local").is_file() else Path(".env")
    load_dotenv(env_file, override=True)
    return env_file


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    arguments = _parser().parse_args(argv)
    env_file = _load_environment()

    settings = TasteGenerationSettings()
    needs_api_key = arguments.command == "preflight" or not arguments.dry_run
    if needs_api_key and not settings.gemini_api_key:
        logging.getLogger(__name__).error("GEMINI_API_KEY is required")
        return 1

    from app.core.database import SessionLocal  # noqa: PLC0415

    pipeline = CocktailTastePipeline(
        SessionLocal,
        GeminiTasteGateway(settings),
    )
    try:
        if arguments.command == "preflight":
            result = pipeline.preflight(
                arguments.output,
                cocktail_ids=tuple(arguments.cocktail_id),
            )
        else:
            summary = pipeline.export(
                arguments.output,
                ExportOptions(
                    limit=arguments.limit,
                    cocktail_ids=tuple(arguments.cocktail_id),
                    force=arguments.force,
                    dry_run=arguments.dry_run,
                ),
            )
            result = {
                "generated": summary.generated,
                "skipped": summary.skipped,
                "failed": summary.failed,
                "invalid": summary.invalid,
                "dry_run": summary.dry_run,
                "output": str(arguments.output),
            }
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning(
            "Interrupted; every completed row was already checkpointed."
        )
        return 130
    except (OSError, ValueError, TasteGenerationError) as error:
        logging.getLogger(__name__).error("%s", error)
        return 1

    result["env_file"] = str(env_file)
    result["model"] = settings.gemini_text_model
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
