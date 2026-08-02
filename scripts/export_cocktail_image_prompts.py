"""Export resumable cocktail image prompts to a UTF-8 CSV."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from app.image_generation.core import ImageGenerationSettings
from app.image_generation.errors import GenerationRequestError
from app.image_generation.gemini import GeminiGateway
from app.image_generation.pipeline import CocktailImagePipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export DB cocktail prompts and fixed image filenames to CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("image-upload/cocktail-image-prompts.csv"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cocktail-id", type=int, action="append", default=[])
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Export only rows with an already cached Gemini contents prompt.",
    )
    return parser


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
    from app.core.database import SessionLocal  # noqa: PLC0415

    settings_overrides: dict[str, object] = {}
    if "COCKTAIL_IMAGE_STATE_DIR" not in os.environ:
        settings_overrides["cocktail_image_state_dir"] = Path("image-upload/state")
    settings = ImageGenerationSettings(**settings_overrides)
    if not arguments.cached_only and not settings.gemini_api_key.get_secret_value():
        logging.getLogger(__name__).error("GEMINI_API_KEY is required")
        return 1

    pipeline = CocktailImagePipeline(
        settings,
        SessionLocal,
        GeminiGateway(settings),
    )
    try:
        summary = pipeline.export_prompts(
            arguments.output,
            limit=arguments.limit,
            cocktail_ids=tuple(arguments.cocktail_id),
            cached_only=arguments.cached_only,
        )
    except (GenerationRequestError, OSError, ValueError) as error:
        logging.getLogger(__name__).error("%s", error)
        return 1

    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "env_file": str(env_file),
                "exported": summary.exported,
                "cache_hits": summary.cache_hits,
                "gemini_generated": summary.gemini_generated,
                "skipped": summary.skipped,
                "failed": summary.failed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
