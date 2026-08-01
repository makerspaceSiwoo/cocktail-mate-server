"""Generate Cocktail Mate prompts with Gemini and images with NVIDIA FLUX."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from app.image_generation.core import ImageGenerationSettings
from app.image_generation.errors import GenerationRequestError
from app.image_generation.gemini import GeminiGateway
from app.image_generation.nvidia import NvidiaImageGateway
from app.image_generation.pipeline import (
    AlreadyRunningError,
    CocktailImagePipeline,
    RunOptions,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate resumable cocktail images with Gemini + NVIDIA FLUX."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser(
        "preflight",
        help="Verify credentials, models, DB, storage, and pending work.",
    )

    run = subcommands.add_parser("run", help="Generate incomplete cocktail images.")
    run.add_argument("--limit", type=int, help="Maximum number of actionable rows.")
    run.add_argument(
        "--cocktail-id",
        type=int,
        action="append",
        default=[],
        help="Only process this cocktail ID; may be repeated.",
    )
    run.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when valid published variants already exist.",
    )
    run.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry rows whose latest state is failed.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="List actionable rows without API, state, file, or DB writes.",
    )
    run.add_argument(
        "--no-db-update",
        action="store_true",
        help="Generate and persist files/state, but leave cocktails.image_url unchanged.",
    )

    subcommands.add_parser(
        "status",
        help="Show persisted generation-state counts and recent errors.",
    )
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _load_runtime_environment() -> Path:
    """Match the Makefile convention: local override first, production otherwise."""

    env_file = Path(".env.local") if Path(".env.local").is_file() else Path(".env")
    load_dotenv(env_file, override=True)
    return env_file


def _require_api_keys(settings: ImageGenerationSettings) -> None:
    if not settings.gemini_api_key.get_secret_value():
        raise ValueError("GEMINI_API_KEY is required for this command")
    if not settings.nvidia_api_key.get_secret_value():
        raise ValueError("NVIDIA_API_KEY is required for this command")


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    env_file = _load_runtime_environment()
    from app.core.database import SessionLocal  # noqa: PLC0415

    arguments = _parser().parse_args(argv)
    settings = ImageGenerationSettings()
    prompt_gateway = GeminiGateway(settings)
    image_gateway = NvidiaImageGateway(settings)
    pipeline = CocktailImagePipeline(
        settings,
        SessionLocal,
        prompt_gateway,
        image_gateway,
    )

    try:
        if arguments.command == "preflight":
            _require_api_keys(settings)
            result = pipeline.preflight()
            result["env_file"] = str(env_file)
            result["provider_warning"] = (
                "NVIDIA API credits and limits are account-specific. Preflight "
                "validates local FLUX settings; use run --limit 1 --no-db-update "
                "to validate actual image access."
            )
        elif arguments.command == "status":
            result = pipeline.status()
        else:
            if not arguments.dry_run:
                _require_api_keys(settings)
            summary = pipeline.run(
                RunOptions(
                    limit=arguments.limit,
                    cocktail_ids=tuple(arguments.cocktail_id),
                    force=arguments.force,
                    retry_failed=arguments.retry_failed,
                    dry_run=arguments.dry_run,
                    no_db_update=arguments.no_db_update,
                )
            )
            result = {
                "generated": summary.generated,
                "repaired": summary.repaired,
                "skipped": summary.skipped,
                "failed": summary.failed,
                "invalid": summary.invalid,
                "dry_run": summary.dry_run,
                "staged": summary.staged,
            }
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted; progress was checkpointed.")
        return 130
    except (AlreadyRunningError, GenerationRequestError, ValueError) as error:
        logging.getLogger(__name__).error("%s", error)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
