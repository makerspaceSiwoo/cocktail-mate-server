"""Persist generated cocktail images directly from a trusted server process."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dotenv import load_dotenv

from app.image_generation.core import ImageGenerationSettings
from app.image_upload.service import (
    MAX_UPLOAD_BYTES,
    IncomingImage,
    persist_batch,
)

BATCH_SIZE = 10
DEFAULT_CSV_PATH = Path("image-upload/cocktail-image-prompts.csv")
DEFAULT_IMAGE_DIR = Path("image-upload/images")
SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persist generated cocktail images to server storage and DB."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--folder", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail when any CSV image file is missing instead of skipping it.",
    )
    return parser


def _load_environment() -> Path:
    env_file = Path(".env.local") if Path(".env.local").is_file() else Path(".env")
    load_dotenv(env_file, override=True)
    return env_file


def _load_image_paths(
    csv_path: Path,
    image_dir: Path,
    *,
    require_all: bool,
    allow_empty: bool = False,
) -> tuple[list[Path], list[str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or "image_filename" not in rows[0]:
        raise ValueError("CSV must contain at least one image_filename row")

    paths: list[Path] = []
    missing: list[str] = []
    seen: set[str] = set()
    for row in rows:
        filename = Path(row["image_filename"]).name
        if not filename or filename in seen:
            raise ValueError(f"Invalid or duplicate image_filename: {filename!r}")
        seen.add(filename)
        requested_path = image_dir / filename
        candidates = (requested_path,) + tuple(
            image_dir / f"{requested_path.stem}{suffix}"
            for suffix in SUPPORTED_IMAGE_SUFFIXES
            if suffix != requested_path.suffix.lower()
        )
        resolved = next((path for path in candidates if path.is_file()), None)
        if resolved is not None:
            paths.append(resolved)
        else:
            missing.append(filename)
    if require_all and missing:
        raise ValueError(f"Missing {len(missing)} image files; first: {missing[:5]}")
    if not paths and not allow_empty:
        raise ValueError(f"No CSV-listed images found in {image_dir}")
    return paths, missing


def _chunks(paths: list[Path]) -> list[list[Path]]:
    return [
        paths[index : index + BATCH_SIZE] for index in range(0, len(paths), BATCH_SIZE)
    ]


def _persist_paths(
    db,
    settings: ImageGenerationSettings,
    paths: list[Path],
) -> list[dict[str, object]]:
    incoming: list[IncomingImage] = []
    for path in paths:
        if path.stat().st_size > MAX_UPLOAD_BYTES:
            raise ValueError(f"{path.name}: file exceeds 15 MiB")
        incoming.append(IncomingImage(filename=path.name, data=path.read_bytes()))
    return persist_batch(db, incoming, settings)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    env_file = _load_environment()
    settings = ImageGenerationSettings()

    try:
        paths, missing = _load_image_paths(
            arguments.csv,
            arguments.folder,
            require_all=arguments.require_all,
            allow_empty=arguments.dry_run,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    batches = _chunks(paths)
    if arguments.dry_run:
        print(
            json.dumps(
                {
                    "env_file": str(env_file),
                    "csv": str(arguments.csv),
                    "folder": str(arguments.folder),
                    "ready": len(paths),
                    "missing": len(missing),
                    "batches": [len(batch) for batch in batches],
                    "output_dir": str(settings.cocktail_image_output_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    from app.core.database import SessionLocal  # noqa: PLC0415

    uploaded = 0
    for index, batch in enumerate(batches, start=1):
        with SessionLocal() as db:
            items = _persist_paths(db, settings, batch)
        uploaded += len(items)
        print(
            json.dumps(
                {
                    "batch": index,
                    "files": [path.name for path in batch],
                    "uploaded": len(items),
                    "items": items,
                },
                ensure_ascii=False,
            )
        )

    print(
        json.dumps(
            {
                "uploaded": uploaded,
                "missing_skipped": len(missing),
                "batches": len(batches),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
