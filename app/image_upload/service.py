"""Validate, transform, persist, and publish uploaded cocktail images."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from cocktail_mate_db.models import Cocktail
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.image_generation.core import (
    ImageGenerationSettings,
    atomic_write,
    build_main_url,
    build_thumbnail_url,
    create_image_variants,
    image_filenames,
)

MAX_BATCH_FILES = 10
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_SOURCE_DIMENSION = 4096
_SOURCE_FILENAME = re.compile(
    r"^cocktail-(?P<cocktail_id>[1-9]\d*)\.(?:png|jpe?g|webp)$",
    re.IGNORECASE,
)


class ImageUploadError(ValueError):
    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class IncomingImage:
    filename: str
    data: bytes


@dataclass(frozen=True, slots=True)
class PreparedImage:
    cocktail_id: int
    source_filename: str
    main_filename: str
    thumbnail_filename: str
    main_data: bytes
    thumbnail_data: bytes
    image_url: str
    thumbnail_url: str


def parse_cocktail_id(filename: str) -> int:
    match = _SOURCE_FILENAME.fullmatch(Path(filename).name)
    if match is None:
        raise ImageUploadError(
            f"Invalid filename {filename!r}; expected cocktail-<id>.png"
        )
    return int(match.group("cocktail_id"))


def _source_size(data: bytes, filename: str) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ImageUploadError(
                    f"{filename}: only PNG, JPEG, and WebP are supported"
                )
            width, height = image.size
            if (
                width > MAX_SOURCE_DIMENSION
                or height > MAX_SOURCE_DIMENSION
                or width < 400
                or height < 300
            ):
                raise ImageUploadError(
                    f"{filename}: unsupported dimensions {width}x{height}"
                )
            image.verify()
            return width, height
    except (UnidentifiedImageError, OSError) as error:
        raise ImageUploadError(f"{filename}: invalid image file") from error


def prepare_batch(
    incoming: list[IncomingImage],
    settings: ImageGenerationSettings,
) -> list[PreparedImage]:
    if not 1 <= len(incoming) <= MAX_BATCH_FILES:
        raise ImageUploadError("Each batch must contain between 1 and 10 images")

    parsed_ids = [parse_cocktail_id(item.filename) for item in incoming]
    if len(set(parsed_ids)) != len(parsed_ids):
        raise ImageUploadError("A batch cannot contain duplicate cocktail IDs")

    prepared: list[PreparedImage] = []
    for item, cocktail_id in zip(incoming, parsed_ids, strict=True):
        if not item.data:
            raise ImageUploadError(f"{item.filename}: empty file")
        if len(item.data) > MAX_UPLOAD_BYTES:
            raise ImageUploadError(f"{item.filename}: file exceeds 15 MiB")
        _source_size(item.data, item.filename)
        try:
            main, thumbnail, digest = create_image_variants(item.data)
        except (OSError, ValueError) as error:
            raise ImageUploadError(f"{item.filename}: {error}") from error
        main_filename, thumbnail_filename = image_filenames(cocktail_id, digest)
        image_url = build_main_url(
            settings.cocktail_image_base_url,
            main_filename,
        )
        prepared.append(
            PreparedImage(
                cocktail_id=cocktail_id,
                source_filename=item.filename,
                main_filename=main_filename,
                thumbnail_filename=thumbnail_filename,
                main_data=main,
                thumbnail_data=thumbnail,
                image_url=image_url,
                thumbnail_url=build_thumbnail_url(
                    image_url,
                    settings.cocktail_image_base_url,
                ),
            )
        )
    return prepared


def persist_batch(
    db: Session,
    incoming: list[IncomingImage],
    settings: ImageGenerationSettings,
) -> list[dict[str, object]]:
    prepared = prepare_batch(incoming, settings)
    cocktail_ids = [item.cocktail_id for item in prepared]
    rows = db.execute(
        select(Cocktail.id, Cocktail.name).where(Cocktail.id.in_(cocktail_ids))
    ).all()
    names = {int(row.id): str(row.name) for row in rows}
    missing = sorted(set(cocktail_ids) - names.keys())
    if missing:
        raise ImageUploadError(
            f"Cocktail IDs do not exist: {missing}",
            status_code=404,
        )

    created_paths: list[Path] = []
    try:
        for item in prepared:
            for filename, data in (
                (item.main_filename, item.main_data),
                (item.thumbnail_filename, item.thumbnail_data),
            ):
                path = settings.cocktail_image_output_dir / filename
                existed = path.exists()
                atomic_write(path, data)
                if not existed:
                    created_paths.append(path)

        for item in prepared:
            result = db.execute(
                update(Cocktail)
                .where(Cocktail.id == item.cocktail_id)
                .values(image_url=item.image_url)
            )
            if result.rowcount != 1:
                raise RuntimeError(
                    f"Cocktail {item.cocktail_id} disappeared during upload"
                )
        db.commit()
    except BaseException:
        db.rollback()
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise

    return [
        {
            "cocktail_id": item.cocktail_id,
            "cocktail_name": names[item.cocktail_id],
            "source_filename": item.source_filename,
            "image_url": item.image_url,
            "thumbnail_url": item.thumbnail_url,
        }
        for item in prepared
    ]
