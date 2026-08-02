"""Pure helpers for preparing and downloading Gemini image batch jobs."""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.image_generation.core import atomic_write

GEMINI_BATCH_IMAGE_MODEL = "gemini-3.1-flash-lite-image"
GEMINI_BATCH_ASPECT_RATIO = "4:3"
GEMINI_BATCH_IMAGE_SIZE = "1K"
GEMINI_BATCH_IMAGE_PRICE_USD = Decimal("0.0168")
SUPPORTED_SOURCE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

_IMAGE_FILENAME = re.compile(
    r"^cocktail-(?P<cocktail_id>[1-9]\d*)\.(?:png|jpe?g|webp)$",
    re.IGNORECASE,
)
_BATCH_KEY = re.compile(r"^cocktail-(?P<cocktail_id>[1-9]\d*)$")


class GeminiBatchError(ValueError):
    """A local batch input or downloaded response is invalid."""


@dataclass(frozen=True, slots=True)
class BatchPromptRow:
    cocktail_id: int
    cocktail_name: str
    image_filename: str
    prompt: str

    @property
    def key(self) -> str:
        return f"cocktail-{self.cocktail_id}"


@dataclass(frozen=True, slots=True)
class BatchDownloadFailure:
    key: str
    cocktail_id: int | None
    cocktail_name: str
    error: str


@dataclass(slots=True)
class BatchDownloadSummary:
    saved: int = 0
    skipped_existing: int = 0
    failed: int = 0
    failures: list[BatchDownloadFailure] = field(default_factory=list)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_prompt_rows(path: Path) -> list[BatchPromptRow]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "id",
            "cocktail_name",
            "image_filename",
            "final_image_prompt",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise GeminiBatchError(
                f"Prompt CSV must contain columns: {sorted(required)}"
            )
        raw_rows = list(reader)

    if not raw_rows:
        raise GeminiBatchError("Prompt CSV contains no cocktail rows")

    rows: list[BatchPromptRow] = []
    seen_ids: set[int] = set()
    for index, raw in enumerate(raw_rows, start=2):
        try:
            cocktail_id = int(raw["id"])
        except (TypeError, ValueError) as error:
            raise GeminiBatchError(f"CSV row {index} has an invalid id") from error
        if cocktail_id <= 0 or cocktail_id in seen_ids:
            raise GeminiBatchError(
                f"CSV row {index} has an invalid or duplicate id={cocktail_id}"
            )
        seen_ids.add(cocktail_id)

        filename = Path(raw["image_filename"]).name
        match = _IMAGE_FILENAME.fullmatch(filename)
        if match is None or int(match.group("cocktail_id")) != cocktail_id:
            raise GeminiBatchError(
                f"CSV row {index} filename does not match id={cocktail_id}"
            )
        prompt = raw["final_image_prompt"].strip()
        if not prompt:
            raise GeminiBatchError(f"CSV row {index} has an empty prompt")
        rows.append(
            BatchPromptRow(
                cocktail_id=cocktail_id,
                cocktail_name=raw["cocktail_name"].strip(),
                image_filename=f"cocktail-{cocktail_id}.png",
                prompt=prompt,
            )
        )
    return rows


def _valid_local_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def find_existing_image(image_dir: Path, cocktail_id: int) -> Path | None:
    for suffix in SUPPORTED_SOURCE_SUFFIXES:
        candidate = image_dir / f"cocktail-{cocktail_id}{suffix}"
        if candidate.is_file() and _valid_local_image(candidate):
            return candidate
    return None


def select_batch_rows(
    rows: list[BatchPromptRow],
    image_dir: Path,
    *,
    min_id: int,
    cocktail_ids: tuple[int, ...] = (),
    include_existing: bool = False,
) -> tuple[list[BatchPromptRow], int]:
    if min_id <= 0:
        raise GeminiBatchError("--min-id must be greater than zero")
    requested_ids = set(cocktail_ids)
    known_ids = {row.cocktail_id for row in rows}
    missing_ids = sorted(requested_ids - known_ids)
    if missing_ids:
        raise GeminiBatchError(f"Cocktail IDs are absent from the CSV: {missing_ids}")

    selected: list[BatchPromptRow] = []
    skipped_existing = 0
    for row in rows:
        if requested_ids:
            if row.cocktail_id not in requested_ids:
                continue
        elif row.cocktail_id < min_id:
            continue
        if not include_existing and find_existing_image(image_dir, row.cocktail_id):
            skipped_existing += 1
            continue
        selected.append(row)
    if not selected:
        raise GeminiBatchError("No image generation requests remain after filtering")
    return selected, skipped_existing


def build_batch_jsonl(rows: list[BatchPromptRow]) -> bytes:
    lines: list[str] = []
    for row in rows:
        request = {
            "key": row.key,
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": row.prompt}],
                    }
                ],
                "generation_config": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {
                        "aspectRatio": GEMINI_BATCH_ASPECT_RATIO,
                        "imageSize": GEMINI_BATCH_IMAGE_SIZE,
                    },
                },
            },
        }
        lines.append(
            json.dumps(
                request,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def estimated_image_output_cost(request_count: int) -> Decimal:
    if request_count < 0:
        raise GeminiBatchError("Request count cannot be negative")
    return GEMINI_BATCH_IMAGE_PRICE_USD * request_count


def build_batch_manifest(
    rows: list[BatchPromptRow],
    *,
    csv_sha256: str,
    jsonl_sha256: str,
) -> dict[str, object]:
    return {
        "version": 1,
        "model": GEMINI_BATCH_IMAGE_MODEL,
        "aspect_ratio": GEMINI_BATCH_ASPECT_RATIO,
        "image_size": GEMINI_BATCH_IMAGE_SIZE,
        "batch_image_output_price_usd": str(GEMINI_BATCH_IMAGE_PRICE_USD),
        "estimated_image_output_cost_usd": str(estimated_image_output_cost(len(rows))),
        "request_count": len(rows),
        "csv_sha256": csv_sha256,
        "jsonl_sha256": jsonl_sha256,
        "requests": [
            {
                "key": row.key,
                "cocktail_id": row.cocktail_id,
                "cocktail_name": row.cocktail_name,
                "image_filename": row.image_filename,
            }
            for row in rows
        ],
    }


def manifest_request_map(
    manifest: dict[str, object],
) -> dict[str, dict[str, object]]:
    raw_requests = manifest.get("requests")
    if not isinstance(raw_requests, list):
        raise GeminiBatchError("Batch manifest has no requests list")
    result: dict[str, dict[str, object]] = {}
    for raw in raw_requests:
        if not isinstance(raw, dict):
            raise GeminiBatchError("Batch manifest contains an invalid request")
        key = raw.get("key")
        if not isinstance(key, str) or _BATCH_KEY.fullmatch(key) is None:
            raise GeminiBatchError("Batch manifest contains an invalid request key")
        if key in result:
            raise GeminiBatchError(f"Batch manifest contains duplicate key={key}")
        result[key] = raw
    return result


def _result_key(result: dict[str, object]) -> str | None:
    key = result.get("key")
    if isinstance(key, str):
        return key
    metadata = result.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("key"), str):
        return str(metadata["key"])
    return None


def _final_image_part(response: object) -> dict[str, object]:
    if not isinstance(response, dict):
        raise GeminiBatchError("response is missing")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise GeminiBatchError("response has no candidates")
    candidate = candidates[0]
    content = candidate.get("content") if isinstance(candidate, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        raise GeminiBatchError("response has no content parts")
    images = [
        part
        for part in parts
        if isinstance(part, dict)
        and isinstance(part.get("inlineData") or part.get("inline_data"), dict)
    ]
    if not images:
        raise GeminiBatchError("response contains no image")
    final_images = [part for part in images if part.get("thought") is not True]
    return (final_images or images)[-1]


def _decode_image(part: dict[str, object]) -> bytes:
    inline_data = part.get("inlineData") or part.get("inline_data")
    if not isinstance(inline_data, dict):
        raise GeminiBatchError("image part has no inline data")
    encoded = inline_data.get("data")
    if not isinstance(encoded, str) or not encoded:
        raise GeminiBatchError("image part contains no base64 data")
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise GeminiBatchError("image part contains invalid base64 data") from error


def normalize_generated_png(data: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.width < 400 or image.height < 300:
                raise GeminiBatchError(
                    f"generated image is too small: {image.width}x{image.height}"
                )
            if abs((image.width / image.height) - (4 / 3)) > 0.01:
                raise GeminiBatchError(
                    f"generated image is not 4:3: {image.width}x{image.height}"
                )
            normalized = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            output = io.BytesIO()
            normalized.save(output, format="PNG", optimize=True)
            return output.getvalue()
    except UnidentifiedImageError as error:
        raise GeminiBatchError("generated bytes are not a valid image") from error


def download_batch_results(
    result_bytes: bytes,
    manifest: dict[str, object],
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> BatchDownloadSummary:
    request_map = manifest_request_map(manifest)
    summary = BatchDownloadSummary()
    seen_keys: set[str] = set()

    for line_number, raw_line in enumerate(result_bytes.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            result = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            summary.failures.append(
                BatchDownloadFailure(
                    key=f"line-{line_number}",
                    cocktail_id=None,
                    cocktail_name="",
                    error=f"invalid result JSON: {error}",
                )
            )
            continue
        if not isinstance(result, dict):
            summary.failures.append(
                BatchDownloadFailure(
                    key=f"line-{line_number}",
                    cocktail_id=None,
                    cocktail_name="",
                    error="result line is not a JSON object",
                )
            )
            continue

        key = _result_key(result) or f"line-{line_number}"
        request = request_map.get(key)
        if request is None:
            summary.failures.append(
                BatchDownloadFailure(
                    key=key,
                    cocktail_id=None,
                    cocktail_name="",
                    error="result key is absent from the submitted manifest",
                )
            )
            continue
        if key in seen_keys:
            summary.failures.append(
                BatchDownloadFailure(
                    key=key,
                    cocktail_id=int(request["cocktail_id"]),
                    cocktail_name=str(request["cocktail_name"]),
                    error="duplicate result key",
                )
            )
            continue
        seen_keys.add(key)

        cocktail_id = int(request["cocktail_id"])
        cocktail_name = str(request["cocktail_name"])
        target = output_dir / str(request["image_filename"])
        if target.is_file() and not overwrite:
            summary.skipped_existing += 1
            continue
        try:
            if result.get("error"):
                raise GeminiBatchError(f"provider error: {result['error']}")
            part = _final_image_part(result.get("response"))
            png_data = normalize_generated_png(_decode_image(part))
            atomic_write(target, png_data)
        except (GeminiBatchError, OSError) as error:
            summary.failures.append(
                BatchDownloadFailure(
                    key=key,
                    cocktail_id=cocktail_id,
                    cocktail_name=cocktail_name,
                    error=str(error),
                )
            )
            continue
        summary.saved += 1

    for key in sorted(request_map.keys() - seen_keys):
        request = request_map[key]
        summary.failures.append(
            BatchDownloadFailure(
                key=key,
                cocktail_id=int(request["cocktail_id"]),
                cocktail_name=str(request["cocktail_name"]),
                error="batch result is missing",
            )
        )
    summary.failed = len(summary.failures)
    return summary
