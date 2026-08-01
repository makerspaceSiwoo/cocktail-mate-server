from __future__ import annotations

import base64
import argparse
import csv
import io
import json
from decimal import Decimal
from pathlib import Path

from PIL import Image
import pytest

from app.image_generation.gemini_batch import (
    BatchPromptRow,
    GEMINI_BATCH_IMAGE_MODEL,
    GeminiBatchError,
    build_batch_jsonl,
    build_batch_manifest,
    download_batch_results,
    estimated_image_output_cost,
    load_prompt_rows,
    select_batch_rows,
    sha256_bytes,
)
from scripts.generate_cocktail_images_batch import _verify_submit_guards


def _image_bytes(size: tuple[int, int] = (1024, 768)) -> bytes:
    image = Image.new("RGB", size, color=(195, 154, 232))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _write_prompt_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "id",
                "cocktail_name",
                "image_filename",
                "final_image_prompt",
            ),
        )
        writer.writeheader()
        for cocktail_id in range(1, 4):
            writer.writerow(
                {
                    "id": cocktail_id,
                    "cocktail_name": f"칵테일 {cocktail_id}",
                    "image_filename": f"cocktail-{cocktail_id}.png",
                    "final_image_prompt": f"Prompt {cocktail_id}",
                }
            )


def test_prepare_skips_existing_image_with_different_extension(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "prompts.csv"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_prompt_csv(csv_path)
    (image_dir / "cocktail-1.jpeg").write_bytes(_image_bytes())

    selected, skipped = select_batch_rows(
        load_prompt_rows(csv_path),
        image_dir,
        min_id=1,
    )

    assert [row.cocktail_id for row in selected] == [2, 3]
    assert skipped == 1


def test_jsonl_requests_image_only_4_by_3_1k() -> None:
    rows = [
        BatchPromptRow(
            cocktail_id=35,
            cocktail_name="테스트",
            image_filename="cocktail-35.png",
            prompt="A test cocktail prompt",
        )
    ]

    line = json.loads(build_batch_jsonl(rows))

    assert line["key"] == "cocktail-35"
    assert line["request"]["contents"][0]["parts"][0]["text"] == (
        "A test cocktail prompt"
    )
    assert line["request"]["generation_config"] == {
        "responseModalities": ["IMAGE"],
        "imageConfig": {"aspectRatio": "4:3", "imageSize": "1K"},
    }
    assert estimated_image_output_cost(568) == Decimal("9.5424")


def test_submit_guard_rejects_changed_model() -> None:
    rows = [BatchPromptRow(35, "테스트", "cocktail-35.png", "Prompt")]
    jsonl = build_batch_jsonl(rows)
    manifest = build_batch_manifest(
        rows,
        csv_sha256="csv-hash",
        jsonl_sha256=sha256_bytes(jsonl),
    )
    manifest["model"] = "gemini-3.1-flash-image"
    arguments = argparse.Namespace(
        confirm_paid_batch=True,
        max_requests=1,
        max_image_output_cost_usd="1.00",
    )

    with pytest.raises(GeminiBatchError, match="expected"):
        _verify_submit_guards(arguments, manifest, jsonl)

    manifest["model"] = GEMINI_BATCH_IMAGE_MODEL
    assert _verify_submit_guards(arguments, manifest, jsonl) == (
        1,
        Decimal("0.0168"),
    )


def test_download_saves_png_and_records_provider_failure(tmp_path: Path) -> None:
    rows = [
        BatchPromptRow(35, "성공", "cocktail-35.png", "Prompt 35"),
        BatchPromptRow(36, "실패", "cocktail-36.png", "Prompt 36"),
    ]
    jsonl = build_batch_jsonl(rows)
    manifest = build_batch_manifest(
        rows,
        csv_sha256="csv-hash",
        jsonl_sha256=sha256_bytes(jsonl),
    )
    encoded = base64.b64encode(_image_bytes()).decode("ascii")
    results = (
        json.dumps(
            {
                "key": "cocktail-35",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "image/png",
                                            "data": encoded,
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "key": "cocktail-36",
                "error": {"code": 400, "message": "safety blocked"},
            }
        )
        + "\n"
    ).encode()

    summary = download_batch_results(results, manifest, tmp_path / "images")

    assert summary.saved == 1
    assert summary.failed == 1
    assert summary.failures[0].cocktail_id == 36
    with Image.open(tmp_path / "images" / "cocktail-35.png") as image:
        assert image.format == "PNG"
        assert image.size == (1024, 768)
