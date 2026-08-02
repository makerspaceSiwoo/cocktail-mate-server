from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from app.image_generation.core import ImageGenerationSettings, validate_webp
from app.image_upload.service import (
    ImageUploadError,
    IncomingImage,
    prepare_batch,
)
from app.main import create_app
from scripts.upload_cocktail_images import _chunks, _load_image_paths, _persist_paths


def _png(size: tuple[int, int] = (1184, 880)) -> bytes:
    image = Image.new("RGB", size, color=(242, 168, 120))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _settings(tmp_path: Path) -> ImageGenerationSettings:
    return ImageGenerationSettings(
        cocktail_image_output_dir=tmp_path / "media",
        cocktail_image_state_dir=tmp_path / "state",
    )


def test_prepare_batch_uses_csv_filename_id_and_exact_variants(
    tmp_path: Path,
) -> None:
    prepared = prepare_batch(
        [IncomingImage(filename="cocktail-42.png", data=_png())],
        _settings(tmp_path),
    )

    assert len(prepared) == 1
    item = prepared[0]
    assert item.cocktail_id == 42
    assert item.source_filename == "cocktail-42.png"
    assert item.main_filename.startswith("42-")
    assert item.thumbnail_filename.endswith("-thumb.webp")
    validate_webp(item.main_data, (400, 300))
    validate_webp(item.thumbnail_data, (128, 96))


def test_prepare_batch_rejects_more_than_ten_images(tmp_path: Path) -> None:
    incoming = [
        IncomingImage(filename=f"cocktail-{index}.png", data=_png())
        for index in range(1, 12)
    ]

    with pytest.raises(ImageUploadError, match="between 1 and 10"):
        prepare_batch(incoming, _settings(tmp_path))


def test_prepare_batch_rejects_invalid_or_duplicate_filenames(
    tmp_path: Path,
) -> None:
    with pytest.raises(ImageUploadError, match="expected cocktail"):
        prepare_batch(
            [IncomingImage(filename="martini.png", data=_png())],
            _settings(tmp_path),
        )

    with pytest.raises(ImageUploadError, match="duplicate"):
        prepare_batch(
            [
                IncomingImage(filename="cocktail-1.png", data=_png()),
                IncomingImage(filename="cocktail-1.webp", data=_png()),
            ],
            _settings(tmp_path),
        )


def test_upload_client_chunks_files_ten_at_a_time() -> None:
    paths = [Path(f"cocktail-{index}.png") for index in range(1, 24)]

    assert [len(batch) for batch in _chunks(paths)] == [10, 10, 3]


def test_upload_dry_run_can_report_an_empty_folder(tmp_path: Path) -> None:
    csv_path = tmp_path / "prompts.csv"
    csv_path.write_text(
        "id,image_filename\n1,cocktail-1.png\n",
        encoding="utf-8",
    )

    paths, missing = _load_image_paths(
        csv_path,
        tmp_path / "images",
        require_all=False,
        allow_empty=True,
    )

    assert paths == []
    assert missing == ["cocktail-1.png"]


def test_upload_client_finds_existing_jpeg_for_csv_png_name(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "prompts.csv"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    csv_path.write_text(
        "id,image_filename\n1,cocktail-1.png\n",
        encoding="utf-8",
    )
    jpeg = image_dir / "cocktail-1.jpeg"
    jpeg.write_bytes(_png())

    paths, missing = _load_image_paths(
        csv_path,
        image_dir,
        require_all=True,
    )

    assert paths == [jpeg]
    assert missing == []


def test_offline_script_persists_files_without_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "cocktail-1.png"
    source.write_bytes(_png())
    captured: list[IncomingImage] = []

    def fake_persist_batch(db, incoming, settings):
        assert db is not None
        assert settings.cocktail_image_output_dir == tmp_path / "media"
        captured.extend(incoming)
        return [
            {
                "cocktail_id": 1,
                "cocktail_name": "테스트",
                "source_filename": "cocktail-1.png",
                "image_url": "https://example.com/1-hash.webp",
                "thumbnail_url": "https://example.com/1-hash-thumb.webp",
            }
        ]

    monkeypatch.setattr(
        "scripts.upload_cocktail_images.persist_batch",
        fake_persist_batch,
    )

    result = _persist_paths(object(), _settings(tmp_path), [source])

    assert len(result) == 1
    assert captured[0].filename == "cocktail-1.png"


def test_admin_image_upload_route_is_not_registered() -> None:
    routes = {route.path for route in create_app().routes}

    assert "/admin/cocktail-images/batch" not in routes
