from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest

from app.image_generation.core import CocktailSource, ImageGenerationSettings
from app.image_generation.errors import GenerationFatalError
from app.image_generation.pipeline import (
    CocktailImagePipeline,
    RunOptions,
    RunSummary,
)


class FakeGateway:
    def __init__(self) -> None:
        self.expand_calls = 0
        self.image_calls = 0

    @property
    def image_config(self) -> dict[str, object]:
        return {"provider": "fake", "mode": "base"}

    def verify_model(self) -> None:
        return None

    def expand_prompt(self, prompt: str) -> str:
        self.expand_calls += 1
        assert "Cocktail content facts:" in prompt
        return (
            "A ruby cocktail with realistic clear liquid, crushed ice, fine water "
            "condensation, a distinct citrus layer, an orange slice, and one red "
            "maraschino cherry beside the rim."
        )

    def generate_image(self, prompt: str) -> bytes:
        self.image_calls += 1
        raise AssertionError("The manual workflow must not call NVIDIA")


class RecordingPipeline(CocktailImagePipeline):
    def __init__(
        self,
        settings: ImageGenerationSettings,
        gateway: FakeGateway,
    ) -> None:
        super().__init__(settings, None, gateway, gateway)  # type: ignore[arg-type]
        self.updated_urls: list[tuple[int, str]] = []

    def _load_sources(
        self,
        cocktail_ids: Iterable[int],
    ) -> Iterator[CocktailSource]:
        source = _source()
        ids = tuple(cocktail_ids)
        if not ids or source.id in ids:
            yield source

    def _update_image_url(self, cocktail_id: int, image_url: str) -> None:
        self.updated_urls.append((cocktail_id, image_url))


def _settings(tmp_path: Path) -> ImageGenerationSettings:
    return ImageGenerationSettings(
        gemini_api_key="test-key",
        nvidia_api_key="test-nvidia-key",
        cocktail_image_output_dir=tmp_path / "media",
        cocktail_image_state_dir=tmp_path / "state",
    )


def _source(image_url: str | None = None) -> CocktailSource:
    return CocktailSource(
        id=7,
        name="테스트 칵테일",
        name_en="Test Cocktail",
        glass="쿠페 글라스",
        base_tag="gin",
        recipe=("재료를 셰이크한다.", "쿠페 글라스에 따른다."),
        image_url=image_url,
    )


def test_export_prompts_writes_csv_without_calling_nvidia(
    tmp_path: Path,
) -> None:
    gateway = FakeGateway()
    pipeline = RecordingPipeline(_settings(tmp_path), gateway)
    output = tmp_path / "cocktail-image-prompts.csv"

    summary = pipeline.export_prompts(output, limit=1)

    assert summary.exported == 1
    assert summary.gemini_generated == 1
    assert gateway.expand_calls == 1
    assert gateway.image_calls == 0
    with output.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["id"] == "7"
    assert rows[0]["cocktail_name"] == "테스트 칵테일"
    assert rows[0]["cocktail_name_en"] == "Test Cocktail"
    assert rows[0]["image_filename"] == "cocktail-7.png"
    assert "Drink:" in rows[0]["final_image_prompt"]
    assert "Glass:" in rows[0]["final_image_prompt"]
    assert "Background:" in rows[0]["final_image_prompt"]
    assert "Style:" in rows[0]["final_image_prompt"]
    assert len(rows[0]["final_image_prompt"]) <= 800


def test_export_prompts_reuses_cached_gemini_contents(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    first_gateway = FakeGateway()
    RecordingPipeline(settings, first_gateway).export_prompts(tmp_path / "first.csv")

    second_gateway = FakeGateway()
    summary = RecordingPipeline(settings, second_gateway).export_prompts(
        tmp_path / "second.csv",
        cached_only=True,
    )

    assert summary.exported == 1
    assert summary.cache_hits == 1
    assert summary.gemini_generated == 0
    assert second_gateway.expand_calls == 0
    assert second_gateway.image_calls == 0


def test_cached_only_skips_uncached_prompt(tmp_path: Path) -> None:
    gateway = FakeGateway()
    pipeline = RecordingPipeline(_settings(tmp_path), gateway)

    summary = pipeline.export_prompts(
        tmp_path / "cached-only.csv",
        cached_only=True,
    )

    assert summary.exported == 0
    assert summary.skipped == 1
    assert gateway.expand_calls == 0
    assert gateway.image_calls == 0


def test_legacy_run_stops_before_nvidia_call(tmp_path: Path) -> None:
    gateway = FakeGateway()
    pipeline = RecordingPipeline(_settings(tmp_path), gateway)

    with pytest.raises(GenerationFatalError, match="disabled"):
        pipeline._process_source(_source(), RunOptions(), RunSummary())

    assert gateway.expand_calls == 1
    assert gateway.image_calls == 0
    state = pipeline.state_store.load(7)
    assert state is not None
    assert state.status == "failed"
    assert state.expanded_prompt is not None


def test_dry_run_does_not_write_state_or_call_models(tmp_path: Path) -> None:
    gateway = FakeGateway()
    pipeline = RecordingPipeline(_settings(tmp_path), gateway)
    summary = RunSummary()

    action = pipeline._process_source(
        _source(),
        RunOptions(dry_run=True),
        summary,
    )

    assert action == "dry_run"
    assert summary.dry_run == 1
    assert gateway.expand_calls == 0
    assert gateway.image_calls == 0
    assert pipeline.state_store.load(7) is None
    assert pipeline.updated_urls == []
