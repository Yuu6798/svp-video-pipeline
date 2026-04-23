"""Tests for pipeline orchestration (M3)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from svp_pipeline.generator.image import ImageResult
from svp_pipeline.pipeline import Pipeline
from svp_pipeline.schema import SVPVideo
from tests.fixtures.mock_gemini import TINY_PNG_BYTES

SAMPLES_DIR = Path(__file__).parent / "samples"


def _load(name: str) -> SVPVideo:
    return SVPVideo.model_validate_json((SAMPLES_DIR / name).read_text(encoding="utf-8"))


class FakePlanner:
    def __init__(self, svp: SVPVideo) -> None:
        self.svp = svp
        self.calls: list[tuple[str, int | None]] = []

    def plan(self, user_prompt: str, duration: int | None = None) -> SVPVideo:
        self.calls.append((user_prompt, duration))
        return self.svp


class FakePlannerWithEffectiveModel(FakePlanner):
    def __init__(self, svp: SVPVideo, effective_model: str) -> None:
        super().__init__(svp)
        self.model = effective_model


class FakeImageGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[SVPVideo, str]] = []

    def generate(self, svp: SVPVideo, resolution: str = "2K") -> ImageResult:
        self.calls.append((svp, resolution))
        return ImageResult(
            png_bytes=TINY_PNG_BYTES,
            cost_usd=0.08 if resolution == "2K" else 0.04,
            elapsed_sec=0.12,
            raw_prompt="prompt",
            model="gemini-3-pro-image-preview",
            resolution=resolution,  # type: ignore[arg-type]
            aspect_ratio="16:9",
        )


def test_pipeline_run_with_no_video(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    planner = FakePlanner(svp)
    image = FakeImageGenerator()
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner=planner,  # type: ignore[arg-type]
        image_generator=image,  # type: ignore[arg-type]
    )

    result = pipeline.run("prompt", duration=5, no_video=True)

    assert result.svp_path.exists()
    assert result.image_path is not None
    assert result.image_path.exists()
    assert result.log_path.exists()
    assert result.video_path is None


def test_output_directory_structure(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    planner = FakePlanner(svp)
    image = FakeImageGenerator()
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner=planner,  # type: ignore[arg-type]
        image_generator=image,  # type: ignore[arg-type]
    )

    result = pipeline.run("prompt", no_video=True)

    assert (result.output_dir / "svp.json").exists()
    assert (result.output_dir / "image.png").exists()
    assert (result.output_dir / "log.json").exists()


def test_log_json_format(tmp_path: Path) -> None:
    svp = _load("action_ninja.json")
    planner = FakePlanner(svp)
    image = FakeImageGenerator()
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner=planner,  # type: ignore[arg-type]
        image_generator=image,  # type: ignore[arg-type]
    )

    result = pipeline.run("ninja prompt", duration=6, no_video=True)
    log_data = json.loads(result.log_path.read_text(encoding="utf-8"))

    assert log_data["user_prompt"] == "ninja prompt"
    assert "timestamp" in log_data
    assert "planner" in log_data["stages"]
    assert "image" in log_data["stages"]
    assert "video" not in log_data["stages"]
    assert "total_cost_usd" in log_data
    assert "total_elapsed_sec" in log_data
    assert log_data["outputs"]["svp"] == "svp.json"
    assert log_data["outputs"]["image"] == "image.png"


def test_log_uses_effective_planner_model(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    planner = FakePlannerWithEffectiveModel(svp, effective_model="claude-opus-4-6")
    image = FakeImageGenerator()
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner_model="claude-opus-4-7",
        planner=planner,  # type: ignore[arg-type]
        image_generator=image,  # type: ignore[arg-type]
    )

    result = pipeline.run("planner model log", no_video=True)
    log_data = json.loads(result.log_path.read_text(encoding="utf-8"))

    assert log_data["stages"]["planner"]["model"] == "claude-opus-4-6"


def test_timestamp_directory_created(tmp_path: Path) -> None:
    svp = _load("still_life_macro.json")
    planner = FakePlanner(svp)
    image = FakeImageGenerator()
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner=planner,  # type: ignore[arg-type]
        image_generator=image,  # type: ignore[arg-type]
    )

    result = pipeline.run("still life", no_video=True)

    assert re.fullmatch(r"\d{8}-\d{6}(?:-\d{2})?", result.output_dir.name)


def test_dry_run_saves_svp_only(tmp_path: Path) -> None:
    svp = _load("still_life_macro.json")
    planner = FakePlanner(svp)
    image = FakeImageGenerator()
    pipeline = Pipeline(
        output_dir=tmp_path,
        dry_run=True,
        planner=planner,  # type: ignore[arg-type]
        image_generator=image,  # type: ignore[arg-type]
    )

    result = pipeline.run("dry run prompt", no_video=True)

    assert result.svp_path.exists()
    assert result.image_path is None
    assert not (result.output_dir / "image.png").exists()
    assert len(image.calls) == 0


def test_dry_run_log_records_estimated_cost(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    planner = FakePlanner(svp)
    pipeline = Pipeline(
        output_dir=tmp_path,
        dry_run=True,
        planner=planner,  # type: ignore[arg-type]
    )

    result = pipeline.run("dry run log prompt", no_video=True)
    log_data = json.loads(result.log_path.read_text(encoding="utf-8"))

    assert log_data["stages"]["image"]["status"] == "skipped_dry_run"
    assert log_data["stages"]["image"]["estimated_cost_usd"] > 0


def test_cheap_mode_uses_1k_resolution(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    planner = FakePlanner(svp)
    image = FakeImageGenerator()
    pipeline = Pipeline(
        output_dir=tmp_path,
        cheap_mode=True,
        planner=planner,  # type: ignore[arg-type]
        image_generator=image,  # type: ignore[arg-type]
    )

    pipeline.run("cheap mode prompt", no_video=True)

    assert len(image.calls) == 1
    assert image.calls[0][1] == "1K"
