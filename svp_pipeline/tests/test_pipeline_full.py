"""Tests for full planner -> image -> video pipeline orchestration (M4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svp_pipeline.exceptions import VideoDownloadError
from svp_pipeline.generator.image import ImageResult
from svp_pipeline.generator.video import VideoResult
from svp_pipeline.pipeline import Pipeline
from svp_pipeline.schema import SVPVideo
from tests.fixtures.mock_fal import TINY_MP4_BYTES
from tests.fixtures.mock_gemini import TINY_PNG_BYTES

SAMPLES_DIR = Path(__file__).parent / "samples"


def _load(name: str) -> SVPVideo:
    return SVPVideo.model_validate_json((SAMPLES_DIR / name).read_text(encoding="utf-8"))


class FakePlanner:
    def __init__(self, svp: SVPVideo) -> None:
        self.svp = svp
        self.calls: list[tuple[str, int | None]] = []
        self.model = "claude-haiku-4-5"

    def plan(self, user_prompt: str, duration: int | None = None) -> SVPVideo:
        self.calls.append((user_prompt, duration))
        return self.svp


class FakeImageGenerator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, svp: SVPVideo, resolution: str = "2K") -> ImageResult:
        self.calls.append(resolution)
        return ImageResult(
            png_bytes=TINY_PNG_BYTES,
            cost_usd=0.08 if resolution == "2K" else 0.04,
            elapsed_sec=0.11,
            raw_prompt="image prompt",
            model="gemini-3-pro-image-preview",
            resolution=resolution,  # type: ignore[arg-type]
            aspect_ratio=svp.composition_layer.aspect_ratio,
        )


class FakeVideoGenerator:
    def __init__(self, tier: str = "standard", fail_download: bool = False) -> None:
        self.tier = tier
        self.fail_download = fail_download
        self.calls: list[tuple[str, str, int]] = []

    def generate(
        self,
        svp: SVPVideo,
        image_path: Path,
        output_path: Path,
        resolution: str = "720p",
    ) -> VideoResult:
        self.calls.append((self.tier, resolution, svp.motion_layer.duration_seconds))
        if self.fail_download:
            raise VideoDownloadError(
                "download failed",
                mp4_url="https://mock.fal.media/download-failed.mp4",
            )
        output_path.write_bytes(TINY_MP4_BYTES)
        return VideoResult(
            mp4_path=output_path,
            cost_usd=1.512 if resolution == "720p" else 0.5375,
            elapsed_sec=0.9,
            raw_prompt="video prompt",
            tier=self.tier,  # type: ignore[arg-type]
            resolution=resolution,  # type: ignore[arg-type]
            aspect_ratio=svp.composition_layer.aspect_ratio,
            duration_seconds=svp.motion_layer.duration_seconds,
            fal_request_id="req_mock",
            mp4_url="https://mock.fal.media/video.mp4",
        )

    def _calculate_cost(self, tier: str, resolution: str, duration_seconds: int) -> float:
        if tier == "fast" and resolution == "480p":
            return 0.1075 * duration_seconds
        return 0.3024 * duration_seconds


def test_full_pipeline_with_video(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner=FakePlanner(svp),  # type: ignore[arg-type]
        image_generator=FakeImageGenerator(),  # type: ignore[arg-type]
        video_generator=FakeVideoGenerator(),  # type: ignore[arg-type]
    )

    result = pipeline.run("prompt", duration=5, no_video=False)

    assert (result.output_dir / "svp.json").exists()
    assert (result.output_dir / "image.png").exists()
    assert (result.output_dir / "video.mp4").exists()
    assert (result.output_dir / "log.json").exists()
    assert result.video_path is not None


def test_log_json_contains_all_three_stages(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner=FakePlanner(svp),  # type: ignore[arg-type]
        image_generator=FakeImageGenerator(),  # type: ignore[arg-type]
        video_generator=FakeVideoGenerator(),  # type: ignore[arg-type]
    )

    result = pipeline.run("prompt", duration=5, no_video=False)
    log_data = json.loads(result.log_path.read_text(encoding="utf-8"))

    assert "planner" in log_data["stages"]
    assert "image" in log_data["stages"]
    assert "video" in log_data["stages"]


def test_total_cost_is_sum_of_three_stages(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner=FakePlanner(svp),  # type: ignore[arg-type]
        image_generator=FakeImageGenerator(),  # type: ignore[arg-type]
        video_generator=FakeVideoGenerator(),  # type: ignore[arg-type]
    )

    result = pipeline.run("prompt", duration=5, no_video=False)
    assert result.total_cost_usd == pytest.approx(0.012 + 0.08 + 1.512)


def test_default_standard_tier(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    fake_video = FakeVideoGenerator(tier="standard")
    pipeline = Pipeline(
        output_dir=tmp_path,
        cheap_mode=False,
        planner=FakePlanner(svp),  # type: ignore[arg-type]
        image_generator=FakeImageGenerator(),  # type: ignore[arg-type]
        video_generator=fake_video,  # type: ignore[arg-type]
    )

    pipeline.run("prompt", duration=5, no_video=False)
    assert fake_video.calls[0][0] == "standard"
    assert fake_video.calls[0][1] == "720p"


def test_cheap_mode_fast_tier_and_480p(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    fake_image = FakeImageGenerator()
    fake_video = FakeVideoGenerator(tier="fast")
    pipeline = Pipeline(
        output_dir=tmp_path,
        cheap_mode=True,
        planner=FakePlanner(svp),  # type: ignore[arg-type]
        image_generator=fake_image,  # type: ignore[arg-type]
        video_generator=fake_video,  # type: ignore[arg-type]
    )

    pipeline.run("prompt", duration=5, no_video=False)
    assert fake_image.calls[0] == "1K"
    assert fake_video.calls[0][0] == "fast"
    assert fake_video.calls[0][1] == "480p"


def test_no_video_true_skips_video(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    fake_video = FakeVideoGenerator()
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner=FakePlanner(svp),  # type: ignore[arg-type]
        image_generator=FakeImageGenerator(),  # type: ignore[arg-type]
        video_generator=fake_video,  # type: ignore[arg-type]
    )

    result = pipeline.run("prompt", duration=5, no_video=True)
    log_data = json.loads(result.log_path.read_text(encoding="utf-8"))

    assert "video" not in log_data["stages"]
    assert len(fake_video.calls) == 0


def test_no_video_false_default(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    fake_video = FakeVideoGenerator()
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner=FakePlanner(svp),  # type: ignore[arg-type]
        image_generator=FakeImageGenerator(),  # type: ignore[arg-type]
        video_generator=fake_video,  # type: ignore[arg-type]
    )

    pipeline.run("prompt", duration=5)
    assert len(fake_video.calls) == 1


def test_missing_fal_key_fails_before_image_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svp = _load("shibuya_dusk.json")
    fake_image = FakeImageGenerator()
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner=FakePlanner(svp),  # type: ignore[arg-type]
        image_generator=fake_image,  # type: ignore[arg-type]
    )
    monkeypatch.delenv("FAL_KEY", raising=False)

    with pytest.raises(ValueError, match="FAL_KEY is required"):
        pipeline.run("prompt", duration=5)

    assert len(fake_image.calls) == 0


def test_video_download_failure_preserves_others(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner=FakePlanner(svp),  # type: ignore[arg-type]
        image_generator=FakeImageGenerator(),  # type: ignore[arg-type]
        video_generator=FakeVideoGenerator(fail_download=True),  # type: ignore[arg-type]
    )

    result = pipeline.run("prompt", duration=5, no_video=False)
    assert (result.output_dir / "svp.json").exists()
    assert (result.output_dir / "image.png").exists()
    assert not (result.output_dir / "video.mp4").exists()
    assert result.video_path is None


def test_video_download_failure_log_records_url(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner=FakePlanner(svp),  # type: ignore[arg-type]
        image_generator=FakeImageGenerator(),  # type: ignore[arg-type]
        video_generator=FakeVideoGenerator(fail_download=True),  # type: ignore[arg-type]
    )

    result = pipeline.run("prompt", duration=5, no_video=False)
    log_data = json.loads(result.log_path.read_text(encoding="utf-8"))

    assert log_data["stages"]["video"]["status"] == "download_failed"
    assert log_data["stages"]["video"]["mp4_url"] == "https://mock.fal.media/download-failed.mp4"
    assert log_data["outputs"]["video"] is None


def test_dry_run_default_includes_video_estimate(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    pipeline = Pipeline(
        output_dir=tmp_path,
        dry_run=True,
        planner=FakePlanner(svp),  # type: ignore[arg-type]
        image_generator=FakeImageGenerator(),  # type: ignore[arg-type]
        video_generator=FakeVideoGenerator(),  # type: ignore[arg-type]
    )

    result = pipeline.run("prompt", duration=5)
    log_data = json.loads(result.log_path.read_text(encoding="utf-8"))

    assert log_data["stages"]["image"]["status"] == "skipped_dry_run"
    assert log_data["stages"]["video"]["status"] == "skipped_dry_run"
    assert log_data["stages"]["video"]["estimated_cost_usd"] == pytest.approx(1.512)
    assert result.total_cost_usd == pytest.approx(0.012 + 0.08 + 1.512)
