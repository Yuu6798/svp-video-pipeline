"""Tests for full planner -> image -> video pipeline orchestration (M4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svp_pipeline.pipeline import Pipeline
from tests.fixtures.fakes import FakeImageGeneratorQualityOnly as FakeImageGenerator
from tests.fixtures.fakes import FakePlanner, FakeVideoGenerator
from tests.fixtures.helpers import load_sample as _load
from tests.fixtures.mock_gemini import TINY_PNG_BYTES


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
    assert fake_image.calls[0] == "cheap"
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


def test_full_pipeline_reuses_existing_image_for_video(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    source_run = tmp_path / "source-run"
    source_run.mkdir()
    (source_run / "svp.json").write_text(svp.model_dump_json(indent=2), encoding="utf-8")
    source_image = source_run / "image.png"
    source_image.write_bytes(TINY_PNG_BYTES)
    fake_video = FakeVideoGenerator()
    pipeline = Pipeline(
        output_dir=tmp_path,
        video_generator=fake_video,  # type: ignore[arg-type]
    )

    result = pipeline.run("reuse full prompt", reuse_run_dir=source_run)
    log_data = json.loads(result.log_path.read_text(encoding="utf-8"))

    assert result.image_path is not None
    assert result.video_path is not None
    assert fake_video.image_paths == [result.image_path]
    assert result.image_path.read_bytes() == source_image.read_bytes()
    assert log_data["stages"]["planner"]["status"] == "reused"
    assert log_data["stages"]["image"]["status"] == "reused"
    assert log_data["stages"]["video"]["status"] == "ok"
