"""Tests for svp-video CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import svp_pipeline.cli as cli_mod
from svp_pipeline.cli import app
from svp_pipeline.exceptions import ImageRefusalError, PlannerAPIError, VideoTimeoutError
from svp_pipeline.pipeline import PipelineResult
from svp_pipeline.schema import SVPVideo
from tests.fixtures.mock_gemini import TINY_PNG_BYTES

SAMPLES_DIR = Path(__file__).parent / "samples"
runner = CliRunner()


def _load(name: str) -> SVPVideo:
    return SVPVideo.model_validate_json((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def _set_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_mod, "load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("FAL_KEY", "fal-test")


def _clear_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_mod, "load_dotenv", lambda: None)
    for key in (
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "FAL_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


class FakePipeline:
    instances: list[FakePipeline] = []
    error: Exception | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.run_calls: list[dict[str, Any]] = []
        FakePipeline.instances.append(self)

    def run(
        self,
        user_prompt: str,
        duration: int | None = None,
        no_video: bool = False,
        progress_callback=None,
    ) -> PipelineResult:
        self.run_calls.append(
            {"user_prompt": user_prompt, "duration": duration, "no_video": no_video}
        )
        if FakePipeline.error is not None:
            raise FakePipeline.error

        output_dir = Path(self.kwargs["output_dir"]) / "20260425-000000"
        output_dir.mkdir(parents=True, exist_ok=True)
        svp_path = output_dir / "svp.json"
        image_path = output_dir / "image.png"
        video_path = None if no_video or self.kwargs.get("dry_run") else output_dir / "video.mp4"
        log_path = output_dir / "log.json"

        svp = _load("shibuya_dusk.json")
        svp_path.write_text(svp.model_dump_json(indent=2), encoding="utf-8")
        if not self.kwargs.get("dry_run"):
            image_path.write_bytes(TINY_PNG_BYTES)
            if video_path is not None:
                video_path.write_bytes(b"mp4")

        if progress_callback is not None:
            progress_callback("planner_start", {"model": self.kwargs["planner_model"]})
            progress_callback("planner_done", {"elapsed_sec": 0.1, "cost_usd": 0.012})
            if not self.kwargs.get("dry_run"):
                progress_callback(
                    "image_start",
                    {"backend": self.kwargs.get("image_backend", "gemini")},
                )
                progress_callback("image_done", {"elapsed_sec": 0.2, "cost_usd": 0.08})
                if not no_video:
                    progress_callback(
                        "video_start",
                        {"tier": "standard", "resolution": "720p", "duration": duration or 5},
                    )
                    progress_callback("video_done", {"elapsed_sec": 0.3, "cost_usd": 1.512})

        log_data = {
            "stages": {
                "planner": {"cost_usd": 0.012},
                "image": {"estimated_cost_usd": 0.08},
            },
            "total_cost_usd": 1.604,
        }
        if not no_video:
            log_data["stages"]["video"] = {"estimated_cost_usd": 1.512}
        log_path.write_text(json.dumps(log_data), encoding="utf-8")
        return PipelineResult(
            output_dir=output_dir,
            svp=svp,
            svp_path=svp_path,
            image_path=None if self.kwargs.get("dry_run") else image_path,
            video_path=video_path,
            log_path=log_path,
            total_cost_usd=1.604,
            total_elapsed_sec=0.6,
        )


@pytest.fixture(autouse=True)
def _reset_fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    FakePipeline.instances = []
    FakePipeline.error = None
    monkeypatch.setattr(cli_mod, "Pipeline", FakePipeline)


def test_help_shows_all_options() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for option in (
        "--duration",
        "--output",
        "--planner-model",
        "--image-backend",
        "--cheap",
        "--dry-run",
        "--no-video",
        "--verbose",
        "--version",
    ):
        assert option in result.output


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "svp-video" in result.output


def test_prompt_required() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "prompt" in result.output.lower()


def test_duration_out_of_range_fails() -> None:
    result = runner.invoke(app, ["prompt", "--duration", "3"])
    assert result.exit_code != 0


def test_default_planner_model_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    monkeypatch.setenv("DEFAULT_PLANNER_MODEL", "claude-haiku-4-5")
    result = runner.invoke(app, ["prompt", "--dry-run", "--output", str(tmp_path)])

    assert result.exit_code == 0
    assert FakePipeline.instances[0].kwargs["planner_model"] == "claude-haiku-4-5"


def test_cli_flag_overrides_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    monkeypatch.setenv("DEFAULT_IMAGE_BACKEND", "openai")
    result = runner.invoke(
        app,
        ["prompt", "--dry-run", "--image-backend", "gemini", "--output", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert FakePipeline.instances[0].kwargs["image_backend"] == "gemini"


def test_missing_anthropic_key_fails_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_required_keys(monkeypatch)
    result = runner.invoke(app, ["prompt", "--dry-run"])

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


def test_openai_backend_requires_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_required_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    result = runner.invoke(app, ["prompt", "--image-backend", "openai", "--dry-run"])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_dry_run_does_not_require_fal_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    monkeypatch.delenv("FAL_KEY", raising=False)
    result = runner.invoke(app, ["prompt", "--dry-run", "--output", str(tmp_path)])

    assert result.exit_code == 0


def test_no_video_does_not_require_fal_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    monkeypatch.delenv("FAL_KEY", raising=False)
    result = runner.invoke(app, ["prompt", "--no-video", "--output", str(tmp_path)])

    assert result.exit_code == 0


def test_cheap_mode_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    result = runner.invoke(app, ["prompt", "--cheap", "--dry-run", "--output", str(tmp_path)])

    assert result.exit_code == 0
    assert FakePipeline.instances[0].kwargs["cheap_mode"] is True


def test_dry_run_skips_image_and_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    result = runner.invoke(app, ["prompt", "--dry-run", "--output", str(tmp_path)])

    assert result.exit_code == 0
    assert FakePipeline.instances[0].kwargs["dry_run"] is True


def test_no_video_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    result = runner.invoke(app, ["prompt", "--no-video", "--output", str(tmp_path)])

    assert result.exit_code == 0
    assert FakePipeline.instances[0].run_calls[0]["no_video"] is True


def test_backend_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    result = runner.invoke(
        app,
        ["prompt", "--image-backend", "openai", "--no-video", "--output", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert FakePipeline.instances[0].kwargs["image_backend"] == "openai"


def test_planner_api_error_shows_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    FakePipeline.error = PlannerAPIError("upstream")
    result = runner.invoke(app, ["prompt", "--dry-run", "--output", str(tmp_path)])

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


def test_image_refusal_shows_guidance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    FakePipeline.error = ImageRefusalError("refused")
    result = runner.invoke(app, ["prompt", "--no-video", "--output", str(tmp_path)])

    assert result.exit_code == 1
    assert "forbidden" in result.output


def test_video_timeout_suggests_cheap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    FakePipeline.error = VideoTimeoutError("timeout")
    result = runner.invoke(app, ["prompt", "--output", str(tmp_path)])

    assert result.exit_code == 1
    assert "--cheap" in result.output


def test_verbose_shows_traceback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    FakePipeline.error = VideoTimeoutError("timeout")
    result = runner.invoke(app, ["prompt", "--verbose", "--output", str(tmp_path)])

    assert result.exit_code == 1
    assert "Traceback" in result.output


def test_no_verbose_hides_traceback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    FakePipeline.error = VideoTimeoutError("timeout")
    result = runner.invoke(app, ["prompt", "--output", str(tmp_path)])

    assert result.exit_code == 1
    assert "Traceback" not in result.output


def test_progress_callback_called_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    result = runner.invoke(app, ["prompt", "--output", str(tmp_path)])

    assert result.exit_code == 0
    assert "Generating SVP" in result.output
    assert "Image generated" in result.output
    assert "Video generated" in result.output


def test_dry_run_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    result = runner.invoke(app, ["prompt", "--dry-run", "--output", str(tmp_path)])

    assert result.exit_code == 0
    assert "SVP generated" in result.output
    assert "Image generated" not in result.output
