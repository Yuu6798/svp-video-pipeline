"""Tests for svp-video CLI."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
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

# Rich/Typer の help 出力には端末判定によって ANSI 制御コードが挿入される
# ことがあり、`--duration` のようなトークンが raw output 上で連続文字列として
# 存在しなくなる。部分文字列マッチを行う前にこの正規表現で剥がして、
# 環境（ローカル/CI/TERM=dumb 等）依存性を排除する。
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


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
        reference_image_path: Path | None = None,
        reference_crop: int | None = None,
        separate_character_bg: bool = False,
        progress_callback=None,
    ) -> PipelineResult:
        self.run_calls.append(
            {
                "user_prompt": user_prompt,
                "duration": duration,
                "no_video": no_video,
                "reference_image_path": reference_image_path,
                "reference_crop": reference_crop,
                "separate_character_bg": separate_character_bg,
            }
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
    output = _strip_ansi(result.output)
    for option in (
        "--duration",
        "--output",
        "--planner-model",
        "--image-backend",
        "reference image",
        "--reference-crop",
        "separately",
        "--no-character",
        "--cheap",
        "--dry-run",
        "--no-video",
        "--archive-drive",
        "--verbose",
        "--version",
    ):
        assert option in output


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "svp-video" in result.output


def test_prompt_required() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "prompt" in result.output.lower()


def test_unquoted_multi_word_prompt_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    result = runner.invoke(
        app,
        [
            "--dry-run",
            "--output",
            str(tmp_path),
            "cyberpunk",
            "rainy",
            "neon",
            "city",
        ],
    )

    assert result.exit_code == 0
    assert FakePipeline.instances[0].run_calls[0]["user_prompt"] == "cyberpunk rainy neon city"


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


def test_archive_drive_flag_archives_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    archived_dirs: list[Path] = []

    def fake_archive(result: PipelineResult) -> SimpleNamespace:
        archived_dirs.append(result.output_dir)
        return SimpleNamespace(
            already_archived=False,
            uploaded_files={"image": "https://drive/image"},
            skipped_files=[],
            drive_folder_url="https://drive/folder",
            log_path=result.log_path,
        )

    monkeypatch.setattr(cli_mod, "_archive_outputs_to_drive", fake_archive)

    result = runner.invoke(
        app,
        ["prompt", "--no-video", "--archive-drive", "--output", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert archived_dirs == [tmp_path / "20260425-000000"]
    assert "Drive archive: uploaded 1 file" in result.output


def test_archive_drive_skipped_for_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)

    def fail_archive(_result: PipelineResult) -> None:
        raise AssertionError("archive should not run for dry-run")

    monkeypatch.setattr(cli_mod, "_archive_outputs_to_drive", fail_archive)

    result = runner.invoke(
        app,
        ["prompt", "--dry-run", "--archive-drive", "--output", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "--archive-drive skipped for --dry-run" in result.output


def test_archive_drive_failure_reports_local_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)

    def fail_archive(result: PipelineResult) -> None:
        raise RuntimeError(
            f"Archive to Drive failed: boom. Local outputs remain at: {result.output_dir}"
        )

    monkeypatch.setattr(cli_mod, "_archive_outputs_to_drive", fail_archive)

    result = runner.invoke(
        app,
        ["prompt", "--no-video", "--archive-drive", "--output", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "Archive to Drive failed" in result.output
    assert "Local outputs remain at" in result.output
    assert "20260425-000000" in result.output


def test_backend_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    result = runner.invoke(
        app,
        ["prompt", "--image-backend", "openai", "--no-video", "--output", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert FakePipeline.instances[0].kwargs["image_backend"] == "openai"


def test_reference_image_flag_passes_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    reference = tmp_path / "reference.png"
    reference.write_bytes(TINY_PNG_BYTES)

    result = runner.invoke(
        app,
        [
            "prompt",
            "--no-video",
            "--reference-image",
            str(reference),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert FakePipeline.instances[0].run_calls[0]["reference_image_path"] == reference


def test_reference_crop_flag_passes_crop_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    reference = tmp_path / "reference.png"
    reference.write_bytes(TINY_PNG_BYTES)

    result = runner.invoke(
        app,
        [
            "prompt",
            "--no-video",
            "--reference-image",
            str(reference),
            "--reference-crop",
            "5",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert FakePipeline.instances[0].run_calls[0]["reference_crop"] == 5


def test_separate_character_bg_flag_passes_to_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    reference = tmp_path / "reference.png"
    reference.write_bytes(TINY_PNG_BYTES)

    result = runner.invoke(
        app,
        [
            "prompt",
            "--no-video",
            "--image-backend",
            "openai",
            "--reference-image",
            str(reference),
            "--separate-character-bg",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert FakePipeline.instances[0].run_calls[0]["separate_character_bg"] is True


def test_separate_character_bg_requires_openai_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    reference = tmp_path / "reference.png"
    reference.write_bytes(TINY_PNG_BYTES)

    result = runner.invoke(
        app,
        [
            "prompt",
            "--no-video",
            "--reference-image",
            str(reference),
            "--separate-character-bg",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "requires --image-backend openai" in result.output


def test_reference_crop_requires_reference_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)

    result = runner.invoke(
        app,
        ["prompt", "--reference-crop", "1", "--dry-run", "--output", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "--reference-crop requires --reference-image" in result.output


def test_missing_reference_image_fails_gracefully(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    missing = tmp_path / "missing.png"

    result = runner.invoke(
        app,
        ["prompt", "--reference-image", str(missing), "--dry-run", "--output", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "Reference image not found" in result.output


def test_no_character_lock_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    result = runner.invoke(
        app,
        ["prompt", "--no-character-lock", "--dry-run", "--output", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert FakePipeline.instances[0].kwargs["character_lock"] is False


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
