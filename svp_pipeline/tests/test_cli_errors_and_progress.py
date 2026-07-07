"""Tests for svp-video CLI (error guidance and progress callback behavior)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import svp_pipeline.cli as cli_mod
from svp_pipeline.cli import app
from svp_pipeline.exceptions import ImageRefusalError, PlannerAPIError, VideoTimeoutError
from tests.fixtures.fakes import FakePipeline

runner = CliRunner()

# Rich/Typer の help 出力には端末判定によって ANSI 制御コードが挿入される
# ことがあり、`--duration` のようなトークンが raw output 上で連続文字列として
# 存在しなくなる。部分文字列マッチを行う前にこの正規表現で剥がして、
# 環境（ローカル/CI/TERM=dumb 等）依存性を排除する。
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


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


@pytest.fixture(autouse=True)
def _reset_fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    FakePipeline.instances = []
    FakePipeline.error = None
    monkeypatch.setattr(cli_mod, "Pipeline", FakePipeline)


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
