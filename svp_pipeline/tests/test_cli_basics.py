"""Tests for svp-video CLI (basics: help/version/prompt/env/API-key requirements)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import svp_pipeline.cli as cli_mod
from svp_pipeline.cli import app
from tests.fixtures.fakes import FakePipeline
from tests.fixtures.helpers import load_sample as _load

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
        "--from-svp",
        "--reuse-run",
        "--reuse-image",
        "--failure-preset",
        "--audit-image",
        "--audit-backend",
        "--compare-image",
        "--audit-repair",
        "object",
        "contact",
        "viewer",
        "pose",
        "failure",
        "failure-preset",
        "regenerate",
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

def test_prompt_can_be_omitted_with_from_svp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")

    result = runner.invoke(
        app,
        ["--from-svp", str(svp_path), "--no-video", "--output", str(tmp_path)],
    )

    assert result.exit_code == 0
    call = FakePipeline.instances[0].run_calls[0]
    assert call["from_svp_path"] == svp_path
    assert "regenerate from SVP" in call["user_prompt"]

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
