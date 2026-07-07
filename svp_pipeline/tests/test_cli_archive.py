"""Tests for svp-video CLI (archive-to-drive flag behavior)."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import svp_pipeline.cli as cli_mod
from svp_pipeline.cli import app
from svp_pipeline.pipeline import PipelineResult
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
