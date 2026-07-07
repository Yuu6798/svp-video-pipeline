"""Tests for svp-video CLI generation flags (dry-run/cheap/no-video/backend/reference/reuse)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import svp_pipeline.cli as cli_mod
from svp_pipeline.cli import app
from tests.fixtures.fakes import FakePipeline
from tests.fixtures.helpers import load_sample as _load
from tests.fixtures.mock_gemini import TINY_PNG_BYTES

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


def test_backend_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    result = runner.invoke(
        app,
        ["prompt", "--image-backend", "openai", "--no-video", "--output", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert FakePipeline.instances[0].kwargs["image_backend"] == "openai"

def test_failure_preset_flag_passes_to_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)

    result = runner.invoke(
        app,
        [
            "prompt",
            "--failure-preset",
            "single-character-weapon-clean-bg",
            "--no-video",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert FakePipeline.instances[0].run_calls[0]["failure_presets"] == [
        "single-character-weapon-clean-bg"
    ]

def test_from_svp_skips_anthropic_key_requirement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    svp_path = tmp_path / "source.svp.json"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")

    result = runner.invoke(
        app,
        ["--from-svp", str(svp_path), "--no-video", "--output", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert FakePipeline.instances[0].run_calls[0]["from_svp_path"] == svp_path

def test_reuse_run_skips_planner_and_image_key_requirements(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    reuse_run = tmp_path / "source-run"
    reuse_run.mkdir()
    (reuse_run / "svp.json").write_text(
        _load("shibuya_dusk.json").model_dump_json(),
        encoding="utf-8",
    )
    (reuse_run / "composite.png").write_bytes(TINY_PNG_BYTES)

    result = runner.invoke(
        app,
        [
            "--reuse-run",
            str(reuse_run),
            "--reuse-image",
            "composite",
            "--no-video",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    call = FakePipeline.instances[0].run_calls[0]
    assert call["reuse_run_dir"] == reuse_run
    assert call["reuse_image"] == "composite"

def test_reuse_run_rejects_reference_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    reuse_run = tmp_path / "source-run"
    reuse_run.mkdir()
    (reuse_run / "svp.json").write_text(
        _load("shibuya_dusk.json").model_dump_json(),
        encoding="utf-8",
    )
    (reuse_run / "image.png").write_bytes(TINY_PNG_BYTES)
    reference = tmp_path / "reference.png"
    reference.write_bytes(TINY_PNG_BYTES)

    result = runner.invoke(
        app,
        [
            "prompt",
            "--reuse-run",
            str(reuse_run),
            "--reference-image",
            str(reference),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "do not combine" in result.output

def test_invalid_reuse_image_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    result = runner.invoke(
        app,
        ["prompt", "--reuse-image", "mask", "--output", str(tmp_path)],
    )

    assert result.exit_code == 2
    assert "Invalid --reuse-image" in result.output

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
