"""Tests for svp-video CLI."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image
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


def _write_png(path: Path, color: tuple[int, int, int] = (0, 0, 0)) -> None:
    Image.new("RGB", (2, 2), color=color).save(path)


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
        from_svp_path: Path | None = None,
        reuse_run_dir: Path | None = None,
        reuse_image: str | None = None,
        failure_presets: list[str | Path] | None = None,
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
                "from_svp_path": from_svp_path,
                "reuse_run_dir": reuse_run_dir,
                "reuse_image": reuse_image,
                "failure_presets": failure_presets,
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
            _write_png(image_path)
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


def test_repair_from_rpe_writes_proposal_without_api_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    observed_rpe = tmp_path / "observed_rpe.json"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    observed_rpe.write_text(
        json.dumps(
            {
                "missing": ["single character remains centered"],
                "violations": ["sword-like background reflection"],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--repair-from-rpe",
            str(observed_rpe),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Semantic repair proposal written" in result.output
    repair_dirs = list(tmp_path.glob("repair-*"))
    assert len(repair_dirs) == 1
    assert (repair_dirs[0] / "semantic_diff.json").exists()
    assert (repair_dirs[0] / "repair_proposal.json").exists()
    assert (repair_dirs[0] / "target_svp.proposed.json").exists()


def test_repair_from_rpe_ignores_bad_generation_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    monkeypatch.setenv("DEFAULT_PLANNER_MODEL", "not-a-planner")
    monkeypatch.setenv("DEFAULT_IMAGE_BACKEND", "not-an-image-backend")
    svp_path = tmp_path / "source.svp.json"
    observed_rpe = tmp_path / "observed_rpe.json"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    observed_rpe.write_text(
        json.dumps({"violations": ["sword-like background reflection"]}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--repair-from-rpe",
            str(observed_rpe),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Semantic repair proposal written" in result.output


def test_repair_from_rpe_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    observed_rpe = tmp_path / "observed_rpe.json"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    observed_rpe.write_text("{not-json", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--repair-from-rpe",
            str(observed_rpe),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "Invalid JSON" in result.output


def test_repair_from_rpe_requires_svp_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    observed_rpe = tmp_path / "observed_rpe.json"
    observed_rpe.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        ["--repair-from-rpe", str(observed_rpe), "--output", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "requires --from-svp or --reuse-run" in result.output


def test_audit_image_writes_observed_rpe_without_api_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(image)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--audit-observed",
            "silver ponytail is visible",
            "--audit-missing",
            "red eyes are weak",
            "--audit-violation",
            "sword-like background reflection",
            "--audit-object-state",
            "scabbard: separate dark sheath at upper-left",
            "--audit-contact-graph",
            "left_hand -> scabbard",
            "--audit-viewer-contact-graph",
            "viewer_left_hand -> katana_handle",
            "--audit-anatomical-contact-graph",
            "character_right_hand -> katana_handle",
            "--audit-pose-intent",
            "unsheathing / draw-pose",
            "--audit-failure-mode",
            "left hand grips the wrong object",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Image RPE audit written" in result.output
    audit_dirs = list(tmp_path.glob("audit-*"))
    assert len(audit_dirs) == 1
    observed = json.loads((audit_dirs[0] / "observed_rpe.json").read_text(encoding="utf-8"))
    assert observed["missing"] == ["red eyes are weak"]
    assert observed["violations"] == ["sword-like background reflection"]
    assert observed["state"]["object_graph"] == ["scabbard: separate dark sheath at upper-left"]
    assert observed["state"]["contact_graph"] == ["left_hand -> scabbard"]
    assert observed["state"]["viewer_contact_graph"] == ["viewer_left_hand -> katana_handle"]
    assert observed["state"]["anatomical_contact_graph"] == [
        "character_right_hand -> katana_handle"
    ]
    assert observed["state"]["pose_intent"] == "unsheathing / draw-pose"
    assert observed["state"]["failure_modes"] == ["left hand grips the wrong object"]
    assert "expected_identity" in observed["state"]


def test_audit_image_ignores_bad_generation_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    monkeypatch.setenv("DEFAULT_PLANNER_MODEL", "not-a-planner")
    monkeypatch.setenv("DEFAULT_IMAGE_BACKEND", "not-an-image-backend")
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(image)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--audit-violation",
            "sword-like background reflection",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Image RPE audit written" in result.output


def test_audit_image_with_repair_writes_proposed_svp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(image)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--audit-violation",
            "sword-like background reflection",
            "--audit-repair",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Proposed SVP" in result.output
    assert list(tmp_path.glob("audit-*/repair-*/target_svp.proposed.json"))


def test_audit_image_with_repair_extracts_failure_preset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(image)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--audit-violation",
            "katana reflection appears as a second sword in the wet background",
            "--audit-repair",
            "--extract-failure-preset",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Failure preset candidate" in result.output
    candidates = list(tmp_path.glob("audit-*/repair-*/failure_preset.candidate.json"))
    assert len(candidates) == 1
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "SVP.failure_preset.v1"
    assert payload["failure_taxonomy"]


def test_extract_failure_preset_requires_audit_repair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(image)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--extract-failure-preset",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "--extract-failure-preset requires --audit-repair" in result.output


def test_audit_image_with_compare_writes_visual_comparison(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    source = tmp_path / "source.png"
    candidate = tmp_path / "candidate.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(source)
    _write_png(candidate)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(source),
            "--compare-image",
            str(candidate),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Visual comparison" in result.output
    comparison = list(tmp_path.glob("audit-*/visual_comparison.json"))
    assert len(comparison) == 1
    payload = json.loads(comparison[0].read_text(encoding="utf-8"))
    assert payload["dimensions_equal"] is True
    assert payload["pixel_rms"] == 0.0


def test_audit_regenerate_requires_repair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(image)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--audit-regenerate",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "--audit-regenerate requires --audit-repair" in result.output


def test_audit_repair_regenerate_writes_pipeline_output_and_comparison(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(source, color=(10, 10, 10))
    _write_png(target, color=(20, 20, 20))

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(source),
            "--compare-image",
            str(target),
            "--audit-missing",
            "glowing lavender eyes from the target image",
            "--audit-repair",
            "--audit-regenerate",
            "--no-video",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Regenerated image" in result.output
    assert "Regenerated visual comparison" in result.output
    audit_dirs = list(tmp_path.glob("audit-*"))
    assert len(audit_dirs) == 1
    proposed_svp = next(audit_dirs[0].glob("repair-*/target_svp.proposed.json"))
    assert FakePipeline.instances[-1].run_calls[0]["from_svp_path"] == proposed_svp
    assert list(audit_dirs[0].glob("regenerated/20260425-000000/image.png"))
    comparison = audit_dirs[0] / "regenerated_comparison" / "visual_comparison.json"
    assert comparison.exists()
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    assert payload["source_image"]["path"].endswith("source_image.png")
    assert payload["candidate_image"]["path"].endswith("candidate_image.png")


def test_openai_audit_requires_openai_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(image)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--audit-backend",
            "openai",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "OPENAI_API_KEY is required" in result.output


def test_audit_image_rejects_unreadable_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    image.write_text("not an image", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "not a readable image" in result.output


def test_audit_flags_require_audit_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)

    result = runner.invoke(
        app,
        ["prompt", "--audit-contact-graph", "left_hand -> scabbard", "--output", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "Audit options require --audit-image" in result.output


def test_compare_image_requires_audit_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    candidate = tmp_path / "candidate.png"
    candidate.write_bytes(TINY_PNG_BYTES)

    result = runner.invoke(
        app,
        ["prompt", "--compare-image", str(candidate), "--output", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "Audit options require --audit-image" in result.output


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
