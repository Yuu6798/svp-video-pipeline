"""Tests for pipeline orchestration (M3)."""

from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image

from svp_pipeline.generator.image import ImageResult
from svp_pipeline.generator.image_openai import OpenAIImageBackend
from svp_pipeline.pipeline import Pipeline
from svp_pipeline.schema import SVPVideo
from tests.fixtures.mock_gemini import TINY_PNG_BYTES
from tests.fixtures.mock_openai import build_image_response

SAMPLES_DIR = Path(__file__).parent / "samples"


def _load(name: str) -> SVPVideo:
    return SVPVideo.model_validate_json((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def _valid_png_bytes(color: str = "white") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


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
        self.calls: list[tuple[SVPVideo, str, Path | None]] = []

    def generate(
        self,
        svp: SVPVideo,
        quality_mode: str = "normal",
        reference_image_path: Path | None = None,
    ) -> ImageResult:
        self.calls.append((svp, quality_mode, reference_image_path))
        return ImageResult(
            png_bytes=TINY_PNG_BYTES,
            cost_usd=0.08 if quality_mode == "normal" else 0.04,
            elapsed_sec=0.12,
            raw_prompt="prompt",
            model="gemini-3-pro-image-preview",
            backend="gemini",
            aspect_ratio="16:9",
            native_size_or_resolution="2K" if quality_mode == "normal" else "1K",
            was_aspect_coerced=False,
        )


class FakeSplitImageGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        svp: SVPVideo,
        reference_image_path: Path,
        output_dir: Path,
        quality_mode: str = "normal",
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "svp": svp,
                "reference_image_path": reference_image_path,
                "output_dir": output_dir,
                "quality_mode": quality_mode,
            }
        )
        character_path = output_dir / "character_green.png"
        background_path = output_dir / "background_clean.png"
        composite_path = output_dir / "composite.png"
        character_path.write_bytes(TINY_PNG_BYTES)
        background_path.write_bytes(TINY_PNG_BYTES)
        composite_path.write_bytes(TINY_PNG_BYTES)
        return SimpleNamespace(
            png_bytes=TINY_PNG_BYTES,
            cost_usd=0.032,
            elapsed_sec=0.25,
            raw_prompt="split prompt",
            model="gpt-image-2",
            backend="openai-split-composite",
            aspect_ratio=svp.composition_layer.aspect_ratio,
            native_size_or_resolution="1536x1024",
            was_aspect_coerced=False,
            character_path=character_path,
            background_path=background_path,
            composite_path=composite_path,
        )


class DummyOpenAIClient:
    def __init__(self) -> None:
        png_bytes = _valid_png_bytes()
        self.images = MagicMock()
        self.images.edit.return_value = build_image_response(png_bytes)
        self.images.generate.return_value = build_image_response(png_bytes)


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
    assert log_data["stages"]["image"]["backend"] == "gemini"
    assert log_data["stages"]["image"]["native_size_or_resolution"] == "2K"
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
    assert log_data["stages"]["image"]["backend"] == "gemini"
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
    assert image.calls[0][1] == "cheap"


def test_pipeline_passes_reference_image_to_image_backend(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(TINY_PNG_BYTES)
    planner = FakePlanner(svp)
    image = FakeImageGenerator()
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner=planner,  # type: ignore[arg-type]
        image_generator=image,  # type: ignore[arg-type]
    )

    result = pipeline.run("reference prompt", no_video=True, reference_image_path=reference_image)
    log_data = json.loads(result.log_path.read_text(encoding="utf-8"))

    assert len(image.calls) == 1
    assert image.calls[0][2] == reference_image
    assert log_data["inputs"]["reference_image"] == str(reference_image)


def test_pipeline_crops_reference_grid_before_image_backend(tmp_path: Path) -> None:
    from PIL import Image

    svp = _load("shibuya_dusk.json")
    reference_image = tmp_path / "reference_grid.png"
    Image.new("RGB", (300, 300), "white").save(reference_image)
    planner = FakePlanner(svp)
    image = FakeImageGenerator()
    pipeline = Pipeline(
        output_dir=tmp_path,
        planner=planner,  # type: ignore[arg-type]
        image_generator=image,  # type: ignore[arg-type]
    )

    result = pipeline.run(
        "reference crop prompt",
        no_video=True,
        reference_image_path=reference_image,
        reference_crop=5,
    )
    log_data = json.loads(result.log_path.read_text(encoding="utf-8"))
    effective_reference = Path(log_data["inputs"]["effective_reference_image"])

    assert len(image.calls) == 1
    assert image.calls[0][2] == effective_reference
    assert effective_reference.name == "reference_crop_5.png"
    assert effective_reference.exists()
    assert log_data["inputs"]["reference_crop"] == 5


def test_reference_crop_invalid_image_raises_value_error(tmp_path: Path) -> None:
    reference_image = tmp_path / "not_an_image.txt"
    reference_image.write_text("not an image", encoding="utf-8")

    try:
        Pipeline._crop_reference_grid(source=reference_image, crop_index=1, run_dir=tmp_path)
    except ValueError as exc:
        assert "failed to crop reference image" in str(exc)
    else:
        raise AssertionError("expected invalid reference image crop to fail")


def test_pipeline_openai_backend_records_backend_fields(tmp_path: Path) -> None:
    svp = _load("action_ninja.json")
    planner = FakePlanner(svp)
    pipeline = Pipeline(
        output_dir=tmp_path,
        image_backend="openai",
        dry_run=True,
        planner=planner,  # type: ignore[arg-type]
    )

    result = pipeline.run("openai dry run prompt", no_video=True)
    log_data = json.loads(result.log_path.read_text(encoding="utf-8"))

    assert log_data["stages"]["image"]["backend"] == "openai"
    assert log_data["stages"]["image"]["model"] == "gpt-image-2"
    assert log_data["stages"]["image"]["native_size_or_resolution"] == "1536x1024"
    assert log_data["stages"]["image"]["was_aspect_coerced"] is True


def test_dry_run_split_composite_estimates_double_openai_cost(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(TINY_PNG_BYTES)
    planner = FakePlanner(svp)
    pipeline = Pipeline(
        output_dir=tmp_path,
        image_backend="openai",
        cheap_mode=True,
        dry_run=True,
        planner=planner,  # type: ignore[arg-type]
    )

    result = pipeline.run(
        "split dry run prompt",
        no_video=True,
        reference_image_path=reference_image,
        separate_character_bg=True,
    )
    log_data = json.loads(result.log_path.read_text(encoding="utf-8"))

    assert log_data["inputs"]["separate_character_bg"] is True
    assert log_data["stages"]["image"]["backend"] == "openai-split-composite"
    assert log_data["stages"]["image"]["estimated_cost_usd"] == 0.032


def test_dry_run_split_composite_requires_openai_backend(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(TINY_PNG_BYTES)
    planner = FakePlanner(svp)
    pipeline = Pipeline(
        output_dir=tmp_path,
        image_backend="gemini",
        dry_run=True,
        planner=planner,  # type: ignore[arg-type]
    )

    try:
        pipeline.run(
            "split dry run prompt",
            no_video=True,
            reference_image_path=reference_image,
            separate_character_bg=True,
        )
    except ValueError as exc:
        assert "requires image_backend='openai'" in str(exc)
    else:
        raise AssertionError("expected split composite with gemini backend to fail")


def test_split_composite_uses_injected_split_generator(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(TINY_PNG_BYTES)
    planner = FakePlanner(svp)
    split_generator = FakeSplitImageGenerator()
    pipeline = Pipeline(
        output_dir=tmp_path,
        image_backend="openai",
        planner=planner,  # type: ignore[arg-type]
        split_image_generator=split_generator,
    )

    result = pipeline.run(
        "split injected prompt",
        no_video=True,
        reference_image_path=reference_image,
        separate_character_bg=True,
    )
    log_data = json.loads(result.log_path.read_text(encoding="utf-8"))

    assert len(split_generator.calls) == 1
    assert split_generator.calls[0]["reference_image_path"] == reference_image
    assert split_generator.calls[0]["quality_mode"] == "normal"
    assert log_data["stages"]["image"]["backend"] == "openai-split-composite"


def test_split_composite_reuses_injected_openai_backend_client(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(TINY_PNG_BYTES)
    planner = FakePlanner(svp)
    client = DummyOpenAIClient()
    openai_backend = OpenAIImageBackend(client=client)
    pipeline = Pipeline(
        output_dir=tmp_path,
        image_backend="openai",
        cheap_mode=True,
        planner=planner,  # type: ignore[arg-type]
        image_generator=openai_backend,
    )

    result = pipeline.run(
        "split reuses client prompt",
        no_video=True,
        reference_image_path=reference_image,
        separate_character_bg=True,
    )

    client.images.edit.assert_called_once()
    client.images.generate.assert_called_once()
    assert result.image_path is not None
    assert result.image_path.exists()


def test_pipeline_rejects_unknown_image_backend(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    planner = FakePlanner(svp)

    try:
        Pipeline(
            output_dir=tmp_path,
            image_backend="gemnii",
            dry_run=True,
            planner=planner,  # type: ignore[arg-type]
        )
    except ValueError as exc:
        assert "Unknown image backend" in str(exc)
    else:
        raise AssertionError("expected invalid image backend to fail")
