"""Shared fake/dummy test doubles.

Consolidates fake ``Planner`` / ``ImageGenerator`` / ``VideoGenerator`` /
``Pipeline`` / Anthropic-client doubles that were previously duplicated
across several ``tests/test_*.py`` files.

Where two source files defined a same-named fake with genuinely incompatible
behavior (different ``.calls`` recording shape, asserted directly by tests),
both variants are kept here under distinct names rather than force-merged,
to avoid changing any test's assertions. See ``FakeImageGenerator`` vs.
``FakeImageGeneratorQualityOnly`` below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from svp_pipeline.exceptions import VideoDownloadError
from svp_pipeline.generator.image import ImageResult
from svp_pipeline.generator.video import VideoResult
from svp_pipeline.pipeline import PipelineResult
from svp_pipeline.schema import SVPVideo
from tests.fixtures.helpers import load_sample, write_png
from tests.fixtures.mock_fal import TINY_MP4_BYTES
from tests.fixtures.mock_gemini import TINY_PNG_BYTES


class FakePlanner:
    """Fake planner recording ``plan()`` calls and returning a fixed SVP.

    ``model`` defaults to the same fixed string ``test_pipeline_full.py``
    used to hardcode on its ``FakePlanner`` (no test asserts this value for
    plain ``FakePlanner`` instances; the dedicated
    ``FakePlannerWithEffectiveModel`` subclass below is used whenever a test
    needs a specific effective model reported).
    """

    def __init__(self, svp: SVPVideo, model: str = "claude-haiku-4-5") -> None:
        self.svp = svp
        self.calls: list[tuple[str, int | None]] = []
        self.model = model

    def plan(self, user_prompt: str, duration: int | None = None) -> SVPVideo:
        self.calls.append((user_prompt, duration))
        return self.svp


class FakePlannerWithEffectiveModel(FakePlanner):
    def __init__(self, svp: SVPVideo, effective_model: str) -> None:
        super().__init__(svp, model=effective_model)


class FakeImageGenerator:
    """Feature-rich fake image generator (reference-image aware).

    Records each call as a ``(svp, quality_mode, reference_image_path)``
    tuple. This is the variant originally defined in ``test_pipeline.py``.
    """

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


class FakeImageGeneratorQualityOnly:
    """Simpler fake image generator recording only ``quality_mode`` strings.

    This is the variant originally defined in ``test_pipeline_full.py``,
    where assertions index ``.calls`` as plain strings (e.g.
    ``fake_image.calls[0] == "cheap"``) rather than tuples, so it cannot be
    merged with :class:`FakeImageGenerator` without changing assertions.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, svp: SVPVideo, quality_mode: str = "normal") -> ImageResult:
        self.calls.append(quality_mode)
        return ImageResult(
            png_bytes=TINY_PNG_BYTES,
            cost_usd=0.08 if quality_mode == "normal" else 0.04,
            elapsed_sec=0.11,
            raw_prompt="image prompt",
            model="gemini-3-pro-image-preview",
            backend="gemini",
            aspect_ratio=svp.composition_layer.aspect_ratio,
            native_size_or_resolution="2K" if quality_mode == "normal" else "1K",
            was_aspect_coerced=False,
        )


class FakeVideoGenerator:
    def __init__(self, tier: str = "standard", fail_download: bool = False) -> None:
        self.tier = tier
        self.fail_download = fail_download
        self.calls: list[tuple[str, str, int]] = []
        self.image_paths: list[Path] = []

    def generate(
        self,
        svp: SVPVideo,
        image_path: Path,
        output_path: Path,
        resolution: str = "720p",
    ) -> VideoResult:
        self.calls.append((self.tier, resolution, svp.motion_layer.duration_seconds))
        self.image_paths.append(image_path)
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

        svp = load_sample("shibuya_dusk.json")
        svp_path.write_text(svp.model_dump_json(indent=2), encoding="utf-8")
        if not self.kwargs.get("dry_run"):
            write_png(image_path)
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


class DummyMessages:
    def __init__(
        self,
        *,
        responses: list[Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        if not self._responses:
            raise AssertionError("no mock response left")
        return self._responses.pop(0)


class DummyClient:
    def __init__(
        self,
        *,
        responses: list[Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.messages = DummyMessages(responses=responses, error=error)
