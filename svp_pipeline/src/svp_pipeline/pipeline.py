"""Orchestrate planner -> image -> video stages."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .exceptions import VideoDownloadError
from .generator.composite import SplitCompositeImageGenerator
from .generator.image import ImageBackend, ImageBackendName, create_image_backend
from .generator.image_gemini import GeminiImageBackend
from .generator.image_openai import OpenAIImageBackend
from .generator.planner import Planner, PlannerModel
from .generator.video import VideoGenerator, VideoResolution
from .schema import SVPVideo
from .utils.logging import write_log_json

ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class PipelineResult:
    """Pipeline execution result."""

    output_dir: Path
    svp: SVPVideo
    svp_path: Path
    image_path: Path | None
    video_path: Path | None
    log_path: Path
    total_cost_usd: float
    total_elapsed_sec: float


class Pipeline:
    """planner -> image -> video orchestrator."""

    PLANNER_ESTIMATED_COST_USD = 0.012

    def __init__(
        self,
        output_dir: Path,
        planner_model: PlannerModel = "claude-opus-4-7",
        image_model: str = "gemini-3-pro-image-preview",
        image_backend: ImageBackendName | str = "gemini",
        cheap_mode: bool = False,
        dry_run: bool = False,
        character_lock: bool = True,
        planner: Planner | None = None,
        image_generator: ImageBackend | None = None,
        split_image_generator: Any | None = None,
        video_generator: VideoGenerator | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.planner_model = planner_model
        self.image_model = image_model
        self.image_backend = image_backend
        self._validate_image_backend()
        self.cheap_mode = cheap_mode
        self.dry_run = dry_run
        self.character_lock = character_lock
        self.image_quality_mode = "cheap" if cheap_mode else "normal"
        self.video_tier = "fast" if cheap_mode else "standard"
        self.video_resolution: VideoResolution = "480p" if cheap_mode else "720p"

        self._planner = (
            planner
            if planner is not None
            else Planner(model=planner_model, character_lock=character_lock)
        )
        self._image_generator = image_generator
        self._split_image_generator = split_image_generator
        self._video_generator = video_generator

    def run(
        self,
        user_prompt: str,
        duration: int | None = None,
        no_video: bool = False,
        reference_image_path: Path | None = None,
        reference_crop: int | None = None,
        separate_character_bg: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        total_started = time.perf_counter()
        run_dir = self._make_timestamp_dir()
        effective_reference_image_path = self._prepare_reference_image(
            reference_image_path=reference_image_path,
            reference_crop=reference_crop,
            run_dir=run_dir,
        )
        self._validate_separate_character_bg(
            separate_character_bg=separate_character_bg,
            effective_reference_image_path=effective_reference_image_path,
        )

        self._emit_progress(progress_callback, "planner_start", {"model": self.planner_model})
        planner_started = time.perf_counter()
        svp = self._planner.plan(user_prompt=user_prompt, duration=duration)
        planner_elapsed = time.perf_counter() - planner_started

        svp_path = run_dir / "svp.json"
        svp_path.write_text(svp.model_dump_json(indent=2), encoding="utf-8")

        planner_model_for_log = self._resolve_planner_model_for_log()
        planner_stage: dict[str, Any] = {
            "status": "ok",
            "elapsed_sec": planner_elapsed,
            "cost_usd": self.PLANNER_ESTIMATED_COST_USD,
            "model": planner_model_for_log,
        }
        self._emit_progress(
            progress_callback,
            "planner_done",
            {
                "elapsed_sec": planner_elapsed,
                "cost_usd": self.PLANNER_ESTIMATED_COST_USD,
                "model": planner_model_for_log,
            },
        )

        image_path: Path | None = None
        video_path: Path | None = None
        video_stage: dict[str, Any] | None = None
        video_generator: VideoGenerator | None = None
        total_cost = self.PLANNER_ESTIMATED_COST_USD
        if self.dry_run:
            image_estimate = self._estimate_image_stage(
                svp,
                separate_character_bg=separate_character_bg,
            )
            image_stage = {
                "status": "skipped_dry_run",
                "elapsed_sec": 0.0,
                **image_estimate,
            }
            total_cost += image_estimate["estimated_cost_usd"]
            if not no_video:
                estimated_video_cost = VideoGenerator.PRICE_PER_SECOND[
                    (self.video_tier, self.video_resolution)
                ] * svp.motion_layer.duration_seconds
                endpoint = (
                    VideoGenerator.ENDPOINT_FAST
                    if self.video_tier == "fast"
                    else VideoGenerator.ENDPOINT_STANDARD
                )
                video_stage = {
                    "status": "skipped_dry_run",
                    "elapsed_sec": 0.0,
                    "estimated_cost_usd": estimated_video_cost,
                    "endpoint": endpoint,
                    "tier": self.video_tier,
                    "resolution": self.video_resolution,
                    "aspect_ratio": svp.composition_layer.aspect_ratio,
                    "duration_seconds": svp.motion_layer.duration_seconds,
                    "fal_request_id": None,
                    "mp4_url": None,
                }
                total_cost += estimated_video_cost
        else:
            if not no_video:
                video_generator = self._video_generator
                if video_generator is None:
                    video_generator = VideoGenerator(tier=self.video_tier)
                    self._video_generator = video_generator

            if separate_character_bg:
                generator = self._resolve_split_image_generator()
            else:
                generator = self._image_generator
                if generator is None:
                    generator = create_image_backend(
                        backend=self.image_backend,
                        model=self.image_model,
                    )
                    self._image_generator = generator

            self._emit_progress(
                progress_callback,
                "image_start",
                {
                    "backend": "openai-split-composite"
                    if separate_character_bg
                    else self.image_backend,
                    "quality_mode": self.image_quality_mode,
                    "reference_image": str(effective_reference_image_path)
                    if effective_reference_image_path is not None
                    else None,
                },
            )
            if separate_character_bg:
                assert effective_reference_image_path is not None
                image_result = generator.generate(
                    svp=svp,
                    reference_image_path=effective_reference_image_path,
                    output_dir=run_dir,
                    quality_mode=self.image_quality_mode,
                )
            elif effective_reference_image_path is None:
                image_result = generator.generate(svp=svp, quality_mode=self.image_quality_mode)
            else:
                image_result = generator.generate(
                    svp=svp,
                    quality_mode=self.image_quality_mode,
                    reference_image_path=effective_reference_image_path,
                )
            image_path = run_dir / "image.png"
            image_path.write_bytes(image_result.png_bytes)
            image_stage = {
                "status": "ok",
                "elapsed_sec": image_result.elapsed_sec,
                "cost_usd": image_result.cost_usd,
                "backend": image_result.backend,
                "model": image_result.model,
                "aspect_ratio": image_result.aspect_ratio,
                "native_size_or_resolution": image_result.native_size_or_resolution,
                "was_aspect_coerced": image_result.was_aspect_coerced,
            }
            if separate_character_bg:
                image_stage.update(
                    {
                        "character_path": image_result.character_path.name,
                        "background_path": image_result.background_path.name,
                        "composite_path": image_result.composite_path.name,
                    }
                )
            total_cost += image_result.cost_usd
            self._emit_progress(
                progress_callback,
                "image_done",
                {
                    "elapsed_sec": image_result.elapsed_sec,
                    "cost_usd": image_result.cost_usd,
                    "backend": image_result.backend,
                    "model": image_result.model,
                },
            )

            if not no_video:
                assert video_generator is not None

                endpoint = (
                    VideoGenerator.ENDPOINT_FAST
                    if video_generator.tier == "fast"
                    else VideoGenerator.ENDPOINT_STANDARD
                )
                output_video_path = run_dir / "video.mp4"
                self._emit_progress(
                    progress_callback,
                    "video_start",
                    {
                        "tier": video_generator.tier,
                        "resolution": self.video_resolution,
                        "duration": svp.motion_layer.duration_seconds,
                    },
                )
                try:
                    video_result = video_generator.generate(
                        svp=svp,
                        image_path=image_path,
                        output_path=output_video_path,
                        resolution=self.video_resolution,
                    )
                except VideoDownloadError as exc:
                    video_path = None
                    estimated_video_cost = video_generator._calculate_cost(
                        tier=video_generator.tier,
                        resolution=self.video_resolution,
                        duration_seconds=svp.motion_layer.duration_seconds,
                    )
                    total_cost += estimated_video_cost
                    video_stage = {
                        "status": "download_failed",
                        "elapsed_sec": None,
                        "cost_usd": estimated_video_cost,
                        "endpoint": endpoint,
                        "tier": video_generator.tier,
                        "resolution": self.video_resolution,
                        "aspect_ratio": svp.composition_layer.aspect_ratio,
                        "duration_seconds": svp.motion_layer.duration_seconds,
                        "fal_request_id": None,
                        "mp4_url": exc.mp4_url,
                    }
                else:
                    video_path = video_result.mp4_path
                    total_cost += video_result.cost_usd
                    video_stage = {
                        "status": "ok",
                        "elapsed_sec": video_result.elapsed_sec,
                        "cost_usd": video_result.cost_usd,
                        "endpoint": endpoint,
                        "tier": video_result.tier,
                        "resolution": video_result.resolution,
                        "aspect_ratio": video_result.aspect_ratio,
                        "duration_seconds": video_result.duration_seconds,
                        "fal_request_id": video_result.fal_request_id,
                        "mp4_url": video_result.mp4_url,
                    }
                    self._emit_progress(
                        progress_callback,
                        "video_done",
                        {
                            "elapsed_sec": video_result.elapsed_sec,
                            "cost_usd": video_result.cost_usd,
                            "tier": video_result.tier,
                            "resolution": video_result.resolution,
                        },
                    )

        total_elapsed = time.perf_counter() - total_started
        stages: dict[str, Any] = {
            "planner": planner_stage,
            "image": image_stage,
        }
        if video_stage is not None:
            stages["video"] = video_stage
        log_data = {
            "user_prompt": user_prompt,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "inputs": {
                "reference_image": str(reference_image_path)
                if reference_image_path is not None
                else None,
                "reference_crop": reference_crop,
                "separate_character_bg": separate_character_bg,
                "effective_reference_image": str(effective_reference_image_path)
                if effective_reference_image_path is not None
                else None,
            },
            "stages": stages,
            "total_cost_usd": total_cost,
            "total_elapsed_sec": total_elapsed,
            "outputs": {
                "svp": svp_path.name,
                "image": image_path.name if image_path is not None else None,
                "video": video_path.name if video_path is not None else None,
            },
        }

        log_path = run_dir / "log.json"
        write_log_json(log_path, log_data)

        return PipelineResult(
            output_dir=run_dir,
            svp=svp,
            svp_path=svp_path,
            image_path=image_path,
            video_path=video_path,
            log_path=log_path,
            total_cost_usd=total_cost,
            total_elapsed_sec=total_elapsed,
        )

    def _make_timestamp_dir(self) -> Path:
        base_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = self.output_dir / base_name
        index = 1
        while candidate.exists():
            candidate = self.output_dir / f"{base_name}-{index:02d}"
            index += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def _resolve_planner_model_for_log(self) -> str:
        effective_model = getattr(self._planner, "model", None)
        if isinstance(effective_model, str) and effective_model:
            return effective_model

        requested_model = getattr(self._planner, "requested_model", None)
        if isinstance(requested_model, str) and requested_model:
            return requested_model

        return self.planner_model

    def _estimate_image_stage(
        self,
        svp: SVPVideo,
        separate_character_bg: bool = False,
    ) -> dict[str, Any]:
        aspect_ratio = svp.composition_layer.aspect_ratio
        if separate_character_bg:
            size = OpenAIImageBackend.ASPECT_TO_SIZE.get(aspect_ratio)
            if size is None:
                raise ValueError(f"unsupported aspect ratio for openai backend: {aspect_ratio!r}")
            was_coerced = aspect_ratio not in OpenAIImageBackend._DIRECT_COMPATIBLE_ASPECTS
            quality = OpenAIImageBackend.QUALITY_MAP[self.image_quality_mode]
            return {
                "estimated_cost_usd": OpenAIImageBackend.COST_PER_IMAGE_USD[
                    (size, quality)
                ]
                * 2,
                "backend": "openai-split-composite",
                "model": OpenAIImageBackend.MODEL_ID,
                "aspect_ratio": aspect_ratio,
                "native_size_or_resolution": size,
                "was_aspect_coerced": was_coerced,
            }

        if self.image_backend == "openai":
            size = OpenAIImageBackend.ASPECT_TO_SIZE.get(aspect_ratio)
            if size is None:
                raise ValueError(f"unsupported aspect ratio for openai backend: {aspect_ratio!r}")
            was_coerced = aspect_ratio not in OpenAIImageBackend._DIRECT_COMPATIBLE_ASPECTS
            quality = OpenAIImageBackend.QUALITY_MAP[self.image_quality_mode]
            return {
                "estimated_cost_usd": OpenAIImageBackend.COST_PER_IMAGE_USD[(size, quality)],
                "backend": "openai",
                "model": OpenAIImageBackend.MODEL_ID,
                "aspect_ratio": aspect_ratio,
                "native_size_or_resolution": size,
                "was_aspect_coerced": was_coerced,
            }

        resolved_aspect = GeminiImageBackend.ASPECT_FALLBACK.get(aspect_ratio, aspect_ratio)
        resolution = GeminiImageBackend.QUALITY_TO_RESOLUTION[self.image_quality_mode]
        return {
            "estimated_cost_usd": GeminiImageBackend.COST_PER_IMAGE_USD[resolution],
            "backend": "gemini",
            "model": self.image_model,
            "aspect_ratio": resolved_aspect,
            "native_size_or_resolution": resolution,
            "was_aspect_coerced": False,
        }

    def _resolve_split_image_generator(self) -> Any:
        if self._split_image_generator is not None:
            return self._split_image_generator
        if isinstance(self._image_generator, SplitCompositeImageGenerator):
            self._split_image_generator = self._image_generator
            return self._split_image_generator
        if isinstance(self._image_generator, OpenAIImageBackend):
            self._split_image_generator = SplitCompositeImageGenerator(
                client=self._image_generator.client
            )
            return self._split_image_generator

        self._split_image_generator = SplitCompositeImageGenerator()
        return self._split_image_generator

    def _validate_image_backend(self) -> None:
        if self.image_backend not in ("gemini", "openai"):
            raise ValueError(f"Unknown image backend: {self.image_backend!r}")

    def _validate_separate_character_bg(
        self,
        separate_character_bg: bool,
        effective_reference_image_path: Path | None,
    ) -> None:
        if not separate_character_bg:
            return
        if self.image_backend != "openai":
            raise ValueError("--separate-character-bg currently requires image_backend='openai'")
        if effective_reference_image_path is None:
            raise ValueError("--separate-character-bg requires a reference image")

    def _prepare_reference_image(
        self,
        reference_image_path: Path | None,
        reference_crop: int | None,
        run_dir: Path,
    ) -> Path | None:
        if reference_image_path is None:
            if reference_crop is not None:
                raise ValueError("reference_crop requires reference_image_path")
            return None

        source = Path(reference_image_path)
        if not source.exists():
            raise ValueError(f"reference image not found: {source}")
        if reference_crop is None:
            return source
        return self._crop_reference_grid(source=source, crop_index=reference_crop, run_dir=run_dir)

    @staticmethod
    def _crop_reference_grid(source: Path, crop_index: int, run_dir: Path) -> Path:
        if not 1 <= crop_index <= 9:
            raise ValueError("reference_crop must be between 1 and 9")

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
            raise ValueError("Pillow is required for --reference-crop") from exc

        try:
            with Image.open(source) as image:
                width, height = image.size
                col = (crop_index - 1) % 3
                row = (crop_index - 1) // 3
                left = col * width // 3
                upper = row * height // 3
                right = (col + 1) * width // 3 if col < 2 else width
                lower = (row + 1) * height // 3 if row < 2 else height
                cropped = image.crop((left, upper, right, lower)).convert("RGBA")
                output = run_dir / f"reference_crop_{crop_index}.png"
                cropped.save(output)
                return output
        except OSError as exc:
            raise ValueError(f"failed to crop reference image: {source}") from exc

    @staticmethod
    def _emit_progress(
        progress_callback: ProgressCallback | None,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        if progress_callback is not None:
            progress_callback(event, payload)
