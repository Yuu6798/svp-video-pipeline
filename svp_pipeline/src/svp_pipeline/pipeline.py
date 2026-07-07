"""Orchestrate planner -> image -> video stages."""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .exceptions import VideoDownloadError
from .generator.composite import SplitCompositeImageGenerator
from .generator.image import ImageBackend, ImageBackendName, create_image_backend
from .generator.image_gemini import GeminiImageBackend
from .generator.image_openai import OpenAIImageBackend
from .generator.planner import Planner, PlannerModel
from .generator.video import VideoGenerator, VideoResolution
from .schema import SVPVideo
from .semantic.failure_presets import apply_failure_presets_to_svp
from .utils.logging import write_log_json
from .utils.timestamps import utc_now_compact, utc_now_iso

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


@dataclass
class _PlannerStageResult:
    svp: SVPVideo
    stage: dict[str, Any]
    cost_usd: float
    model: str
    source_path: Path | None


@dataclass
class _ImageStageResult:
    path: Path | None
    stage: dict[str, Any]
    cost_usd: float


@dataclass
class _VideoStageResult:
    path: Path | None
    stage: dict[str, Any]
    cost_usd: float


class Pipeline:
    """planner -> image -> video orchestrator."""

    PLANNER_ESTIMATED_COST_USD = 0.012
    REUSABLE_IMAGE_ARTIFACTS: dict[str, str] = {
        "image": "image.png",
        "composite": "composite.png",
        "character_green": "character_green.png",
        "background_clean": "background_clean.png",
    }

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
        from_svp_path: Path | None = None,
        reuse_run_dir: Path | None = None,
        reuse_image: str | None = None,
        failure_presets: list[str | Path] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        total_started = time.perf_counter()
        run_dir = self._make_timestamp_dir()
        self._validate_run_inputs(
            from_svp_path=from_svp_path,
            reuse_run_dir=reuse_run_dir,
            reuse_image=reuse_image,
            reference_image_path=reference_image_path,
            reference_crop=reference_crop,
            separate_character_bg=separate_character_bg,
        )
        effective_reference_image_path = self._prepare_reference_image(
            reference_image_path=reference_image_path,
            reference_crop=reference_crop,
            run_dir=run_dir,
        )
        self._validate_separate_character_bg(
            separate_character_bg=separate_character_bg,
            effective_reference_image_path=effective_reference_image_path,
        )
        svp_source_path = self._resolve_svp_source(
            from_svp_path=from_svp_path,
            reuse_run_dir=reuse_run_dir,
        )
        planner_result = self._load_or_plan_svp(
            user_prompt=user_prompt,
            duration=duration,
            svp_source_path=svp_source_path,
            progress_callback=progress_callback,
        )

        svp = planner_result.svp
        applied_failure_presets = [str(item) for item in (failure_presets or [])]
        if failure_presets:
            svp = apply_failure_presets_to_svp(svp, list(failure_presets))

        svp_path = run_dir / "svp.json"
        svp_path.write_text(svp.model_dump_json(indent=2), encoding="utf-8")

        self._emit_progress(
            progress_callback,
            "planner_done",
            {
                "elapsed_sec": planner_result.stage["elapsed_sec"],
                "cost_usd": planner_result.cost_usd,
                "model": planner_result.model,
                "reused": planner_result.source_path is not None,
                "source": str(planner_result.source_path)
                if planner_result.source_path is not None
                else None,
            },
        )

        total_cost = planner_result.cost_usd
        reusable_image_path = self._resolve_reusable_image_path(
            reuse_run_dir=reuse_run_dir,
            reuse_image=reuse_image,
        )

        if self.dry_run:
            image_result = self._build_dry_run_image_stage(
                svp=svp,
                reusable_image_path=reusable_image_path,
                reuse_image=reuse_image,
                separate_character_bg=separate_character_bg,
            )
            video_result = self._build_dry_run_video_stage(svp=svp, no_video=no_video)
        else:
            video_generator = self._ensure_video_generator(no_video=no_video)
            image_result = self._run_image_stage(
                svp=svp,
                run_dir=run_dir,
                reusable_image_path=reusable_image_path,
                reuse_image=reuse_image,
                separate_character_bg=separate_character_bg,
                effective_reference_image_path=effective_reference_image_path,
                progress_callback=progress_callback,
            )
            video_result = self._run_video_stage(
                svp=svp,
                run_dir=run_dir,
                image_path=image_result.path,
                video_generator=video_generator,
                progress_callback=progress_callback,
            )

        total_cost += image_result.cost_usd
        if video_result is not None:
            total_cost += video_result.cost_usd

        total_elapsed = time.perf_counter() - total_started
        log_data = self._build_log_data(
            user_prompt=user_prompt,
            from_svp_path=from_svp_path,
            reuse_run_dir=reuse_run_dir,
            reuse_image=reuse_image,
            reference_image_path=reference_image_path,
            reference_crop=reference_crop,
            separate_character_bg=separate_character_bg,
            applied_failure_presets=applied_failure_presets,
            effective_reference_image_path=effective_reference_image_path,
            planner_stage=planner_result.stage,
            image_stage=image_result.stage,
            video_stage=video_result.stage if video_result is not None else None,
            total_cost=total_cost,
            total_elapsed=total_elapsed,
            svp_path=svp_path,
            image_path=image_result.path,
            video_path=video_result.path if video_result is not None else None,
        )

        log_path = run_dir / "log.json"
        write_log_json(log_path, log_data)

        return PipelineResult(
            output_dir=run_dir,
            svp=svp,
            svp_path=svp_path,
            image_path=image_result.path,
            video_path=video_result.path if video_result is not None else None,
            log_path=log_path,
            total_cost_usd=total_cost,
            total_elapsed_sec=total_elapsed,
        )

    def _validate_run_inputs(
        self,
        from_svp_path: Path | None,
        reuse_run_dir: Path | None,
        reuse_image: str | None,
        reference_image_path: Path | None,
        reference_crop: int | None,
        separate_character_bg: bool,
    ) -> None:
        self._validate_reuse_inputs(
            from_svp_path=from_svp_path,
            reuse_run_dir=reuse_run_dir,
            reuse_image=reuse_image,
            reference_image_path=reference_image_path,
            reference_crop=reference_crop,
            separate_character_bg=separate_character_bg,
        )

    def _load_or_plan_svp(
        self,
        user_prompt: str,
        duration: int | None,
        svp_source_path: Path | None,
        progress_callback: ProgressCallback | None,
    ) -> _PlannerStageResult:
        if svp_source_path is None:
            self._emit_progress(progress_callback, "planner_start", {"model": self.planner_model})
            planner_started = time.perf_counter()
            svp = self._planner.plan(user_prompt=user_prompt, duration=duration)
            planner_elapsed = time.perf_counter() - planner_started
            planner_model_for_log = self._resolve_planner_model_for_log()
            return _PlannerStageResult(
                svp=svp,
                stage={
                    "status": "ok",
                    "elapsed_sec": planner_elapsed,
                    "cost_usd": self.PLANNER_ESTIMATED_COST_USD,
                    "model": planner_model_for_log,
                },
                cost_usd=self.PLANNER_ESTIMATED_COST_USD,
                model=planner_model_for_log,
                source_path=None,
            )

        self._emit_progress(
            progress_callback,
            "planner_start",
            {"model": "existing-svp", "source": str(svp_source_path), "reused": True},
        )
        planner_started = time.perf_counter()
        svp = SVPVideo.model_validate_json(svp_source_path.read_text(encoding="utf-8"))
        planner_elapsed = time.perf_counter() - planner_started
        return _PlannerStageResult(
            svp=svp,
            stage={
                "status": "reused",
                "elapsed_sec": planner_elapsed,
                "cost_usd": 0.0,
                "model": "existing-svp",
                "source": str(svp_source_path),
            },
            cost_usd=0.0,
            model="existing-svp",
            source_path=svp_source_path,
        )

    def _build_dry_run_image_stage(
        self,
        svp: SVPVideo,
        reusable_image_path: Path | None,
        reuse_image: str | None,
        separate_character_bg: bool,
    ) -> _ImageStageResult:
        if reusable_image_path is not None:
            return _ImageStageResult(
                path=None,
                stage={
                    "status": "reused_dry_run",
                    "elapsed_sec": 0.0,
                    "estimated_cost_usd": 0.0,
                    "backend": "artifact",
                    "model": "existing-image",
                    "aspect_ratio": svp.composition_layer.aspect_ratio,
                    "native_size_or_resolution": reuse_image or "image",
                    "was_aspect_coerced": False,
                    "source": str(reusable_image_path),
                },
                cost_usd=0.0,
            )

        image_estimate = self._estimate_image_stage(
            svp,
            separate_character_bg=separate_character_bg,
        )
        return _ImageStageResult(
            path=None,
            stage={
                "status": "skipped_dry_run",
                "elapsed_sec": 0.0,
                **image_estimate,
            },
            cost_usd=image_estimate["estimated_cost_usd"],
        )

    def _build_dry_run_video_stage(
        self,
        svp: SVPVideo,
        no_video: bool,
    ) -> _VideoStageResult | None:
        if no_video:
            return None

        estimated_video_cost = (
            VideoGenerator.PRICE_PER_SECOND[(self.video_tier, self.video_resolution)]
            * svp.motion_layer.duration_seconds
        )
        endpoint = self._video_endpoint_for_tier(self.video_tier)
        return _VideoStageResult(
            path=None,
            stage={
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
            },
            cost_usd=estimated_video_cost,
        )

    def _ensure_video_generator(self, no_video: bool) -> VideoGenerator | None:
        if no_video:
            return None
        if self._video_generator is None:
            self._video_generator = VideoGenerator(tier=self.video_tier)
        return self._video_generator

    def _run_image_stage(
        self,
        svp: SVPVideo,
        run_dir: Path,
        reusable_image_path: Path | None,
        reuse_image: str | None,
        separate_character_bg: bool,
        effective_reference_image_path: Path | None,
        progress_callback: ProgressCallback | None,
    ) -> _ImageStageResult:
        if reusable_image_path is not None:
            return self._reuse_image_stage(
                svp=svp,
                run_dir=run_dir,
                reusable_image_path=reusable_image_path,
                reuse_image=reuse_image,
                progress_callback=progress_callback,
            )

        return self._generate_image_stage(
            svp=svp,
            run_dir=run_dir,
            separate_character_bg=separate_character_bg,
            effective_reference_image_path=effective_reference_image_path,
            progress_callback=progress_callback,
        )

    def _reuse_image_stage(
        self,
        svp: SVPVideo,
        run_dir: Path,
        reusable_image_path: Path,
        reuse_image: str | None,
        progress_callback: ProgressCallback | None,
    ) -> _ImageStageResult:
        self._emit_progress(
            progress_callback,
            "image_start",
            {
                "backend": "artifact-reuse",
                "quality_mode": "reused",
                "source": str(reusable_image_path),
                "reused": True,
            },
        )
        image_started = time.perf_counter()
        image_path = run_dir / "image.png"
        shutil.copy2(reusable_image_path, image_path)
        image_elapsed = time.perf_counter() - image_started
        stage = {
            "status": "reused",
            "elapsed_sec": image_elapsed,
            "cost_usd": 0.0,
            "backend": "artifact",
            "model": "existing-image",
            "aspect_ratio": svp.composition_layer.aspect_ratio,
            "native_size_or_resolution": reuse_image or "image",
            "was_aspect_coerced": False,
            "source": str(reusable_image_path),
        }
        self._emit_progress(
            progress_callback,
            "image_done",
            {
                "elapsed_sec": image_elapsed,
                "cost_usd": 0.0,
                "backend": "artifact",
                "model": "existing-image",
                "reused": True,
                "source": str(reusable_image_path),
            },
        )
        return _ImageStageResult(path=image_path, stage=stage, cost_usd=0.0)

    def _generate_image_stage(
        self,
        svp: SVPVideo,
        run_dir: Path,
        separate_character_bg: bool,
        effective_reference_image_path: Path | None,
        progress_callback: ProgressCallback | None,
    ) -> _ImageStageResult:
        generator = self._resolve_image_generator(separate_character_bg=separate_character_bg)
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
            if effective_reference_image_path is None:
                raise RuntimeError(
                    "effective_reference_image_path must be set when separate_character_bg is True"
                )
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
        stage = {
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
            stage.update(
                {
                    "character_path": image_result.character_path.name,
                    "background_path": image_result.background_path.name,
                    "composite_path": image_result.composite_path.name,
                }
            )
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
        return _ImageStageResult(
            path=image_path,
            stage=stage,
            cost_usd=image_result.cost_usd,
        )

    def _resolve_image_generator(self, separate_character_bg: bool) -> Any:
        if separate_character_bg:
            return self._resolve_split_image_generator()
        if self._image_generator is None:
            self._image_generator = create_image_backend(
                backend=self.image_backend,
                model=self.image_model,
            )
        return self._image_generator

    def _run_video_stage(
        self,
        svp: SVPVideo,
        run_dir: Path,
        image_path: Path | None,
        video_generator: VideoGenerator | None,
        progress_callback: ProgressCallback | None,
    ) -> _VideoStageResult | None:
        if video_generator is None:
            return None
        if image_path is None:
            raise RuntimeError("image_path must be set before running the video stage")

        endpoint = self._video_endpoint_for_tier(video_generator.tier)
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
            estimated_video_cost = video_generator._calculate_cost(
                tier=video_generator.tier,
                resolution=self.video_resolution,
                duration_seconds=svp.motion_layer.duration_seconds,
            )
            return _VideoStageResult(
                path=None,
                stage={
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
                },
                cost_usd=estimated_video_cost,
            )

        stage = {
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
        return _VideoStageResult(
            path=video_result.mp4_path,
            stage=stage,
            cost_usd=video_result.cost_usd,
        )

    @staticmethod
    def _video_endpoint_for_tier(tier: str) -> str:
        return VideoGenerator.ENDPOINT_FAST if tier == "fast" else VideoGenerator.ENDPOINT_STANDARD

    def _build_log_data(
        self,
        user_prompt: str,
        from_svp_path: Path | None,
        reuse_run_dir: Path | None,
        reuse_image: str | None,
        reference_image_path: Path | None,
        reference_crop: int | None,
        separate_character_bg: bool,
        applied_failure_presets: list[str],
        effective_reference_image_path: Path | None,
        planner_stage: dict[str, Any],
        image_stage: dict[str, Any],
        video_stage: dict[str, Any] | None,
        total_cost: float,
        total_elapsed: float,
        svp_path: Path,
        image_path: Path | None,
        video_path: Path | None,
    ) -> dict[str, Any]:
        stages: dict[str, Any] = {
            "planner": planner_stage,
            "image": image_stage,
        }
        if video_stage is not None:
            stages["video"] = video_stage
        return {
            "user_prompt": user_prompt,
            "timestamp": utc_now_iso(),
            "inputs": {
                "from_svp": str(from_svp_path) if from_svp_path is not None else None,
                "reuse_run": str(reuse_run_dir) if reuse_run_dir is not None else None,
                "reuse_image": reuse_image,
                "reference_image": str(reference_image_path)
                if reference_image_path is not None
                else None,
                "reference_crop": reference_crop,
                "separate_character_bg": separate_character_bg,
                "failure_presets": applied_failure_presets,
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

    def _make_timestamp_dir(self) -> Path:
        base_name = utc_now_compact()
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
                "estimated_cost_usd": OpenAIImageBackend.COST_PER_IMAGE_USD[(size, quality)] * 2,
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

    def _resolve_svp_source(
        self,
        from_svp_path: Path | None,
        reuse_run_dir: Path | None,
    ) -> Path | None:
        if from_svp_path is not None:
            return Path(from_svp_path)
        if reuse_run_dir is not None:
            return Path(reuse_run_dir) / "svp.json"
        return None

    def _resolve_reusable_image_path(
        self,
        reuse_run_dir: Path | None,
        reuse_image: str | None,
    ) -> Path | None:
        if reuse_run_dir is None:
            return None
        artifact_key = reuse_image or "image"
        artifact_name = self.REUSABLE_IMAGE_ARTIFACTS[artifact_key]
        return Path(reuse_run_dir) / artifact_name

    def _validate_reuse_inputs(
        self,
        from_svp_path: Path | None,
        reuse_run_dir: Path | None,
        reuse_image: str | None,
        reference_image_path: Path | None,
        reference_crop: int | None,
        separate_character_bg: bool,
    ) -> None:
        self._validate_from_svp_path(from_svp_path)
        self._validate_reuse_image_key(reuse_image)
        if reuse_run_dir is None:
            self._validate_no_reuse_image_without_run_dir(reuse_image)
            return

        reuse_dir = Path(reuse_run_dir)
        self._validate_reuse_run_dir(reuse_dir, reuse_run_dir)
        self._validate_reusable_svp(
            from_svp_path=from_svp_path,
            reuse_run_dir=reuse_run_dir,
        )
        self._validate_reusable_image(
            reuse_run_dir=reuse_run_dir,
            reuse_image=reuse_image,
        )
        self._validate_reuse_exclusive_options(
            reference_image_path=reference_image_path,
            reference_crop=reference_crop,
            separate_character_bg=separate_character_bg,
        )

    def _validate_from_svp_path(self, from_svp_path: Path | None) -> None:
        if from_svp_path is not None and not Path(from_svp_path).is_file():
            raise ValueError(f"SVP file not found: {from_svp_path}")

    def _validate_reuse_image_key(self, reuse_image: str | None) -> None:
        if reuse_image is not None and reuse_image not in self.REUSABLE_IMAGE_ARTIFACTS:
            allowed = ", ".join(sorted(self.REUSABLE_IMAGE_ARTIFACTS))
            raise ValueError(f"unsupported reuse image artifact: {reuse_image!r}; choose {allowed}")

    def _validate_no_reuse_image_without_run_dir(self, reuse_image: str | None) -> None:
        if reuse_image is not None:
            raise ValueError("reuse_image requires reuse_run_dir")

    def _validate_reuse_run_dir(self, reuse_dir: Path, reuse_run_dir: Path | None) -> None:
        if not reuse_dir.is_dir():
            raise ValueError(f"reuse run directory not found: {reuse_run_dir}")

    def _validate_reusable_svp(
        self,
        from_svp_path: Path | None,
        reuse_run_dir: Path | None,
    ) -> None:
        svp_source = self._resolve_svp_source(
            from_svp_path=from_svp_path,
            reuse_run_dir=reuse_run_dir,
        )
        if svp_source is None or not svp_source.is_file():
            raise ValueError(f"reusable SVP not found: {svp_source}")

    def _validate_reusable_image(
        self,
        reuse_run_dir: Path | None,
        reuse_image: str | None,
    ) -> None:
        image_source = self._resolve_reusable_image_path(
            reuse_run_dir=reuse_run_dir,
            reuse_image=reuse_image,
        )
        if image_source is None or not image_source.is_file():
            artifact_key = reuse_image or "image"
            raise ValueError(f"reusable image artifact {artifact_key!r} not found: {image_source}")

    def _validate_reuse_exclusive_options(
        self,
        reference_image_path: Path | None,
        reference_crop: int | None,
        separate_character_bg: bool,
    ) -> None:
        if reference_image_path is not None or reference_crop is not None or separate_character_bg:
            raise ValueError(
                "reuse_run_dir reuses an existing image artifact; do not combine it with "
                "reference image, reference crop, or split-composite generation"
            )
