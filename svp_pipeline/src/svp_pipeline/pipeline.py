"""Orchestrate planner -> image -> video stages."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .exceptions import VideoDownloadError
from .generator.image import ImageGenerator, ImageResolution
from .generator.planner import Planner, PlannerModel
from .generator.video import VideoGenerator, VideoResolution
from .schema import SVPVideo


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
        cheap_mode: bool = False,
        dry_run: bool = False,
        planner: Planner | None = None,
        image_generator: ImageGenerator | None = None,
        video_generator: VideoGenerator | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.planner_model = planner_model
        self.image_model = image_model
        self.cheap_mode = cheap_mode
        self.dry_run = dry_run
        self.image_resolution: ImageResolution = "1K" if cheap_mode else "2K"
        self.video_tier = "fast" if cheap_mode else "standard"
        self.video_resolution: VideoResolution = "480p" if cheap_mode else "720p"

        self._planner = planner if planner is not None else Planner(model=planner_model)
        self._image_generator = image_generator
        self._video_generator = video_generator

    def run(
        self,
        user_prompt: str,
        duration: int | None = None,
        no_video: bool = False,
    ) -> PipelineResult:
        total_started = time.perf_counter()
        run_dir = self._make_timestamp_dir()

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

        image_path: Path | None = None
        video_path: Path | None = None
        video_stage: dict[str, Any] | None = None
        total_cost = self.PLANNER_ESTIMATED_COST_USD
        if self.dry_run:
            estimated_image_cost = ImageGenerator.COST_PER_IMAGE_USD[self.image_resolution]
            resolved_aspect = ImageGenerator.ASPECT_FALLBACK.get(
                svp.composition_layer.aspect_ratio,
                svp.composition_layer.aspect_ratio,
            )
            image_stage = {
                "status": "skipped_dry_run",
                "elapsed_sec": 0.0,
                "estimated_cost_usd": estimated_image_cost,
                "model": self.image_model,
                "resolution": self.image_resolution,
                "aspect_ratio": resolved_aspect,
            }
            total_cost += estimated_image_cost
        else:
            generator = self._image_generator
            if generator is None:
                generator = ImageGenerator(model=self.image_model)
                self._image_generator = generator
            image_result = generator.generate(svp=svp, resolution=self.image_resolution)
            image_path = run_dir / "image.png"
            image_path.write_bytes(image_result.png_bytes)
            image_stage = {
                "status": "ok",
                "elapsed_sec": image_result.elapsed_sec,
                "cost_usd": image_result.cost_usd,
                "model": image_result.model,
                "resolution": image_result.resolution,
                "aspect_ratio": image_result.aspect_ratio,
            }
            total_cost += image_result.cost_usd

            if not no_video:
                video_generator = self._video_generator
                if video_generator is None:
                    video_generator = VideoGenerator(tier=self.video_tier)
                    self._video_generator = video_generator

                endpoint = (
                    VideoGenerator.ENDPOINT_FAST
                    if video_generator.tier == "fast"
                    else VideoGenerator.ENDPOINT_STANDARD
                )
                output_video_path = run_dir / "video.mp4"
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
        log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")

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
