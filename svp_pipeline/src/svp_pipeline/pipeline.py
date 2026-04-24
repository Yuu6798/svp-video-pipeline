"""Orchestrate planner -> image stages (M3 scope)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .generator.image import ImageBackendName, create_image_backend
from .generator.image_base import ImageBackend
from .generator.image_gemini import GeminiImageBackend
from .generator.image_openai import OpenAIImageBackend
from .generator.planner import Planner, PlannerModel
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
    """planner -> image -> (video) orchestrator.

    M3 supports planner and image only.
    """

    PLANNER_ESTIMATED_COST_USD = 0.012

    def __init__(
        self,
        output_dir: Path,
        planner_model: PlannerModel = "claude-opus-4-7",
        image_backend: ImageBackendName = "gemini",
        cheap_mode: bool = False,
        dry_run: bool = False,
        planner: Planner | None = None,
        image_generator: ImageBackend | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.planner_model = planner_model
        self.image_backend_name = image_backend
        self.cheap_mode = cheap_mode
        self.dry_run = dry_run
        self.image_quality_mode = "cheap" if cheap_mode else "normal"

        self._planner = planner if planner is not None else Planner(model=planner_model)
        self._image_generator = image_generator

    def run(
        self,
        user_prompt: str,
        duration: int | None = None,
        no_video: bool = True,
    ) -> PipelineResult:
        if not no_video:
            raise NotImplementedError("video stage is not implemented in M3")

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
        if self.dry_run:
            dryrun_meta = self._estimate_image_dryrun(svp)
            image_stage = {
                "status": "skipped_dry_run",
                "elapsed_sec": 0.0,
                "estimated_cost_usd": dryrun_meta["estimated_cost_usd"],
                "backend": dryrun_meta["backend"],
                "model": dryrun_meta["model"],
                "aspect_ratio": dryrun_meta["aspect_ratio"],
                "native_size_or_resolution": dryrun_meta["native_size_or_resolution"],
                "was_aspect_coerced": dryrun_meta["was_aspect_coerced"],
            }
            total_cost = self.PLANNER_ESTIMATED_COST_USD + dryrun_meta["estimated_cost_usd"]
        else:
            generator = self._image_generator
            if generator is None:
                generator = create_image_backend(backend=self.image_backend_name)
                self._image_generator = generator
            image_result = generator.generate(svp=svp, quality_mode=self.image_quality_mode)
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
            total_cost = self.PLANNER_ESTIMATED_COST_USD + image_result.cost_usd

        total_elapsed = time.perf_counter() - total_started
        log_data = {
            "user_prompt": user_prompt,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "stages": {
                "planner": planner_stage,
                "image": image_stage,
            },
            "total_cost_usd": total_cost,
            "total_elapsed_sec": total_elapsed,
            "outputs": {
                "svp": svp_path.name,
                "image": image_path.name if image_path is not None else None,
            },
        }

        log_path = run_dir / "log.json"
        log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")

        return PipelineResult(
            output_dir=run_dir,
            svp=svp,
            svp_path=svp_path,
            image_path=image_path,
            video_path=None,
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

    def _estimate_image_dryrun(self, svp: SVPVideo) -> dict[str, Any]:
        aspect_ratio = svp.composition_layer.aspect_ratio
        if self.image_backend_name == "gemini":
            resolution = GeminiImageBackend.QUALITY_TO_RESOLUTION[self.image_quality_mode]
            return {
                "estimated_cost_usd": GeminiImageBackend.COST_PER_IMAGE_USD[resolution],
                "backend": GeminiImageBackend.BACKEND_NAME,
                "model": GeminiImageBackend.MODEL_ID,
                "aspect_ratio": aspect_ratio,
                "native_size_or_resolution": resolution,
                "was_aspect_coerced": False,
            }

        if self.image_backend_name == "openai":
            quality = OpenAIImageBackend.QUALITY_MAP[self.image_quality_mode]
            size = OpenAIImageBackend.ASPECT_TO_SIZE[aspect_ratio]
            was_coerced = aspect_ratio not in OpenAIImageBackend._DIRECT_COMPATIBLE_ASPECTS
            return {
                "estimated_cost_usd": OpenAIImageBackend.COST_PER_IMAGE_USD[(size, quality)],
                "backend": OpenAIImageBackend.BACKEND_NAME,
                "model": OpenAIImageBackend.MODEL_ID,
                "aspect_ratio": aspect_ratio,
                "native_size_or_resolution": size,
                "was_aspect_coerced": was_coerced,
            }

        raise ValueError(f"Unknown image backend: {self.image_backend_name!r}")
