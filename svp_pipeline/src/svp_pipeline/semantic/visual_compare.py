"""Visual artifact comparison helpers for audit packets."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat


@dataclass
class VisualComparisonResult:
    """Files written by a visual comparison pass."""

    output_dir: Path
    source_image_path: Path
    candidate_image_path: Path
    comparison_path: Path
    metrics: dict[str, Any]


def write_visual_comparison(
    *,
    source_image_path: Path,
    candidate_image_path: Path,
    output_dir: Path,
    notes: list[str] | None = None,
) -> VisualComparisonResult:
    """Copy two image artifacts into an audit packet and write comparison metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_copy = _copy_named_image(Path(source_image_path), output_dir, "source_image")
    candidate_copy = _copy_named_image(Path(candidate_image_path), output_dir, "candidate_image")
    metrics = _build_metrics(
        source_image_path=source_copy,
        candidate_image_path=candidate_copy,
        notes=notes or [],
    )
    comparison_path = output_dir / "visual_comparison.json"
    comparison_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return VisualComparisonResult(
        output_dir=output_dir,
        source_image_path=source_copy,
        candidate_image_path=candidate_copy,
        comparison_path=comparison_path,
        metrics=metrics,
    )


def copy_source_image_to_packet(image_path: Path, output_dir: Path) -> Path:
    """Copy the audited source image into the packet under a stable name."""
    return _copy_named_image(Path(image_path), Path(output_dir), "source_image")


def pixel_metrics(image_a: Path, image_b: Path) -> dict[str, float]:
    """Compute pixel RMS / mean-RGB-delta between two same-dimension images.

    Public wrapper around the module-private ``_pixel_metrics`` helper so
    other modules (e.g. ``probe/noise.py``'s A/A noise-floor harness) can
    reuse the same pixel comparison logic without duplicating it. Callers
    are responsible for ensuring both images have equal dimensions
    (``_build_metrics`` above skips the pixel comparison when they differ;
    this wrapper does not re-check that).
    """
    return _pixel_metrics(Path(image_a), Path(image_b))


def validate_image_file(path: Path) -> None:
    """Validate that a file is readable as an image."""
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()


def _build_metrics(
    *,
    source_image_path: Path,
    candidate_image_path: Path,
    notes: list[str],
) -> dict[str, Any]:
    source_info = _image_info(source_image_path)
    candidate_info = _image_info(candidate_image_path)
    metrics: dict[str, Any] = {
        "schema_version": "SVP.visual_comparison.v1",
        "source_image": source_info,
        "candidate_image": candidate_info,
        "dimensions_equal": source_info["size"] == candidate_info["size"],
        "notes": notes,
    }
    if metrics["dimensions_equal"]:
        metrics.update(_pixel_metrics(source_image_path, candidate_image_path))
    else:
        metrics["pixel_rms"] = None
        metrics["mean_rgb_delta"] = None
        metrics["comparison_note"] = "pixel metrics skipped because dimensions differ"
    return metrics


def _image_info(path: Path) -> dict[str, Any]:
    validate_image_file(path)
    with Image.open(path) as image:
        return {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "format": image.format,
            "mode": image.mode,
            "size": list(image.size),
        }


def _pixel_metrics(source_path: Path, candidate_path: Path) -> dict[str, float]:
    with Image.open(source_path) as source, Image.open(candidate_path) as candidate:
        source_rgb = source.convert("RGB")
        candidate_rgb = candidate.convert("RGB")
        diff = ImageChops.difference(source_rgb, candidate_rgb)
        rms = ImageStat.Stat(diff).rms
        source_mean = ImageStat.Stat(source_rgb).mean
        candidate_mean = ImageStat.Stat(candidate_rgb).mean
        mean_delta = [
            abs(candidate_value - source_value)
            for source_value, candidate_value in zip(source_mean, candidate_mean, strict=True)
        ]
        return {
            "pixel_rms": round(sum(rms) / len(rms), 4),
            "mean_rgb_delta": round(sum(mean_delta) / len(mean_delta), 4),
        }


def _copy_named_image(path: Path, output_dir: Path, stem: str) -> Path:
    suffix = path.suffix.lower() or ".png"
    destination = output_dir / f"{stem}{suffix}"
    if path.resolve() != destination.resolve():
        shutil.copy2(path, destination)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
