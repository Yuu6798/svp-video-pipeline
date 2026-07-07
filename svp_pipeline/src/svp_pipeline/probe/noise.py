"""A/A noise-floor measurement: N regenerations from one fixed SVP.

``run_noise_probe`` skips the planner entirely (SVP JSON is read directly)
and calls the existing image backend factory N times, then measures every
pairwise pixel distance with ``semantic.visual_compare.pixel_metrics``. The
resulting :class:`NoiseFloorReport` is a measurement instrument: it reports
mean/min/max distances and cost, and deliberately carries no verdict field,
threshold, or pass/fail judgement (see ``docs/measurement.md``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from ..generator.image import ImageBackend, ImageBackendName, create_image_backend
from ..schema import SVPVideo
from ..semantic.visual_compare import pixel_metrics
from ..utils.timestamps import utc_now_iso

SCHEMA_VERSION = "SVP.noise_floor.v1"
REPORT_FILENAME = "noise_floor.json"


@dataclass(frozen=True)
class NoisePairMetric:
    """Pixel distance between one pair of probe images."""

    index_a: int
    index_b: int
    pixel_rms: float
    mean_rgb_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_a": self.index_a,
            "index_b": self.index_b,
            "pixel_rms": self.pixel_rms,
            "mean_rgb_delta": self.mean_rgb_delta,
        }


@dataclass(frozen=True)
class NoiseFloorReport:
    """A/A noise-floor measurement for one SVP under one backend.

    Instrument-only: no verdict/threshold/pass-fail field by design (Design
    Memo PROBE-0 AC3). Consumers (repair-effect checks, backend comparisons,
    future grip measurements) interpret the numbers themselves.
    """

    svp_path: str
    svp_sha256: str
    backend: str
    model: str
    n: int
    timestamp_utc: str
    per_image_cost_usd: list[float]
    total_cost_usd: float
    pairs: list[NoisePairMetric]
    pixel_rms_mean: float
    pixel_rms_min: float
    pixel_rms_max: float
    mean_rgb_delta_mean: float
    mean_rgb_delta_min: float
    mean_rgb_delta_max: float
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "svp_path": self.svp_path,
            "svp_sha256": self.svp_sha256,
            "backend": self.backend,
            "model": self.model,
            "n": self.n,
            "timestamp_utc": self.timestamp_utc,
            "per_image_cost_usd": self.per_image_cost_usd,
            "total_cost_usd": self.total_cost_usd,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "pixel_rms_mean": self.pixel_rms_mean,
            "pixel_rms_min": self.pixel_rms_min,
            "pixel_rms_max": self.pixel_rms_max,
            "mean_rgb_delta_mean": self.mean_rgb_delta_mean,
            "mean_rgb_delta_min": self.mean_rgb_delta_min,
            "mean_rgb_delta_max": self.mean_rgb_delta_max,
        }


def run_noise_probe(
    *,
    svp_path: Path,
    backend: ImageBackendName | str,
    n: int,
    output_dir: Path,
    image_backend: ImageBackend | None = None,
    preloaded_svp: SVPVideo | None = None,
) -> NoiseFloorReport:
    """Generate ``n`` images from ``svp_path`` and measure all-pairs noise.

    ``image_backend`` allows dependency injection of a pre-built backend
    (used by the corpus loop to avoid re-authenticating per SVP, and by
    tests to inject a fake backend without touching real APIs).

    ``preloaded_svp`` allows a caller that already parsed and schema-validated
    the SVP (e.g. the corpus preflight step) to pass the result in directly,
    avoiding a second JSON parse of the same file.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2 to compute a pairwise noise floor, got {n}")

    svp_path = Path(svp_path)
    if not svp_path.is_file():
        raise ValueError(f"SVP file not found: {svp_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    svp = (
        preloaded_svp
        if preloaded_svp is not None
        else SVPVideo.model_validate_json(svp_path.read_text(encoding="utf-8"))
    )
    backend_instance = (
        image_backend if image_backend is not None else create_image_backend(backend=backend)
    )

    image_paths: list[Path] = []
    per_image_cost: list[float] = []
    model_name = ""
    for i in range(n):
        result = backend_instance.generate(svp)
        image_path = output_dir / f"probe_{i:02d}.png"
        image_path.write_bytes(result.png_bytes)
        image_paths.append(image_path)
        per_image_cost.append(round(result.cost_usd, 4))
        model_name = result.model

    pairs: list[NoisePairMetric] = []
    for i, j in combinations(range(n), 2):
        metrics = pixel_metrics(image_paths[i], image_paths[j])
        pairs.append(
            NoisePairMetric(
                index_a=i,
                index_b=j,
                pixel_rms=metrics["pixel_rms"],
                mean_rgb_delta=metrics["mean_rgb_delta"],
            )
        )

    rms_values = [pair.pixel_rms for pair in pairs]
    delta_values = [pair.mean_rgb_delta for pair in pairs]

    return NoiseFloorReport(
        svp_path=str(svp_path),
        svp_sha256=_sha256(svp_path),
        backend=str(backend),
        model=model_name,
        n=n,
        timestamp_utc=utc_now_iso(),
        per_image_cost_usd=per_image_cost,
        total_cost_usd=round(sum(per_image_cost), 4),
        pairs=pairs,
        pixel_rms_mean=round(sum(rms_values) / len(rms_values), 4),
        pixel_rms_min=round(min(rms_values), 4),
        pixel_rms_max=round(max(rms_values), 4),
        mean_rgb_delta_mean=round(sum(delta_values) / len(delta_values), 4),
        mean_rgb_delta_min=round(min(delta_values), 4),
        mean_rgb_delta_max=round(max(delta_values), 4),
    )


def write_noise_floor_report(report: NoiseFloorReport, output_dir: Path) -> Path:
    """Write ``report`` as ``noise_floor.json`` inside ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
