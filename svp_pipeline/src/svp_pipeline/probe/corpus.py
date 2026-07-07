"""Corpus manifest loading for batch A/A noise-floor probing.

Manifest format is JSON (not YAML): ``svp_pipeline``'s ``pyproject.toml``
does not depend on PyYAML anywhere, and CLAUDE.md forbids adding new
dependencies for this feature, so ``--corpus`` takes a small JSON file
instead of the YAML manifest sketched in the original design memo draft.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..generator.image import ImageBackend, create_image_backend
from .noise import NoiseFloorReport, run_noise_probe, write_noise_floor_report

_REQUIRED_KEYS = {"backend", "n", "svp_files"}


@dataclass(frozen=True)
class NoiseCorpusManifest:
    """Parsed and validated ``--corpus`` manifest."""

    backend: str
    n: int
    svp_files: list[Path]


def load_corpus_manifest(path: Path) -> NoiseCorpusManifest:
    """Load and validate a corpus manifest JSON file. Fails fast on drift.

    Unknown top-level keys, a missing required key, or an out-of-range
    ``n`` all raise ``ValueError`` immediately rather than being silently
    ignored or deferred to generation time.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"corpus manifest not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"corpus manifest must be a JSON object: {path}")

    unknown = set(raw) - _REQUIRED_KEYS
    if unknown:
        raise ValueError(f"unknown corpus manifest key(s): {sorted(unknown)}")
    missing = _REQUIRED_KEYS - set(raw)
    if missing:
        raise ValueError(f"corpus manifest missing required key(s): {sorted(missing)}")

    backend = raw["backend"]
    if not isinstance(backend, str) or not backend:
        raise ValueError("corpus manifest 'backend' must be a non-empty string")

    n = raw["n"]
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError(f"corpus manifest 'n' must be an integer >= 2, got {n!r}")

    svp_files_raw = raw["svp_files"]
    if not isinstance(svp_files_raw, list) or not svp_files_raw:
        raise ValueError("corpus manifest 'svp_files' must be a non-empty list")

    base_dir = path.parent
    svp_files: list[Path] = []
    for item in svp_files_raw:
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"corpus manifest 'svp_files' entries must be non-empty strings: {item!r}"
            )
        item_path = Path(item)
        svp_files.append(item_path if item_path.is_absolute() else base_dir / item_path)

    return NoiseCorpusManifest(backend=backend, n=n, svp_files=svp_files)


def run_corpus_noise_probe(
    *,
    manifest: NoiseCorpusManifest,
    output_dir: Path,
    image_backend: ImageBackend | None = None,
) -> list[NoiseFloorReport]:
    """Run ``run_noise_probe`` for every SVP file in ``manifest``.

    Each SVP gets its own sub-directory (named after the SVP file stem)
    under ``output_dir``, with its own ``noise_floor.json``. All SVP files
    share the manifest's single ``backend``/``n`` configuration, and (for
    real backends) a single backend instance so credentials are resolved
    once for the whole corpus run.
    """
    output_dir = Path(output_dir)
    backend_instance = (
        image_backend
        if image_backend is not None
        else create_image_backend(backend=manifest.backend)
    )

    reports: list[NoiseFloorReport] = []
    for svp_file in manifest.svp_files:
        sub_dir = output_dir / svp_file.stem
        report = run_noise_probe(
            svp_path=svp_file,
            backend=manifest.backend,
            n=manifest.n,
            output_dir=sub_dir,
            image_backend=backend_instance,
        )
        write_noise_floor_report(report, sub_dir)
        reports.append(report)
    return reports
