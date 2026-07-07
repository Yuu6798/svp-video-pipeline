"""Shared, side-effect-free test helpers.

Consolidates the ``_load`` / ``_write_png`` / ``_valid_png_bytes``-style helpers
that were previously duplicated (with identical or near-identical bodies)
across many ``tests/test_*.py`` files.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

from svp_pipeline.schema import SVPVideo

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def load_sample(name: str) -> SVPVideo:
    """Load a sample ``SVPVideo`` fixture by filename from ``tests/samples``."""
    return SVPVideo.model_validate_json((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def write_png(path: Path, color: tuple[int, int, int] = (0, 0, 0)) -> None:
    """Write a tiny 2x2 PNG file to ``path``."""
    Image.new("RGB", (2, 2), color=color).save(path)


def valid_png_bytes(color: str = "white") -> bytes:
    """Return the raw bytes of a tiny 4x4 in-memory PNG."""
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()
