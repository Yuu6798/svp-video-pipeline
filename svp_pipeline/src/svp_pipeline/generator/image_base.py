"""Shared image backend interfaces and result models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ImageResult:
    """Common image generation result used by all backends."""

    png_bytes: bytes
    cost_usd: float
    elapsed_sec: float
    raw_prompt: str
    model: str
    backend: str
    aspect_ratio: str
    native_size_or_resolution: str
    was_aspect_coerced: bool = False

    @property
    def resolution(self) -> str:
        """Backward-compatible alias used by the original Gemini tests."""
        return self.native_size_or_resolution


@runtime_checkable
class ImageBackend(Protocol):
    """Protocol implemented by image backends."""

    def generate(
        self,
        svp: object,
        quality_mode: str = "normal",
    ) -> ImageResult:
        """Generate one image from SVP input."""
        ...
