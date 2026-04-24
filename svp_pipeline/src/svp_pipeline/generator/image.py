"""Image generation entrypoint and backend factory."""

from __future__ import annotations

from typing import Literal

from .image_base import ImageBackend, ImageResult
from .image_gemini import GeminiImageBackend
from .image_openai import OpenAIImageBackend

ImageBackendName = Literal["gemini", "openai"]


def create_image_backend(
    backend: ImageBackendName | str = "gemini",
    api_key: str | None = None,
) -> ImageBackend:
    """Create image backend instance by backend name."""
    if backend == "gemini":
        return GeminiImageBackend(api_key=api_key)
    if backend == "openai":
        return OpenAIImageBackend(api_key=api_key)
    raise ValueError(f"Unknown image backend: {backend!r}")


# Backward compatibility alias for existing imports.
ImageGenerator = GeminiImageBackend

__all__ = [
    "ImageBackendName",
    "ImageBackend",
    "ImageResult",
    "create_image_backend",
    "ImageGenerator",
]

