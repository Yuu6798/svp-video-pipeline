"""Image generation entrypoint and backend factory."""

from __future__ import annotations

from typing import Literal

from .image_base import ImageBackend, ImageResult
from .image_gemini import GeminiImageBackend, GeminiResolution
from .image_openai import OpenAIImageBackend

ImageBackendName = Literal["gemini", "openai"]
ImageResolution = GeminiResolution


def create_image_backend(
    backend: ImageBackendName | str = "gemini",
    model: str | None = None,
    api_key: str | None = None,
) -> ImageBackend:
    """Create image backend instance by backend name."""
    if backend == "gemini":
        return GeminiImageBackend(model=model or GeminiImageBackend.MODEL_ID, api_key=api_key)
    if backend == "openai":
        return OpenAIImageBackend(api_key=api_key)
    raise ValueError(f"Unknown image backend: {backend!r}")


# Backward compatibility alias for existing imports.
ImageGenerator = GeminiImageBackend

__all__ = [
    "ImageBackendName",
    "ImageBackend",
    "ImageGenerator",
    "ImageResolution",
    "ImageResult",
    "create_image_backend",
]
