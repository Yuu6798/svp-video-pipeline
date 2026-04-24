"""Tests for image backend factory."""

from __future__ import annotations

import pytest

from svp_pipeline.generator.image import ImageGenerator, create_image_backend
from svp_pipeline.generator.image_gemini import GeminiImageBackend
from svp_pipeline.generator.image_openai import OpenAIImageBackend


def test_create_gemini_backend() -> None:
    backend = create_image_backend(backend="gemini", api_key="dummy")
    assert isinstance(backend, GeminiImageBackend)
    assert backend.BACKEND_NAME == "gemini"


def test_create_openai_backend() -> None:
    backend = create_image_backend(backend="openai", api_key="dummy")
    assert isinstance(backend, OpenAIImageBackend)
    assert backend.BACKEND_NAME == "openai"


def test_default_backend_is_gemini() -> None:
    backend = create_image_backend(api_key="dummy")
    assert isinstance(backend, GeminiImageBackend)


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown image backend"):
        create_image_backend(backend="midjourney")


def test_backward_compat_image_generator_alias() -> None:
    assert ImageGenerator is GeminiImageBackend

