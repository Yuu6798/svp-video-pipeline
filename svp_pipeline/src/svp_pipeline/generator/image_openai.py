"""OpenAI gpt-image-2 backend for SVP pipeline."""

from __future__ import annotations

import base64
import os
import time
import warnings
from typing import Any, Literal

try:
    from openai import OpenAI, OpenAIError
except ImportError:  # pragma: no cover - covered by dependency install in CI
    OpenAI = Any  # type: ignore[assignment,misc]

    class OpenAIError(Exception):
        """Fallback OpenAIError when openai package is unavailable."""


from ..exceptions import ImageAPIError, ImageRefusalError
from ..schema import SVPVideo
from ..utils.prompt_render import render_image_prompt
from .image_base import ImageResult

OpenAIQuality = Literal["low", "medium", "high", "auto"]
OpenAISize = Literal["1024x1024", "1536x1024", "1024x1536", "auto"]


class OpenAIImageBackend:
    """Generate images using OpenAI gpt-image-2."""

    MODEL_ID = "gpt-image-2"
    BACKEND_NAME = "openai"

    ASPECT_TO_SIZE: dict[str, OpenAISize] = {
        "1:1": "1024x1024",
        "16:9": "1536x1024",
        "4:3": "1536x1024",
        "21:9": "1536x1024",
        "9:16": "1024x1536",
        "3:4": "1024x1536",
        "auto": "auto",
    }

    QUALITY_MAP: dict[str, OpenAIQuality] = {
        "normal": "high",
        "cheap": "low",
    }

    COST_PER_IMAGE_USD: dict[tuple[OpenAISize, OpenAIQuality], float] = {
        ("1024x1024", "high"): 0.17,
        ("1024x1024", "medium"): 0.04,
        ("1024x1024", "low"): 0.011,
        ("1536x1024", "high"): 0.25,
        ("1536x1024", "medium"): 0.06,
        ("1536x1024", "low"): 0.016,
        ("1024x1536", "high"): 0.25,
        ("1024x1536", "medium"): 0.06,
        ("1024x1536", "low"): 0.016,
        ("auto", "high"): 0.25,
        ("auto", "medium"): 0.06,
        ("auto", "low"): 0.016,
        ("auto", "auto"): 0.06,
    }

    _DIRECT_COMPATIBLE_ASPECTS = {"1:1", "16:9", "9:16", "auto"}

    def __init__(
        self,
        api_key: str | None = None,
        client: OpenAI | Any | None = None,
    ) -> None:
        self.model = self.MODEL_ID
        if client is not None:
            self._client = client
            return

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIImageBackend")
        self._client = OpenAI(api_key=resolved_key)

    def generate(
        self,
        svp: SVPVideo,
        quality_mode: str = "normal",
    ) -> ImageResult:
        if quality_mode not in self.QUALITY_MAP:
            raise ValueError(f"unsupported quality mode: {quality_mode}")

        prompt = render_image_prompt(svp)
        size, was_coerced = self._resolve_size(svp.composition_layer.aspect_ratio)
        quality = self.QUALITY_MAP[quality_mode]
        if was_coerced:
            warnings.warn(
                (
                    "OpenAI gpt-image-2 coerced aspect ratio "
                    f"{svp.composition_layer.aspect_ratio!r} -> {size!r}"
                ),
                UserWarning,
                stacklevel=2,
            )

        started = time.perf_counter()
        try:
            response = self._client.images.generate(
                model=self.model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1,
                output_format="png",
            )
        except OpenAIError as exc:
            if self._is_content_policy_error(exc):
                raise ImageRefusalError(
                    "openai image generation refused by content policy"
                ) from exc
            if getattr(exc, "status_code", None) == 401:
                raise ImageAPIError("openai image generation failed: unauthorized (401)") from exc
            if getattr(exc, "status_code", None) == 429:
                raise ImageAPIError("openai image generation failed: rate limited (429)") from exc
            raise ImageAPIError("openai image generation failed") from exc
        except Exception as exc:  # noqa: BLE001
            raise ImageAPIError("openai image generation failed") from exc

        png_bytes = self._extract_png_bytes(response)
        elapsed_sec = time.perf_counter() - started
        cost = self.COST_PER_IMAGE_USD[(size, quality)]
        return ImageResult(
            png_bytes=png_bytes,
            cost_usd=cost,
            elapsed_sec=elapsed_sec,
            raw_prompt=prompt,
            model=self.model,
            backend=self.BACKEND_NAME,
            aspect_ratio=svp.composition_layer.aspect_ratio,
            native_size_or_resolution=size,
            was_aspect_coerced=was_coerced,
        )

    def _resolve_size(self, svp_aspect_ratio: str) -> tuple[OpenAISize, bool]:
        size = self.ASPECT_TO_SIZE.get(svp_aspect_ratio)
        if size is None:
            raise ValueError(f"unsupported aspect ratio for openai backend: {svp_aspect_ratio!r}")
        was_coerced = svp_aspect_ratio not in self._DIRECT_COMPATIBLE_ASPECTS
        return size, was_coerced

    @staticmethod
    def _extract_png_bytes(response: Any) -> bytes:
        data_items = getattr(response, "data", None) or []
        if not data_items:
            raise ImageRefusalError("openai image generation returned empty data")

        first = data_items[0]
        b64_json = getattr(first, "b64_json", None)
        if not b64_json:
            raise ImageRefusalError("openai image generation returned no image payload")
        try:
            return base64.b64decode(b64_json)
        except Exception as exc:  # noqa: BLE001
            raise ImageAPIError("openai image payload decode failed") from exc

    @staticmethod
    def _is_content_policy_error(exc: OpenAIError) -> bool:
        status_code = getattr(exc, "status_code", None)
        code = getattr(exc, "code", None)

        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error_obj = body.get("error")
            if isinstance(error_obj, dict):
                code = code or error_obj.get("code")

        code_text = str(code or "").lower()
        message_text = str(exc).lower()
        if "content_policy" in code_text or "policy_violation" in code_text:
            return True
        if "content policy" in message_text or "safety" in message_text:
            return status_code == 400
        return False
