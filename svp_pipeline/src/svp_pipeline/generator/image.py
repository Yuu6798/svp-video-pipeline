"""Gemini image generation layer for SVP pipeline."""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

from google import genai
from google.genai import types

from ..exceptions import ImageAPIError, ImageRefusalError
from ..schema import SVPVideo
from ..utils.prompt_render import render_image_prompt

ImageResolution = Literal["1K", "2K", "4K"]
ImageModel = Literal["gemini-3-pro-image-preview"]


@dataclass
class ImageResult:
    """Image generation result payload."""

    png_bytes: bytes
    cost_usd: float
    elapsed_sec: float
    raw_prompt: str
    model: str
    resolution: ImageResolution
    aspect_ratio: str


class ImageGenerator:
    """Generate still image PNG from SVPVideo via Gemini image API."""

    DEFAULT_MODEL: ImageModel = "gemini-3-pro-image-preview"
    DEFAULT_RESOLUTION: ImageResolution = "2K"
    ASPECT_FALLBACK: dict[str, str] = {"auto": "16:9"}
    COST_PER_IMAGE_USD: dict[ImageResolution, float] = {
        "1K": 0.04,
        "2K": 0.08,
        "4K": 0.16,
    }

    def __init__(
        self,
        model: ImageModel | str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: genai.Client | Any | None = None,
    ) -> None:
        if model != self.DEFAULT_MODEL:
            raise ValueError(f"unsupported image model: {model}")
        self.model = model

        if client is not None:
            self._client = client
            return

        resolved_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required for ImageGenerator")
        self._client = genai.Client(api_key=resolved_key)

    def generate(
        self,
        svp: SVPVideo,
        resolution: ImageResolution = DEFAULT_RESOLUTION,
    ) -> ImageResult:
        if resolution not in self.COST_PER_IMAGE_USD:
            raise ValueError(f"unsupported resolution: {resolution}")

        prompt = render_image_prompt(svp)
        aspect_ratio = self._resolve_aspect_ratio(svp.composition_layer.aspect_ratio)
        started = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=[types.Modality.IMAGE],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=resolution,
                    ),
                ),
            )
            png_bytes = self._extract_png_bytes(response)
        except ImageRefusalError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ImageAPIError("gemini image generation failed") from exc

        elapsed_sec = time.perf_counter() - started
        return ImageResult(
            png_bytes=png_bytes,
            cost_usd=self.COST_PER_IMAGE_USD[resolution],
            elapsed_sec=elapsed_sec,
            raw_prompt=prompt,
            model=self.model,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
        )

    def _resolve_aspect_ratio(self, svp_aspect_ratio: str) -> str:
        return self.ASPECT_FALLBACK.get(svp_aspect_ratio, svp_aspect_ratio)

    def _extract_png_bytes(self, response: Any) -> bytes:
        prompt_feedback = getattr(response, "prompt_feedback", None)
        prompt_block_reason = (
            getattr(prompt_feedback, "block_reason", None)
            or getattr(prompt_feedback, "blockReason", None)
            or ""
        )
        if prompt_block_reason:
            raise ImageRefusalError(f"image generation refused: {prompt_block_reason}")

        candidates = getattr(response, "candidates", None) or []
        refusal_reasons: list[str] = []
        for candidate in candidates:
            finish_reason = getattr(candidate, "finish_reason", None)
            if self._is_refusal_finish_reason(finish_reason):
                refusal_reasons.append(str(finish_reason))

            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data is None:
                    continue
                raw = getattr(inline_data, "data", None)
                if raw is None:
                    continue
                return self._coerce_image_bytes(raw)

        if refusal_reasons:
            reason_text = ", ".join(refusal_reasons)
            raise ImageRefusalError(f"image generation refused: {reason_text}")
        raise ImageAPIError("gemini response did not contain image bytes")

    @staticmethod
    def _coerce_image_bytes(raw: bytes | bytearray | str) -> bytes:
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, bytearray):
            return bytes(raw)
        if isinstance(raw, str):
            return base64.b64decode(raw)
        raise ImageAPIError(f"unsupported image payload type: {type(raw)!r}")

    @staticmethod
    def _is_refusal_finish_reason(finish_reason: Any) -> bool:
        if finish_reason is None:
            return False

        normalized = str(getattr(finish_reason, "name", finish_reason)).upper()
        refusal_reasons = {
            "SAFETY",
            "PROHIBITED_CONTENT",
            "BLOCKLIST",
            "SPII",
            "RECITATION",
            "MODEL_ARMOR",
            "IMAGE_SAFETY",
        }
        return normalized in refusal_reasons
