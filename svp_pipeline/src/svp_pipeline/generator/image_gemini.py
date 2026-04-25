"""Gemini image generation backend for SVP pipeline."""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types

from ..exceptions import ImageAPIError, ImageRefusalError
from ..schema import SVPVideo
from ..utils.prompt_render import append_reference_usage_policy, render_image_prompt
from .image_base import ImageResult

GeminiResolution = Literal["1K", "2K", "4K"]


class GeminiImageBackend:
    """Generate still image bytes from SVPVideo via Gemini image API."""

    MODEL_ID = "gemini-3-pro-image-preview"
    BACKEND_NAME = "gemini"
    ASPECT_FALLBACK: dict[str, str] = {"auto": "16:9"}
    COST_PER_IMAGE_USD: dict[GeminiResolution, float] = {
        "1K": 0.04,
        "2K": 0.08,
        "4K": 0.16,
    }
    QUALITY_TO_RESOLUTION: dict[str, GeminiResolution] = {
        "normal": "2K",
        "cheap": "1K",
    }

    def __init__(
        self,
        model: str = MODEL_ID,
        api_key: str | None = None,
        client: genai.Client | Any | None = None,
    ) -> None:
        if model != self.MODEL_ID:
            raise ValueError(f"unsupported gemini image model: {model}")
        self.model = model
        if client is not None:
            self._client = client
            return

        resolved_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required for GeminiImageBackend")
        self._client = genai.Client(api_key=resolved_key)

    def generate(
        self,
        svp: SVPVideo,
        quality_mode: str = "normal",
        resolution: GeminiResolution | None = None,
        reference_image_path: Path | None = None,
    ) -> ImageResult:
        if quality_mode not in self.QUALITY_TO_RESOLUTION:
            raise ValueError(f"unsupported quality mode: {quality_mode}")

        resolution = resolution or self.QUALITY_TO_RESOLUTION[quality_mode]
        if resolution not in self.COST_PER_IMAGE_USD:
            raise ValueError(f"unsupported resolution: {resolution}")

        prompt = render_image_prompt(svp)
        if reference_image_path is not None:
            prompt = append_reference_usage_policy(prompt=prompt, svp=svp)
        contents = self._build_contents(prompt, reference_image_path)
        resolved_aspect = self._resolve_aspect_ratio(svp.composition_layer.aspect_ratio)
        started = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=[types.Modality.IMAGE],
                    image_config=types.ImageConfig(
                        aspect_ratio=resolved_aspect,
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
            backend=self.BACKEND_NAME,
            aspect_ratio=resolved_aspect,
            native_size_or_resolution=resolution,
            was_aspect_coerced=False,
        )

    def _resolve_aspect_ratio(self, svp_aspect_ratio: str) -> str:
        return self.ASPECT_FALLBACK.get(svp_aspect_ratio, svp_aspect_ratio)

    def _build_contents(self, prompt: str, reference_image_path: Path | None) -> Any:
        if reference_image_path is None:
            return prompt

        path = Path(reference_image_path)
        if not path.exists():
            raise ValueError(f"reference image not found: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        return [
            prompt,
            types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type),
        ]

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
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_RECITATION",
            "NO_IMAGE",
        }
        return normalized in refusal_reasons
