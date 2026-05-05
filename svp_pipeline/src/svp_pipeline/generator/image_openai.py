"""OpenAI gpt-image-2 backend for SVP pipeline."""

from __future__ import annotations

import base64
import mimetypes
import os
import time
import warnings
from pathlib import Path
from typing import Any, Literal, NoReturn

try:
    from openai import OpenAI, OpenAIError
except ImportError:  # pragma: no cover - covered by dependency install in CI
    OpenAI = Any  # type: ignore[assignment,misc]

    class OpenAIError(Exception):
        """Fallback OpenAIError when openai package is unavailable."""


from ..exceptions import ImageAPIError, ImageRefusalError
from ..schema import SVPVideo
from ..utils.prompt_render import append_reference_usage_policy, render_image_prompt
from .image_base import ImageResult

OpenAIQuality = Literal["low", "medium", "high", "auto"]
OpenAISize = Literal["1024x1024", "1536x1024", "1024x1536", "auto"]
ReferenceFile = tuple[str, bytes, str]


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

    @property
    def client(self) -> OpenAI | Any:
        """OpenAI client used by this backend, exposed for split-composite reuse."""
        return self._client

    def generate(
        self,
        svp: SVPVideo,
        quality_mode: str = "normal",
        reference_image_path: Path | None = None,
    ) -> ImageResult:
        quality = self._resolve_quality(quality_mode)
        prompt = self._render_prompt(svp, reference_image_path)
        size, was_coerced = self._resolve_size(svp.composition_layer.aspect_ratio)
        reference_file = self._resolve_reference_file(reference_image_path)
        self._warn_if_aspect_coerced(
            svp_aspect_ratio=svp.composition_layer.aspect_ratio,
            size=size,
            was_coerced=was_coerced,
        )

        started = time.perf_counter()
        response = self._request_image(
            prompt=prompt,
            size=size,
            quality=quality,
            reference_file=reference_file,
        )
        png_bytes = self._extract_png_bytes(response)
        elapsed_sec = time.perf_counter() - started
        return self._build_result(
            svp=svp,
            png_bytes=png_bytes,
            prompt=prompt,
            size=size,
            quality=quality,
            elapsed_sec=elapsed_sec,
            was_coerced=was_coerced,
        )

    def _resolve_quality(self, quality_mode: str) -> OpenAIQuality:
        quality = self.QUALITY_MAP.get(quality_mode)
        if quality is None:
            raise ValueError(f"unsupported quality mode: {quality_mode}")
        return quality

    @staticmethod
    def _render_prompt(svp: SVPVideo, reference_image_path: Path | None) -> str:
        prompt = render_image_prompt(svp)
        if reference_image_path is not None:
            return append_reference_usage_policy(prompt=prompt, svp=svp)
        return prompt

    def _resolve_reference_file(self, reference_image_path: Path | None) -> ReferenceFile | None:
        if reference_image_path is None:
            return None
        return self._build_reference_file(reference_image_path)

    @staticmethod
    def _warn_if_aspect_coerced(
        *,
        svp_aspect_ratio: str,
        size: OpenAISize,
        was_coerced: bool,
    ) -> None:
        if not was_coerced:
            return
        warnings.warn(
            (f"OpenAI gpt-image-2 coerced aspect ratio {svp_aspect_ratio!r} -> {size!r}"),
            UserWarning,
            stacklevel=2,
        )

    def _request_image(
        self,
        *,
        prompt: str,
        size: OpenAISize,
        quality: OpenAIQuality,
        reference_file: ReferenceFile | None,
    ) -> Any:
        try:
            if reference_file is None:
                return self._client.images.generate(
                    model=self.model,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=1,
                    output_format="png",
                )
            return self._client.images.edit(
                model=self.model,
                image=reference_file,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1,
                output_format="png",
            )
        except OpenAIError as exc:
            self._raise_wrapped_openai_error(exc)
        except Exception as exc:  # noqa: BLE001
            raise ImageAPIError("openai image generation failed") from exc

    def _raise_wrapped_openai_error(self, exc: OpenAIError) -> NoReturn:
        if self._is_content_policy_error(exc):
            raise ImageRefusalError("openai image generation refused by content policy") from exc
        if getattr(exc, "status_code", None) == 401:
            raise ImageAPIError("openai image generation failed: unauthorized (401)") from exc
        if getattr(exc, "status_code", None) == 429:
            raise ImageAPIError("openai image generation failed: rate limited (429)") from exc
        raise ImageAPIError("openai image generation failed") from exc

    def _build_result(
        self,
        *,
        svp: SVPVideo,
        png_bytes: bytes,
        prompt: str,
        size: OpenAISize,
        quality: OpenAIQuality,
        elapsed_sec: float,
        was_coerced: bool,
    ) -> ImageResult:
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
    def _build_reference_file(reference_image_path: Path) -> tuple[str, bytes, str]:
        path = Path(reference_image_path)
        if not path.exists():
            raise ValueError(f"reference image not found: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"failed to read reference image: {path}") from exc
        return (path.name, data, mime_type)

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
