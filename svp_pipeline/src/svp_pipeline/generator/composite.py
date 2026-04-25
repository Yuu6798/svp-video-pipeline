"""Experimental character/background split image generation."""

from __future__ import annotations

import base64
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from openai import OpenAI, OpenAIError
from PIL import Image, ImageEnhance, ImageFilter

from ..exceptions import ImageAPIError, ImageRefusalError
from ..schema import SVPVideo
from .image_openai import OpenAIImageBackend, OpenAIQuality, OpenAISize


@dataclass
class CompositeImageResult:
    """Result of split character/background image generation."""

    png_bytes: bytes
    cost_usd: float
    elapsed_sec: float
    raw_prompt: str
    model: str
    backend: str
    aspect_ratio: str
    native_size_or_resolution: str
    was_aspect_coerced: bool
    character_path: Path
    background_path: Path
    composite_path: Path


class SplitCompositeImageGenerator:
    """Generate character and background separately, then composite them."""

    MODEL_ID = OpenAIImageBackend.MODEL_ID
    BACKEND_NAME = "openai-split-composite"
    QUALITY_MAP: dict[str, OpenAIQuality] = OpenAIImageBackend.QUALITY_MAP
    ASPECT_TO_SIZE: dict[str, OpenAISize] = OpenAIImageBackend.ASPECT_TO_SIZE
    COST_PER_IMAGE_USD = OpenAIImageBackend.COST_PER_IMAGE_USD

    def __init__(self, api_key: str | None = None, client: OpenAI | Any | None = None) -> None:
        self._openai_backend = OpenAIImageBackend(api_key=api_key, client=client)
        self._client = self._openai_backend._client

    def generate(
        self,
        svp: SVPVideo,
        reference_image_path: Path,
        output_dir: Path,
        quality_mode: str = "normal",
    ) -> CompositeImageResult:
        if quality_mode not in self.QUALITY_MAP:
            raise ValueError(f"unsupported quality mode: {quality_mode}")
        if not reference_image_path.exists():
            raise ValueError(f"reference image not found: {reference_image_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        size, was_coerced = self._openai_backend._resolve_size(svp.composition_layer.aspect_ratio)
        quality = self.QUALITY_MAP[quality_mode]
        character_prompt = _render_character_cutout_prompt(svp)
        background_prompt = _render_background_prompt(svp)

        started = time.perf_counter()
        character_bytes = self._generate_character(
            reference_image_path=reference_image_path,
            prompt=character_prompt,
            size=size,
            quality=quality,
        )
        background_bytes = self._generate_background(
            prompt=background_prompt,
            size=size,
            quality=quality,
        )

        character_path = output_dir / "character_green.png"
        background_path = output_dir / "background_clean.png"
        composite_path = output_dir / "composite.png"
        character_path.write_bytes(character_bytes)
        background_path.write_bytes(background_bytes)
        composite = composite_character_background(
            character_path=character_path,
            background_path=background_path,
        )
        composite.save(composite_path)
        png_bytes = composite_path.read_bytes()

        elapsed = time.perf_counter() - started
        image_cost = self.COST_PER_IMAGE_USD[(size, quality)]
        return CompositeImageResult(
            png_bytes=png_bytes,
            cost_usd=image_cost * 2,
            elapsed_sec=elapsed,
            raw_prompt=f"{character_prompt}\n\n--- BACKGROUND ---\n{background_prompt}",
            model=self.MODEL_ID,
            backend=self.BACKEND_NAME,
            aspect_ratio=svp.composition_layer.aspect_ratio,
            native_size_or_resolution=size,
            was_aspect_coerced=was_coerced,
            character_path=character_path,
            background_path=background_path,
            composite_path=composite_path,
        )

    def _generate_character(
        self,
        reference_image_path: Path,
        prompt: str,
        size: OpenAISize,
        quality: OpenAIQuality,
    ) -> bytes:
        reference_file = _build_reference_file(reference_image_path)
        try:
            response = self._client.images.edit(
                model=self.MODEL_ID,
                image=reference_file,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1,
                output_format="png",
            )
        except OpenAIError as exc:
            _raise_openai_image_error(exc)
        return _extract_png_bytes(response)

    def _generate_background(
        self,
        prompt: str,
        size: OpenAISize,
        quality: OpenAIQuality,
    ) -> bytes:
        try:
            response = self._client.images.generate(
                model=self.MODEL_ID,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1,
                output_format="png",
            )
        except OpenAIError as exc:
            _raise_openai_image_error(exc)
        return _extract_png_bytes(response)


def composite_character_background(character_path: Path, background_path: Path) -> Image.Image:
    """Composite a green-screen character image over a background image."""
    character = Image.open(character_path).convert("RGB")
    background = Image.open(background_path).convert("RGB").resize(
        character.size,
        Image.Resampling.LANCZOS,
    )
    arr = np.asarray(character).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    max_rb = np.maximum(r, b)
    green_dom = g - max_rb

    bg_prob = np.clip((green_dom - 10) / 75, 0, 1) * np.clip((g - 65) / 135, 0, 1)
    bright_green = ((g > 115) & (green_dom > 22)).astype(np.float32)
    bg_prob = np.maximum(bg_prob, bright_green * np.clip((green_dom - 18) / 70, 0, 1))
    fg_alpha = 1 - bg_prob
    alpha_img = Image.fromarray(np.clip(fg_alpha * 255, 0, 255).astype(np.uint8), "L")
    alpha_img = alpha_img.filter(ImageFilter.MedianFilter(7))
    alpha_img = alpha_img.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.9))

    spill = np.clip((green_dom - 5) / 50, 0, 1) * np.clip(g / 140, 0, 1)
    arr2 = arr.copy()
    arr2[..., 1] = arr2[..., 1] * (1 - spill * 0.95) + max_rb * (spill * 0.95)
    arr2[..., 0] = np.clip(arr2[..., 0] + spill * 12, 0, 255)
    arr2[..., 2] = np.clip(arr2[..., 2] + spill * 18, 0, 255)
    strong = spill > 0.55
    arr2[..., 0][strong] = np.maximum(arr2[..., 0][strong], arr2[..., 1][strong] * 0.85)
    arr2[..., 2][strong] = np.maximum(arr2[..., 2][strong], arr2[..., 1][strong] * 1.05)
    arr2[..., 1][strong] = np.minimum(
        arr2[..., 1][strong],
        np.maximum(arr2[..., 0][strong], arr2[..., 2][strong]) * 0.92,
    )

    char = Image.fromarray(np.clip(arr2, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    char.putalpha(alpha_img)
    cr, cg, cb, ca = char.split()
    cr = cr.point(lambda x: min(255, int(x * 0.97 + 2)))
    cg = cg.point(lambda x: min(255, int(x * 0.91)))
    cb = cb.point(lambda x: min(255, int(x * 1.07 + 5)))
    char = Image.merge("RGBA", (cr, cg, cb, ca))
    char = ImageEnhance.Contrast(char).enhance(1.06)
    char = ImageEnhance.Color(char).enhance(0.94)

    shadow_alpha = ca.filter(ImageFilter.GaussianBlur(24)).point(lambda x: int(x * 0.24))
    shadow = Image.new("RGBA", char.size, (4, 8, 20, 0))
    shadow.putalpha(shadow_alpha)
    shadow_layer = Image.new("RGBA", char.size, (0, 0, 0, 0))
    shadow_layer.alpha_composite(shadow, dest=(-16, 24))

    bg_rgba = background.convert("RGBA")
    comp = Image.alpha_composite(bg_rgba, shadow_layer)
    comp = Image.alpha_composite(comp, char)
    comp_rgb = comp.convert("RGB")
    comp_rgb = ImageEnhance.Contrast(comp_rgb).enhance(1.03)
    comp_rgb = ImageEnhance.Color(comp_rgb).enhance(1.03)
    return comp_rgb


def _render_character_cutout_prompt(svp: SVPVideo) -> str:
    identity_items = svp.identity_locks or svp.por_core
    required = _join_lines(identity_items + svp.face_layer.distinctive_features)
    return f"""Create a single anime character cutout for compositing.
Use the reference image ONLY for the character identity.
Preserve these character traits:
{required}

Pose and framing:
- {svp.composition_layer.framing}
- {svp.composition_layer.camera_angle}
- {svp.pose_layer.body_pose}
- {svp.pose_layer.hand_state}

Background must be flat pure chroma key green (#00FF00), evenly lit.
No city, no neon signs, no rain, no background texture, no shadows baked into the green area.
Exactly one main weapon/prop if specified. No duplicated blade reflections.
No text, logo, watermark, numbered panels, collage grid, or extra characters.
Keep clean edges suitable for compositing over a cyberpunk rainy neon city background.
"""


def _render_background_prompt(svp: SVPVideo) -> str:
    context = svp.c3.context
    colors = ", ".join(svp.color_axis)
    textures = ", ".join(svp.texture_axis)
    return f"""Create a clean anime background plate for compositing.
Scene context: {context}
Style: {svp.style_family} / {svp.style_pack or "default"}
Colors: {colors}
Textures: {textures}

No people, no characters, no faces, no bodies, no swords, no katana, no weapon-like silhouettes.
Wet street reflections, magenta and cyan neon bokeh, deep perspective, cinematic clean background.
Smooth distant details, reduced micro-line density.
No gritty texture noise, no scratch-like artifacts, no compression speckles.
No text, logo, or watermark.
"""


def _join_lines(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item)


def _build_reference_file(reference_image_path: Path) -> tuple[str, bytes, str]:
    mime_type = mimetypes.guess_type(reference_image_path.name)[0] or "image/png"
    return (reference_image_path.name, reference_image_path.read_bytes(), mime_type)


def _extract_png_bytes(response: Any) -> bytes:
    data_items = getattr(response, "data", None) or []
    if not data_items:
        raise ImageRefusalError("openai image generation returned empty data")
    b64_json = getattr(data_items[0], "b64_json", None)
    if not b64_json:
        raise ImageRefusalError("openai image generation returned no image payload")
    try:
        return base64.b64decode(b64_json)
    except Exception as exc:  # noqa: BLE001
        raise ImageAPIError("openai image payload decode failed") from exc


def _raise_openai_image_error(exc: OpenAIError) -> Literal[False]:
    if OpenAIImageBackend._is_content_policy_error(exc):
        raise ImageRefusalError("openai image generation refused by content policy") from exc
    if getattr(exc, "status_code", None) == 401:
        raise ImageAPIError("openai image generation failed: unauthorized (401)") from exc
    if getattr(exc, "status_code", None) == 429:
        raise ImageAPIError("openai image generation failed: rate limited (429)") from exc
    raise ImageAPIError("openai image generation failed") from exc
