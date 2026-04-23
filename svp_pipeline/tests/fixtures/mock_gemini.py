"""Gemini response fixtures for image tests."""

from __future__ import annotations

from unittest.mock import MagicMock

TINY_PNG_BYTES: bytes = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\xfa\xcf\x00\x00\x00\x03\x00"
    b"\x01\xc4x\xbfg\x00\x00\x00\x00IEND\xaeB`\x82"
)


def build_image_response(png_bytes: bytes) -> MagicMock:
    response = MagicMock()
    candidate = MagicMock()
    candidate.finish_reason = "STOP"
    part = MagicMock()
    inline_data = MagicMock()
    inline_data.data = png_bytes
    inline_data.mime_type = "image/png"
    part.inline_data = inline_data
    content = MagicMock()
    content.parts = [part]
    candidate.content = content
    response.candidates = [candidate]
    response.prompt_feedback = None
    return response


def build_refusal_response(reason: str = "SAFETY") -> MagicMock:
    response = MagicMock()
    candidate = MagicMock()
    candidate.finish_reason = reason
    content = MagicMock()
    content.parts = []
    candidate.content = content
    response.candidates = [candidate]
    response.prompt_feedback = None
    return response
