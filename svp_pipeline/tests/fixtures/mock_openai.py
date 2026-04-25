"""OpenAI API mock fixtures for image backend tests."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

from svp_pipeline.generator.image_openai import OpenAIError


class MockOpenAIError(OpenAIError):
    """Simple OpenAIError-compatible exception for tests."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.body = {"error": {"code": code, "message": message}}


def build_image_response(png_bytes: bytes) -> MagicMock:
    response = MagicMock()
    response.data = [MagicMock()]
    response.data[0].b64_json = base64.b64encode(png_bytes).decode("ascii")
    return response


def build_content_policy_error() -> MockOpenAIError:
    return MockOpenAIError(
        "Request blocked by content policy",
        status_code=400,
        code="content_policy_violation",
    )
