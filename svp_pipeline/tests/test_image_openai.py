"""Tests for OpenAIImageBackend."""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from svp_pipeline.exceptions import ImageAPIError, ImageRefusalError
from svp_pipeline.generator.image_openai import OpenAIImageBackend
from svp_pipeline.schema import SVPVideo
from tests.fixtures.mock_gemini import TINY_PNG_BYTES
from tests.fixtures.mock_openai import (
    MockOpenAIError,
    build_content_policy_error,
    build_image_response,
)

SAMPLES_DIR = Path(__file__).parent / "samples"


def _load(name: str) -> SVPVideo:
    return SVPVideo.model_validate_json((SAMPLES_DIR / name).read_text(encoding="utf-8"))


class DummyOpenAIClient:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.images = MagicMock()
        if error is not None:
            self.images.generate.side_effect = error
            self.images.edit.side_effect = error
        else:
            self.images.generate.return_value = response
            self.images.edit.return_value = response


def _with_aspect(svp: SVPVideo, aspect: str) -> SVPVideo:
    payload = copy.deepcopy(svp.model_dump())
    payload["composition_layer"]["aspect_ratio"] = aspect
    return SVPVideo.model_validate(payload)


def test_generate_returns_image_result() -> None:
    svp = _load("shibuya_dusk.json")
    backend = OpenAIImageBackend(
        client=DummyOpenAIClient(response=build_image_response(TINY_PNG_BYTES))
    )

    result = backend.generate(svp=svp, quality_mode="normal")

    assert result.png_bytes == TINY_PNG_BYTES
    assert result.backend == "openai"
    assert result.model == "gpt-image-2"


def test_size_mapping_16_9_landscape() -> None:
    svp = _load("shibuya_dusk.json")
    client = DummyOpenAIClient(response=build_image_response(TINY_PNG_BYTES))
    backend = OpenAIImageBackend(client=client)

    backend.generate(svp=svp, quality_mode="normal")

    assert client.images.generate.call_args.kwargs["size"] == "1536x1024"


def test_size_mapping_1_1_square() -> None:
    svp = _with_aspect(_load("shibuya_dusk.json"), "1:1")
    client = DummyOpenAIClient(response=build_image_response(TINY_PNG_BYTES))
    backend = OpenAIImageBackend(client=client)

    backend.generate(svp=svp, quality_mode="normal")

    assert client.images.generate.call_args.kwargs["size"] == "1024x1024"


def test_size_mapping_9_16_portrait() -> None:
    svp = _with_aspect(_load("shibuya_dusk.json"), "9:16")
    client = DummyOpenAIClient(response=build_image_response(TINY_PNG_BYTES))
    backend = OpenAIImageBackend(client=client)

    backend.generate(svp=svp, quality_mode="normal")

    assert client.images.generate.call_args.kwargs["size"] == "1024x1536"


def test_size_mapping_auto_passthrough() -> None:
    svp = _with_aspect(_load("shibuya_dusk.json"), "auto")
    client = DummyOpenAIClient(response=build_image_response(TINY_PNG_BYTES))
    backend = OpenAIImageBackend(client=client)

    backend.generate(svp=svp, quality_mode="normal")

    assert client.images.generate.call_args.kwargs["size"] == "auto"


def test_size_coercion_21_9_to_landscape() -> None:
    svp = _load("action_ninja.json")
    backend = OpenAIImageBackend(
        client=DummyOpenAIClient(response=build_image_response(TINY_PNG_BYTES))
    )

    with pytest.warns(UserWarning, match="coerced aspect ratio"):
        result = backend.generate(svp=svp, quality_mode="normal")

    assert result.native_size_or_resolution == "1536x1024"
    assert result.was_aspect_coerced is True


def test_quality_mode_normal_to_high() -> None:
    svp = _load("shibuya_dusk.json")
    client = DummyOpenAIClient(response=build_image_response(TINY_PNG_BYTES))
    backend = OpenAIImageBackend(client=client)

    backend.generate(svp=svp, quality_mode="normal")

    assert client.images.generate.call_args.kwargs["quality"] == "high"


def test_quality_mode_cheap_to_low() -> None:
    svp = _load("shibuya_dusk.json")
    client = DummyOpenAIClient(response=build_image_response(TINY_PNG_BYTES))
    backend = OpenAIImageBackend(client=client)

    backend.generate(svp=svp, quality_mode="cheap")

    assert client.images.generate.call_args.kwargs["quality"] == "low"


def test_cost_calculation_1536x1024_high() -> None:
    svp = _load("shibuya_dusk.json")
    backend = OpenAIImageBackend(
        client=DummyOpenAIClient(response=build_image_response(TINY_PNG_BYTES))
    )

    result = backend.generate(svp=svp, quality_mode="normal")

    assert result.cost_usd == 0.25


def test_b64_decoded_to_bytes() -> None:
    svp = _load("shibuya_dusk.json")
    backend = OpenAIImageBackend(
        client=DummyOpenAIClient(response=build_image_response(TINY_PNG_BYTES))
    )

    result = backend.generate(svp=svp, quality_mode="normal")

    assert result.png_bytes == TINY_PNG_BYTES


def test_reference_image_uses_edit_endpoint(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    reference = tmp_path / "reference.png"
    reference.write_bytes(TINY_PNG_BYTES)
    client = DummyOpenAIClient(response=build_image_response(TINY_PNG_BYTES))
    backend = OpenAIImageBackend(client=client)

    result = backend.generate(
        svp=svp,
        quality_mode="cheap",
        reference_image_path=reference,
    )

    assert result.png_bytes == TINY_PNG_BYTES
    client.images.generate.assert_not_called()
    client.images.edit.assert_called_once()
    kwargs = client.images.edit.call_args.kwargs
    assert kwargs["image"] == ("reference.png", TINY_PNG_BYTES, "image/png")
    assert "Reference Image Usage Policy" in kwargs["prompt"]
    assert "extra swords or weapon trails" in kwargs["prompt"]
    assert "Object instance rules" in kwargs["prompt"]
    assert "Background quality rules" in kwargs["prompt"]


def test_reference_file_read_error_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="failed to read reference image"):
        OpenAIImageBackend._build_reference_file(tmp_path)


def test_image_result_backend_field() -> None:
    svp = _load("shibuya_dusk.json")
    backend = OpenAIImageBackend(
        client=DummyOpenAIClient(response=build_image_response(TINY_PNG_BYTES))
    )

    result = backend.generate(svp=svp, quality_mode="normal")

    assert result.backend == "openai"


def test_image_result_native_size_field() -> None:
    svp = _load("shibuya_dusk.json")
    backend = OpenAIImageBackend(
        client=DummyOpenAIClient(response=build_image_response(TINY_PNG_BYTES))
    )

    result = backend.generate(svp=svp, quality_mode="normal")

    assert result.native_size_or_resolution == "1536x1024"


def test_content_policy_refusal() -> None:
    svp = _load("shibuya_dusk.json")
    backend = OpenAIImageBackend(client=DummyOpenAIClient(error=build_content_policy_error()))

    with pytest.raises(ImageRefusalError):
        backend.generate(svp=svp, quality_mode="normal")


def test_generic_api_error_wrapped() -> None:
    svp = _load("shibuya_dusk.json")
    backend = OpenAIImageBackend(
        client=DummyOpenAIClient(
            error=MockOpenAIError("boom", status_code=500, code="server_error")
        )
    )

    with pytest.raises(ImageAPIError):
        backend.generate(svp=svp, quality_mode="normal")


def test_auth_error_wrapped() -> None:
    svp = _load("shibuya_dusk.json")
    backend = OpenAIImageBackend(
        client=DummyOpenAIClient(
            error=MockOpenAIError("unauthorized", status_code=401, code="invalid_api_key")
        )
    )

    with pytest.raises(ImageAPIError, match="unauthorized"):
        backend.generate(svp=svp, quality_mode="normal")


def test_coercion_emits_warning() -> None:
    svp = _load("action_ninja.json")
    backend = OpenAIImageBackend(
        client=DummyOpenAIClient(response=build_image_response(TINY_PNG_BYTES))
    )

    with pytest.warns(UserWarning, match="coerced aspect ratio"):
        backend.generate(svp=svp, quality_mode="normal")
