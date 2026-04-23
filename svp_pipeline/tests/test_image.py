"""Tests for ImageGenerator."""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from svp_pipeline.exceptions import ImageAPIError, ImageRefusalError
from svp_pipeline.generator.image import ImageGenerator
from svp_pipeline.schema import SVPVideo
from tests.fixtures.mock_gemini import TINY_PNG_BYTES, build_image_response, build_refusal_response

SAMPLES_DIR = Path(__file__).parent / "samples"


def _load(name: str) -> SVPVideo:
    return SVPVideo.model_validate_json((SAMPLES_DIR / name).read_text(encoding="utf-8"))


class DummyClient:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.models = MagicMock()
        if error is not None:
            self.models.generate_content.side_effect = error
        else:
            self.models.generate_content.return_value = response


def test_generate_returns_image_result() -> None:
    svp = _load("shibuya_dusk.json")
    client = DummyClient(response=build_image_response(TINY_PNG_BYTES))
    generator = ImageGenerator(client=client)

    result = generator.generate(svp=svp)

    assert result.png_bytes == TINY_PNG_BYTES
    assert result.model == "gemini-3-pro-image-preview"
    assert result.resolution == "2K"
    assert result.aspect_ratio == "16:9"


def test_generate_passes_correct_aspect_ratio() -> None:
    svp = _load("action_ninja.json")
    client = DummyClient(response=build_image_response(TINY_PNG_BYTES))
    generator = ImageGenerator(client=client)

    generator.generate(svp=svp, resolution="2K")

    config = client.models.generate_content.call_args.kwargs["config"]
    image_config = config.image_config
    actual = getattr(image_config, "aspect_ratio", None) or getattr(
        image_config,
        "aspectRatio",
        None,
    )
    assert actual == "21:9"


def test_generate_passes_correct_resolution() -> None:
    svp = _load("shibuya_dusk.json")
    client = DummyClient(response=build_image_response(TINY_PNG_BYTES))
    generator = ImageGenerator(client=client)

    generator.generate(svp=svp, resolution="1K")

    config = client.models.generate_content.call_args.kwargs["config"]
    image_config = config.image_config
    actual = getattr(image_config, "image_size", None) or getattr(image_config, "imageSize", None)
    assert actual == "1K"


def test_auto_aspect_ratio_fallback() -> None:
    svp = _load("shibuya_dusk.json")
    payload = copy.deepcopy(svp.model_dump())
    payload["composition_layer"]["aspect_ratio"] = "auto"
    svp_auto = SVPVideo.model_validate(payload)

    client = DummyClient(response=build_image_response(TINY_PNG_BYTES))
    generator = ImageGenerator(client=client)
    result = generator.generate(svp=svp_auto, resolution="2K")

    config = client.models.generate_content.call_args.kwargs["config"]
    image_config = config.image_config
    actual = getattr(image_config, "aspect_ratio", None) or getattr(
        image_config,
        "aspectRatio",
        None,
    )
    assert actual == "16:9"
    assert result.aspect_ratio == "16:9"


def test_cost_calculation() -> None:
    svp = _load("shibuya_dusk.json")
    client = DummyClient(response=build_image_response(TINY_PNG_BYTES))
    generator = ImageGenerator(client=client)

    result = generator.generate(svp=svp, resolution="4K")

    assert result.cost_usd == generator.COST_PER_IMAGE_USD["4K"]


def test_raw_prompt_in_result() -> None:
    svp = _load("shibuya_dusk.json")
    client = DummyClient(response=build_image_response(TINY_PNG_BYTES))
    generator = ImageGenerator(client=client)

    result = generator.generate(svp=svp)

    assert svp.por_identity in result.raw_prompt


def test_refusal_raises_image_refusal_error() -> None:
    svp = _load("shibuya_dusk.json")
    client = DummyClient(response=build_refusal_response("SAFETY"))
    generator = ImageGenerator(client=client)

    with pytest.raises(ImageRefusalError):
        generator.generate(svp=svp)


def test_api_error_wrapped() -> None:
    svp = _load("shibuya_dusk.json")
    client = DummyClient(error=RuntimeError("upstream failed"))
    generator = ImageGenerator(client=client)

    with pytest.raises(ImageAPIError):
        generator.generate(svp=svp)


def test_mock_png_bytes_are_valid() -> None:
    assert TINY_PNG_BYTES.startswith(b"\x89PNG\r\n\x1a\n")
