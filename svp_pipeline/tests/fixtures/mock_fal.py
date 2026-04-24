"""fal_client and MP4 download test fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

TINY_MP4_BYTES: bytes = (
    b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    b"\x00\x00\x00\x08free\x00\x00\x00\x08mdat"
)


def build_successful_subscribe_response(
    mp4_url: str = "https://mock.fal.media/test.mp4",
) -> dict:
    """Build a successful fal subscribe payload."""
    return {
        "video": {
            "url": mp4_url,
            "content_type": "video/mp4",
            "file_size": 100000,
        },
        "request_id": "req_mock_123",
        "seed": 42,
    }


def build_timeout_mock() -> MagicMock:
    """Build a callable that raises TimeoutError."""
    mock = MagicMock()
    mock.side_effect = TimeoutError("timed out")
    return mock


def build_api_error_mock() -> MagicMock:
    """Build a callable that raises generic API failure."""
    mock = MagicMock()
    mock.side_effect = RuntimeError("fal api failed")
    return mock


def patch_fal_upload(monkeypatch, url: str = "https://storage.fal.ai/mock-image.png") -> None:
    """Patch fal_client.upload_file in video generator module."""
    monkeypatch.setattr("svp_pipeline.generator.video.fal_client.upload_file", lambda _path: url)


def patch_fal_subscribe(monkeypatch, response: dict) -> None:
    """Patch fal_client.subscribe in video generator module."""
    monkeypatch.setattr(
        "svp_pipeline.generator.video.fal_client.subscribe",
        lambda *_args, **_kwargs: response,
    )


def patch_httpx_download(monkeypatch, content: bytes = TINY_MP4_BYTES) -> None:
    """Patch httpx.get in video generator module to return bytes."""

    def _fake_get(*_args, **_kwargs):
        response = MagicMock()
        response.content = content
        response.raise_for_status.return_value = None
        return response

    monkeypatch.setattr("svp_pipeline.generator.video.httpx.get", _fake_get)
