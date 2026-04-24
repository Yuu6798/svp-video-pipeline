"""Tests for VideoGenerator (Seedance r2v)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from svp_pipeline.exceptions import VideoAPIError, VideoDownloadError, VideoTimeoutError
from svp_pipeline.generator.video import VideoGenerator
from svp_pipeline.schema import SVPVideo
from tests.fixtures.mock_fal import (
    TINY_MP4_BYTES,
    build_successful_subscribe_response,
    patch_fal_upload,
    patch_httpx_download,
)
from tests.fixtures.mock_gemini import TINY_PNG_BYTES

SAMPLES_DIR = Path(__file__).parent / "samples"
SYNC_SUBSCRIBE_PATH = "svp_pipeline.generator.video.fal_client.SyncClient.subscribe"


def _load(name: str) -> SVPVideo:
    return SVPVideo.model_validate_json((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def _write_ref_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "ref.png"
    image_path.write_bytes(TINY_PNG_BYTES)
    return image_path


def _make_generator(
    *,
    tier: str = "standard",
    timeout_mode: str = "thread",
) -> VideoGenerator:
    return VideoGenerator(tier=tier, api_key="test", timeout_mode=timeout_mode)  # type: ignore[arg-type]


def test_generate_returns_video_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svp = _load("shibuya_dusk.json")
    image_path = _write_ref_image(tmp_path)
    output_path = tmp_path / "video.mp4"

    patch_fal_upload(monkeypatch)
    subscribe_mock = MagicMock(return_value=build_successful_subscribe_response())
    monkeypatch.setattr(SYNC_SUBSCRIBE_PATH, subscribe_mock)
    patch_httpx_download(monkeypatch, TINY_MP4_BYTES)

    generator = VideoGenerator(tier="standard", api_key="test-fal-key", timeout_mode="thread")
    result = generator.generate(svp=svp, image_path=image_path, output_path=output_path)

    assert result.mp4_path == output_path
    assert output_path.exists()
    assert output_path.read_bytes() == TINY_MP4_BYTES
    assert result.tier == "standard"
    assert result.resolution == "720p"


def test_standard_tier_uses_correct_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svp = _load("shibuya_dusk.json")
    image_path = _write_ref_image(tmp_path)

    patch_fal_upload(monkeypatch)
    subscribe_mock = MagicMock(return_value=build_successful_subscribe_response())
    monkeypatch.setattr(SYNC_SUBSCRIBE_PATH, subscribe_mock)
    patch_httpx_download(monkeypatch)

    _make_generator(tier="standard").generate(
        svp=svp,
        image_path=image_path,
        output_path=tmp_path / "video.mp4",
    )

    assert subscribe_mock.call_args.args[0] == VideoGenerator.ENDPOINT_STANDARD


def test_fast_tier_uses_correct_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svp = _load("shibuya_dusk.json")
    image_path = _write_ref_image(tmp_path)

    patch_fal_upload(monkeypatch)
    subscribe_mock = MagicMock(return_value=build_successful_subscribe_response())
    monkeypatch.setattr(SYNC_SUBSCRIBE_PATH, subscribe_mock)
    patch_httpx_download(monkeypatch)

    _make_generator(tier="fast").generate(
        svp=svp,
        image_path=image_path,
        output_path=tmp_path / "video.mp4",
    )

    assert subscribe_mock.call_args.args[0] == VideoGenerator.ENDPOINT_FAST


def test_duration_passed_as_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svp = _load("shibuya_dusk.json")
    image_path = _write_ref_image(tmp_path)

    patch_fal_upload(monkeypatch)
    subscribe_mock = MagicMock(return_value=build_successful_subscribe_response())
    monkeypatch.setattr(SYNC_SUBSCRIBE_PATH, subscribe_mock)
    patch_httpx_download(monkeypatch)

    _make_generator().generate(
        svp=svp,
        image_path=image_path,
        output_path=tmp_path / "video.mp4",
    )

    args = subscribe_mock.call_args.kwargs["arguments"]
    assert args["duration"] == str(svp.motion_layer.duration_seconds)
    assert isinstance(args["duration"], str)


def test_aspect_ratio_passed_directly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svp = _load("action_ninja.json")
    image_path = _write_ref_image(tmp_path)

    patch_fal_upload(monkeypatch)
    subscribe_mock = MagicMock(return_value=build_successful_subscribe_response())
    monkeypatch.setattr(SYNC_SUBSCRIBE_PATH, subscribe_mock)
    patch_httpx_download(monkeypatch)

    _make_generator().generate(
        svp=svp,
        image_path=image_path,
        output_path=tmp_path / "video.mp4",
    )

    args = subscribe_mock.call_args.kwargs["arguments"]
    assert args["aspect_ratio"] == "21:9"


def test_image_url_in_image_urls_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svp = _load("shibuya_dusk.json")
    image_path = _write_ref_image(tmp_path)
    image_url = "https://storage.fal.ai/mock-image.png"

    patch_fal_upload(monkeypatch, url=image_url)
    subscribe_mock = MagicMock(return_value=build_successful_subscribe_response())
    monkeypatch.setattr(SYNC_SUBSCRIBE_PATH, subscribe_mock)
    patch_httpx_download(monkeypatch)

    _make_generator().generate(
        svp=svp,
        image_path=image_path,
        output_path=tmp_path / "video.mp4",
    )

    args = subscribe_mock.call_args.kwargs["arguments"]
    assert isinstance(args["image_urls"], list)
    assert args["image_urls"] == [image_url]


def test_generate_audio_always_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svp = _load("shibuya_dusk.json")
    image_path = _write_ref_image(tmp_path)

    patch_fal_upload(monkeypatch)
    subscribe_mock = MagicMock(return_value=build_successful_subscribe_response())
    monkeypatch.setattr(SYNC_SUBSCRIBE_PATH, subscribe_mock)
    patch_httpx_download(monkeypatch)

    _make_generator().generate(
        svp=svp,
        image_path=image_path,
        output_path=tmp_path / "video.mp4",
    )
    args = subscribe_mock.call_args.kwargs["arguments"]
    assert args["generate_audio"] is False


def test_resolution_720p_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svp = _load("shibuya_dusk.json")
    image_path = _write_ref_image(tmp_path)

    patch_fal_upload(monkeypatch)
    subscribe_mock = MagicMock(return_value=build_successful_subscribe_response())
    monkeypatch.setattr(SYNC_SUBSCRIBE_PATH, subscribe_mock)
    patch_httpx_download(monkeypatch)

    _make_generator().generate(
        svp=svp,
        image_path=image_path,
        output_path=tmp_path / "video.mp4",
    )
    args = subscribe_mock.call_args.kwargs["arguments"]
    assert args["resolution"] == "720p"


def test_resolution_480p_applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svp = _load("shibuya_dusk.json")
    image_path = _write_ref_image(tmp_path)

    patch_fal_upload(monkeypatch)
    subscribe_mock = MagicMock(return_value=build_successful_subscribe_response())
    monkeypatch.setattr(SYNC_SUBSCRIBE_PATH, subscribe_mock)
    patch_httpx_download(monkeypatch)

    _make_generator().generate(
        svp=svp,
        image_path=image_path,
        output_path=tmp_path / "video.mp4",
        resolution="480p",
    )
    args = subscribe_mock.call_args.kwargs["arguments"]
    assert args["resolution"] == "480p"


def test_cost_calculation_standard_720p_5sec() -> None:
    generator = _make_generator(tier="standard")
    assert generator._calculate_cost("standard", "720p", 5) == pytest.approx(1.512)


def test_cost_calculation_fast_720p_5sec() -> None:
    generator = _make_generator(tier="fast")
    assert generator._calculate_cost("fast", "720p", 5) == pytest.approx(1.2095)


def test_cost_calculation_cheap_combo() -> None:
    generator = _make_generator(tier="fast")
    assert generator._calculate_cost("fast", "480p", 5) == pytest.approx(0.5375)


def test_timeout_raises_video_timeout_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svp = _load("shibuya_dusk.json")
    image_path = _write_ref_image(tmp_path)
    patch_fal_upload(monkeypatch)
    patch_httpx_download(monkeypatch)

    generator = _make_generator()
    monkeypatch.setattr(
        generator,
        "_subscribe_with_timeout",
        MagicMock(side_effect=VideoTimeoutError("timeout")),
    )

    with pytest.raises(VideoTimeoutError):
        generator.generate(
            svp=svp,
            image_path=image_path,
            output_path=tmp_path / "video.mp4",
        )


def test_api_error_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svp = _load("shibuya_dusk.json")
    image_path = _write_ref_image(tmp_path)
    patch_fal_upload(monkeypatch)
    patch_httpx_download(monkeypatch)

    generator = _make_generator()
    monkeypatch.setattr(
        generator,
        "_subscribe_with_timeout",
        MagicMock(side_effect=RuntimeError("bad request")),
    )

    with pytest.raises(VideoAPIError):
        generator.generate(
            svp=svp,
            image_path=image_path,
            output_path=tmp_path / "video.mp4",
        )


def test_download_failure_preserves_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svp = _load("shibuya_dusk.json")
    image_path = _write_ref_image(tmp_path)
    mp4_url = "https://mock.fal.media/failure.mp4"

    patch_fal_upload(monkeypatch)
    subscribe_mock = MagicMock(return_value=build_successful_subscribe_response(mp4_url=mp4_url))
    monkeypatch.setattr(SYNC_SUBSCRIBE_PATH, subscribe_mock)

    def _raise_download(*_args, **_kwargs):
        raise httpx.ConnectError("network", request=httpx.Request("GET", mp4_url))

    monkeypatch.setattr("svp_pipeline.generator.video.httpx.get", _raise_download)

    generator = _make_generator()
    with pytest.raises(VideoDownloadError) as exc_info:
        generator.generate(
            svp=svp,
            image_path=image_path,
            output_path=tmp_path / "video.mp4",
        )

    assert exc_info.value.mp4_url == mp4_url


def test_retry_on_transient_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svp = _load("shibuya_dusk.json")
    image_path = _write_ref_image(tmp_path)

    patch_fal_upload(monkeypatch)
    patch_httpx_download(monkeypatch)
    monkeypatch.setattr("svp_pipeline.generator.video.time.sleep", lambda _s: None)

    generator = _make_generator()
    subscribe = MagicMock(
        side_effect=[
            RuntimeError("connection reset by peer"),
            build_successful_subscribe_response(),
        ]
    )
    monkeypatch.setattr(generator, "_subscribe_with_timeout", subscribe)

    result = generator.generate(
        svp=svp,
        image_path=image_path,
        output_path=tmp_path / "video.mp4",
    )
    assert result.mp4_path.exists()
    assert subscribe.call_count == 2


def test_max_retries_exceeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svp = _load("shibuya_dusk.json")
    image_path = _write_ref_image(tmp_path)

    patch_fal_upload(monkeypatch)
    patch_httpx_download(monkeypatch)
    monkeypatch.setattr("svp_pipeline.generator.video.time.sleep", lambda _s: None)

    generator = _make_generator()
    subscribe = MagicMock(
        side_effect=[
            RuntimeError("connection reset 1"),
            RuntimeError("connection reset 2"),
            RuntimeError("connection reset 3"),
        ]
    )
    monkeypatch.setattr(generator, "_subscribe_with_timeout", subscribe)

    with pytest.raises(VideoAPIError):
        generator.generate(
            svp=svp,
            image_path=image_path,
            output_path=tmp_path / "video.mp4",
        )
    assert subscribe.call_count == 3


def test_timeout_process_mode_terminates_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from queue import Empty

    class _FakeProcess:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs
            self._alive = True
            self.started = False
            self.terminated = False
            self.join_calls: list[float | int | None] = []

        def start(self) -> None:
            self.started = True

        def join(self, timeout=None) -> None:
            self.join_calls.append(timeout)

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self.terminated = True
            self._alive = False

    class _FakeQueue:
        def __init__(self, *args, **kwargs) -> None:
            self.timeouts: list[float | int | None] = []

        def get(self, timeout=None):
            self.timeouts.append(timeout)
            raise Empty

    holder: dict[str, _FakeProcess | _FakeQueue] = {}

    def _queue_factory(*args, **kwargs):
        queue = _FakeQueue(*args, **kwargs)
        holder["queue"] = queue
        return queue

    def _process_factory(*args, **kwargs):
        proc = _FakeProcess(*args, **kwargs)
        holder["process"] = proc
        return proc

    monkeypatch.setattr("svp_pipeline.generator.video.Process", _process_factory)
    monkeypatch.setattr("svp_pipeline.generator.video.Queue", _queue_factory)

    generator = _make_generator(timeout_mode="process")
    with pytest.raises(VideoTimeoutError):
        generator._subscribe_with_timeout(endpoint="mock/endpoint", arguments={})

    process = holder["process"]
    queue = holder["queue"]
    assert process.started is True
    assert process.terminated is True
    assert process.join_calls[-1] == 5
    assert queue.timeouts == [generator.TIMEOUT_SEC]
