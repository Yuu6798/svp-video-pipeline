"""Seedance 2.0 reference-to-video generation layer."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from typing import Any, Literal

import fal_client
import httpx

from ..exceptions import VideoAPIError, VideoDownloadError, VideoTimeoutError
from ..schema import SVPVideo
from ..utils.prompt_render import render_motion_prompt

VideoTier = Literal["standard", "fast"]
VideoResolution = Literal["480p", "720p"]
TimeoutMode = Literal["process", "thread"]


def _subscribe_worker(
    endpoint: str,
    arguments: dict[str, Any],
    api_key: str,
    result_queue: Queue,
) -> None:
    """Run fal subscribe in a child process and return payload via queue."""
    try:
        client = fal_client.SyncClient(key=api_key)
        result = client.subscribe(
            endpoint,
            arguments=arguments,
            with_logs=True,
        )
    except Exception as exc:  # noqa: BLE001
        result_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    else:
        result_queue.put(
            {
                "ok": True,
                "result": result,
            }
        )


@dataclass
class VideoResult:
    """Video generation result payload."""

    mp4_path: Path
    cost_usd: float
    elapsed_sec: float
    raw_prompt: str
    tier: VideoTier
    resolution: VideoResolution
    aspect_ratio: str
    duration_seconds: int
    fal_request_id: str | None
    mp4_url: str


class VideoGenerator:
    """Generate MP4 video from SVPVideo + reference image via Seedance r2v."""

    ENDPOINT_STANDARD: str = "bytedance/seedance-2.0/reference-to-video"
    ENDPOINT_FAST: str = "bytedance/seedance-2.0/fast/reference-to-video"

    PRICE_PER_SECOND: dict[tuple[VideoTier, VideoResolution], float] = {
        ("standard", "720p"): 0.3024,
        ("standard", "480p"): 0.1344,
        ("fast", "720p"): 0.2419,
        ("fast", "480p"): 0.1075,
    }

    TIMEOUT_SEC: int = 300
    MAX_RETRIES: int = 2

    def __init__(
        self,
        tier: VideoTier = "standard",
        api_key: str | None = None,
        timeout_mode: TimeoutMode = "process",
    ) -> None:
        if tier not in ("standard", "fast"):
            raise ValueError(f"unsupported video tier: {tier!r}")
        self.tier: VideoTier = tier
        if timeout_mode not in ("process", "thread"):
            raise ValueError(f"unsupported timeout_mode: {timeout_mode!r}")
        self.timeout_mode: TimeoutMode = timeout_mode

        resolved_key = api_key or os.getenv("FAL_KEY")
        if not resolved_key:
            raise ValueError("FAL_KEY is required for VideoGenerator")
        self._api_key = resolved_key
        self._fal_client = fal_client.SyncClient(key=resolved_key)

    def generate(
        self,
        svp: SVPVideo,
        image_path: Path,
        output_path: Path,
        resolution: VideoResolution = "720p",
    ) -> VideoResult:
        if resolution not in ("480p", "720p"):
            raise ValueError(f"unsupported video resolution: {resolution!r}")
        if not image_path.exists():
            raise ValueError(f"reference image not found: {image_path}")

        started = time.perf_counter()
        image_url = self._upload_image(image_path=image_path)
        prompt = render_motion_prompt(svp)
        arguments = self._build_arguments(
            svp=svp,
            image_url=image_url,
            resolution=resolution,
            prompt=prompt,
        )
        endpoint = self.ENDPOINT_FAST if self.tier == "fast" else self.ENDPOINT_STANDARD

        result = self._subscribe_with_retry(endpoint=endpoint, arguments=arguments)
        mp4_url = self._extract_mp4_url(result)
        fal_request_id = self._extract_request_id(result)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._download_mp4(mp4_url=mp4_url, output_path=output_path)

        elapsed_sec = time.perf_counter() - started
        cost_usd = self._calculate_cost(
            tier=self.tier,
            resolution=resolution,
            duration_seconds=svp.motion_layer.duration_seconds,
        )
        return VideoResult(
            mp4_path=output_path,
            cost_usd=cost_usd,
            elapsed_sec=elapsed_sec,
            raw_prompt=prompt,
            tier=self.tier,
            resolution=resolution,
            aspect_ratio=svp.composition_layer.aspect_ratio,
            duration_seconds=svp.motion_layer.duration_seconds,
            fal_request_id=fal_request_id,
            mp4_url=mp4_url,
        )

    def _upload_image(self, image_path: Path) -> str:
        try:
            uploaded = self._fal_client.upload_file(str(image_path))
        except Exception as exc:  # noqa: BLE001
            raise VideoAPIError("failed to upload reference image to fal.ai") from exc

        if isinstance(uploaded, str):
            return uploaded
        if isinstance(uploaded, dict):
            maybe_url = uploaded.get("url")
            if isinstance(maybe_url, str) and maybe_url:
                return maybe_url
        raise VideoAPIError("fal_client.upload_file returned unsupported payload")

    def _build_arguments(
        self,
        svp: SVPVideo,
        image_url: str,
        resolution: VideoResolution,
        prompt: str,
    ) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "image_urls": [image_url],
            "duration": str(svp.motion_layer.duration_seconds),
            "aspect_ratio": svp.composition_layer.aspect_ratio,
            "resolution": resolution,
            "generate_audio": False,
        }

    def _download_mp4(self, mp4_url: str, output_path: Path) -> None:
        try:
            response = httpx.get(mp4_url, timeout=60.0, follow_redirects=True)
            response.raise_for_status()
            output_path.write_bytes(response.content)
        except Exception as exc:  # noqa: BLE001
            raise VideoDownloadError("failed to download generated mp4", mp4_url=mp4_url) from exc

    def _subscribe_with_retry(self, endpoint: str, arguments: dict[str, Any]) -> dict[str, Any]:
        backoff = 1.0
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return self._subscribe_with_timeout(endpoint=endpoint, arguments=arguments)
            except VideoTimeoutError:
                raise
            except Exception as exc:  # noqa: BLE001
                is_last_attempt = attempt >= self.MAX_RETRIES
                if is_last_attempt or not self._is_retryable_exception(exc):
                    raise VideoAPIError("seedance reference-to-video call failed") from exc
                time.sleep(backoff)
                backoff *= 2.0
        raise VideoAPIError("seedance reference-to-video call failed")

    def _subscribe_with_timeout(self, endpoint: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.timeout_mode == "thread":
            return self._subscribe_with_timeout_thread(endpoint=endpoint, arguments=arguments)
        return self._subscribe_with_timeout_process(endpoint=endpoint, arguments=arguments)

    def _subscribe_with_timeout_thread(
        self,
        endpoint: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Timeout mode used in tests where monkeypatching fal_client is required."""
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self._fal_client.subscribe,
            endpoint,
            arguments=arguments,
            with_logs=True,
        )
        try:
            result = future.result(timeout=self.TIMEOUT_SEC)
        except FuturesTimeoutError as exc:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise VideoTimeoutError(
                f"seedance request exceeded timeout ({self.TIMEOUT_SEC}s)"
            ) from exc
        except Exception:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True, cancel_futures=False)

        if not isinstance(result, dict):
            raise VideoAPIError("seedance response is not a dictionary")
        return result

    def _subscribe_with_timeout_process(
        self,
        endpoint: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Hard timeout mode for production path."""
        result_queue: Queue = Queue(maxsize=1)
        process = Process(
            target=_subscribe_worker,
            args=(endpoint, arguments, self._api_key, result_queue),
            daemon=True,
        )
        process.start()
        try:
            payload = result_queue.get(timeout=self.TIMEOUT_SEC)
        except Empty as exc:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                raise VideoTimeoutError(
                    f"seedance request exceeded timeout ({self.TIMEOUT_SEC}s)"
                ) from exc
            process.join(timeout=1)
            raise VideoAPIError("seedance worker returned no payload") from exc

        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

        if not isinstance(payload, dict):
            raise VideoAPIError("seedance worker returned invalid payload")

        if not payload.get("ok"):
            raise RuntimeError(payload.get("error", "seedance worker failed"))

        result = payload.get("result")
        if not isinstance(result, dict):
            raise VideoAPIError("seedance response is not a dictionary")
        return result

    @staticmethod
    def _extract_mp4_url(result: dict[str, Any]) -> str:
        video = result.get("video")
        if isinstance(video, dict):
            maybe_url = video.get("url")
            if isinstance(maybe_url, str) and maybe_url:
                return maybe_url
        raise VideoAPIError("seedance response missing video.url")

    @staticmethod
    def _extract_request_id(result: dict[str, Any]) -> str | None:
        for key in ("request_id", "requestId", "id"):
            value = result.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        if isinstance(exc, (ConnectionError, TimeoutError, OSError, httpx.TransportError)):
            return True
        message = str(exc).lower()
        return any(token in message for token in ("timeout", "temporar", "connection reset"))

    def _calculate_cost(
        self,
        tier: VideoTier,
        resolution: VideoResolution,
        duration_seconds: int,
    ) -> float:
        rate = self.PRICE_PER_SECOND[(tier, resolution)]
        return rate * duration_seconds
