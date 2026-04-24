"""Custom exception hierarchy for svp_pipeline."""

from __future__ import annotations


class SVPPipelineError(Exception):
    """Base exception for svp_pipeline."""


class PlannerError(SVPPipelineError):
    """Base exception for planner-related failures."""


class PlannerSchemaError(PlannerError):
    """Raised when planner output does not satisfy SVP schema."""


class PlannerAPIError(PlannerError):
    """Raised when Anthropic API calls fail in planner."""


class ImageGenerationError(SVPPipelineError):
    """Image-layer failure."""


class ImageAPIError(ImageGenerationError):
    """Raised when Gemini API calls fail."""


class ImageRefusalError(ImageGenerationError):
    """Raised when the image model refuses generation."""


class VideoGenerationError(SVPPipelineError):
    """Video-layer failure."""


class VideoAPIError(VideoGenerationError):
    """Raised when Seedance API calls fail."""


class VideoTimeoutError(VideoGenerationError):
    """Raised when video generation exceeds timeout."""


class VideoDownloadError(VideoGenerationError):
    """Raised when generated MP4 cannot be downloaded."""

    def __init__(self, message: str, *, mp4_url: str | None = None) -> None:
        super().__init__(message)
        self.mp4_url = mp4_url
