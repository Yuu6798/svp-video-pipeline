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
