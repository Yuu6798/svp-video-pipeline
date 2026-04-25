"""Command line interface for SVP Video Pipeline."""

# ruff: noqa: B008

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console

from . import __version__
from .exceptions import (
    ImageAPIError,
    ImageRefusalError,
    PlannerAPIError,
    PlannerSchemaError,
    SVPPipelineError,
    VideoAPIError,
    VideoDownloadError,
    VideoTimeoutError,
)
from .pipeline import Pipeline, PipelineResult
from .utils.logging import setup_verbose_logger

app = typer.Typer(
    name="svp-video",
    help="Generate SVP videos from natural language prompts.",
    add_completion=False,
)
console = Console()

PLANNER_MODELS = {"claude-opus-4-7", "claude-haiku-4-5"}
IMAGE_BACKENDS = {"gemini", "openai"}


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"svp-video {__version__}")
        raise typer.Exit()


@app.command()
def main(
    prompt: str = typer.Argument(..., help="Video generation prompt."),
    duration: int = typer.Option(5, "--duration", min=4, max=15, help="Duration in seconds."),
    output: Path | None = typer.Option(None, "--output", help="Output directory."),
    planner_model: str | None = typer.Option(None, "--planner-model", help="Claude model."),
    image_backend: str | None = typer.Option(
        None,
        "--image-backend",
        help="Image backend: gemini or openai.",
    ),
    cheap: bool = typer.Option(False, "--cheap", help="Use low-cost generation settings."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate SVP only."),
    no_video: bool = typer.Option(False, "--no-video", help="Generate image only, no video."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print verbose JSON logs."),
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit.",
        is_eager=True,
        callback=_version_callback,
    ),
) -> None:
    """Generate a video from a natural language prompt."""
    load_dotenv()
    output_dir = output or Path(os.environ.get("DEFAULT_OUTPUT_DIR", "./out"))
    planner_model = planner_model or os.environ.get("DEFAULT_PLANNER_MODEL", "claude-opus-4-7")
    image_backend = image_backend or os.environ.get("DEFAULT_IMAGE_BACKEND", "gemini")

    _validate_choice("planner model", planner_model, PLANNER_MODELS)
    _validate_choice("image backend", image_backend, IMAGE_BACKENDS)
    _check_api_keys(image_backend=image_backend, require_video=not (no_video or dry_run))
    _check_output_dir(output_dir)

    logger = setup_verbose_logger() if verbose else None
    try:
        pipeline = Pipeline(
            output_dir=output_dir,
            planner_model=planner_model,  # type: ignore[arg-type]
            image_backend=image_backend,
            cheap_mode=cheap,
            dry_run=dry_run,
        )
        result = _run_with_progress(
            pipeline=pipeline,
            prompt=prompt,
            duration=duration,
            no_video=no_video,
            verbose=verbose,
        )
        if logger is not None:
            logger.info("pipeline_completed", extra={"extra_data": _result_payload(result)})
        _print_summary(result, dry_run=dry_run, no_video=no_video)
    except (SVPPipelineError, ValueError) as exc:
        _handle_error(exc, verbose=verbose)
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted by user[/yellow]")
        raise typer.Exit(130) from None


def _validate_choice(label: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        console.print(f"[red]Invalid {label}: {value!r}. Choose one of: {allowed_text}[/red]")
        raise typer.Exit(2)


def _check_api_keys(image_backend: str, require_video: bool) -> None:
    missing: list[str] = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if image_backend == "gemini" and not (
        os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    ):
        missing.append("GOOGLE_API_KEY or GEMINI_API_KEY")
    if image_backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if require_video and not os.getenv("FAL_KEY"):
        missing.append("FAL_KEY")

    if missing:
        console.print("[red]Missing required environment variables:[/red]")
        for name in missing:
            console.print(f"  - {name}")
        console.print("Set them in .env or export them in your shell.")
        raise typer.Exit(1)


def _check_output_dir(output_dir: Path) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".svp_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        console.print(f"[red]Cannot write to output directory: {output_dir}[/red]")
        raise typer.Exit(1) from exc


def _run_with_progress(
    pipeline: Pipeline,
    prompt: str,
    duration: int,
    no_video: bool,
    verbose: bool,
) -> PipelineResult:
    logger = setup_verbose_logger() if verbose else None

    def _progress(event: str, payload: dict[str, Any]) -> None:
        _print_progress_event(event, payload)
        if logger is not None:
            logger.info(event, extra={"extra_data": payload})

    with console.status("[bold]Running SVP pipeline...[/bold]"):
        return pipeline.run(
            user_prompt=prompt,
            duration=duration,
            no_video=no_video,
            progress_callback=_progress,
        )


def _print_progress_event(event: str, payload: dict[str, Any]) -> None:
    if event == "planner_start":
        console.print(f"[1/3] Generating SVP with Claude ({payload['model']})...")
    elif event == "planner_done":
        console.print(
            f"[1/3] SVP generated ({payload['elapsed_sec']:.1f}s, "
            f"${payload['cost_usd']:.3f})"
        )
    elif event == "image_start":
        console.print(f"[2/3] Generating image with {payload['backend']}...")
    elif event == "image_done":
        console.print(
            f"[2/3] Image generated ({payload['elapsed_sec']:.1f}s, "
            f"${payload['cost_usd']:.3f})"
        )
    elif event == "video_start":
        console.print(
            "[3/3] Generating video with Seedance 2.0 "
            f"({payload['tier']}, {payload['resolution']}, {payload['duration']}s)..."
        )
        console.print("      This may take 1-3 minutes.")
    elif event == "video_done":
        console.print(
            f"[3/3] Video generated ({payload['elapsed_sec']:.1f}s, "
            f"${payload['cost_usd']:.3f})"
        )


def _print_summary(result: PipelineResult, dry_run: bool, no_video: bool) -> None:
    log_data = json.loads(result.log_path.read_text(encoding="utf-8"))
    if dry_run:
        console.print("\n--dry-run: skipping image and video generation.")
        console.print("\nEstimated cost if run normally:")
        _print_cost_lines(log_data)
        console.print(f"\nSVP saved to: {result.svp_path}")
        return

    if no_video:
        console.print("\n--no-video: skipping video generation.")

    console.print(f"\nTotal cost: ${result.total_cost_usd:.4f}")
    console.print(f"Total time: {result.total_elapsed_sec:.1f}s")
    console.print(f"\nOutputs saved to: {result.output_dir}")
    for path in (result.svp_path, result.image_path, result.video_path, result.log_path):
        if path is not None and path.exists():
            console.print(f"  {path.name:<10} {_format_size(path.stat().st_size):>9}")


def _print_cost_lines(log_data: dict[str, Any]) -> None:
    stages = log_data["stages"]
    planner_cost = stages["planner"].get("cost_usd", 0.0)
    image_cost = stages["image"].get("estimated_cost_usd", stages["image"].get("cost_usd", 0.0))
    video = stages.get("video")
    video_cost = 0.0
    if isinstance(video, dict):
        video_cost = video.get("estimated_cost_usd", video.get("cost_usd", 0.0))
    console.print(f"  Planner: ${planner_cost:.3f}")
    console.print(f"  Image:   ${image_cost:.3f}")
    console.print(f"  Video:   ${video_cost:.3f}")
    console.print(f"  Total:   ${log_data['total_cost_usd']:.3f}")


def _format_size(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def _handle_error(error: Exception, verbose: bool) -> None:
    if verbose:
        traceback.print_exc()

    if isinstance(error, PlannerAPIError):
        message = "Claude API error. Check ANTHROPIC_API_KEY."
    elif isinstance(error, PlannerSchemaError):
        message = "Claude output did not match the SVP schema. Try a more specific prompt."
    elif isinstance(error, ImageRefusalError):
        message = "Image generation was refused. Check the SVP forbidden constraints."
    elif isinstance(error, ImageAPIError):
        message = f"Image API error: {error}"
    elif isinstance(error, VideoTimeoutError):
        message = "Video generation timed out. Retry with --cheap or shorter --duration."
    elif isinstance(error, VideoDownloadError):
        message = f"Video URL was generated but download failed. URL: {error.mp4_url}"
    elif isinstance(error, VideoAPIError):
        message = f"Video API error: {error}"
    else:
        message = str(error)

    console.print(f"[red]{message}[/red]")


def _result_payload(result: PipelineResult) -> dict[str, Any]:
    return {
        "output_dir": str(result.output_dir),
        "svp_path": str(result.svp_path),
        "image_path": str(result.image_path) if result.image_path else None,
        "video_path": str(result.video_path) if result.video_path else None,
        "log_path": str(result.log_path),
        "total_cost_usd": result.total_cost_usd,
        "total_elapsed_sec": result.total_elapsed_sec,
    }
