"""Command line interface for SVP Video Pipeline."""

# ruff: noqa: B008

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass
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
from .semantic.failure_presets import (
    extract_failure_preset_candidate,
    write_failure_preset_candidate,
)
from .semantic.image_audit import ImageAuditResult, audit_image_to_observed_rpe
from .semantic.repair import RepairResult, repair_svp_from_observed_rpe
from .semantic.visual_compare import (
    VisualComparisonResult,
    validate_image_file,
    write_visual_comparison,
)
from .utils.logging import setup_verbose_logger

app = typer.Typer(
    name="svp-video",
    help="Generate SVP videos from natural language prompts.",
    add_completion=False,
)
console = Console()

PLANNER_MODELS = {"claude-opus-4-7", "claude-haiku-4-5"}
IMAGE_BACKENDS = {"gemini", "openai"}
IMAGE_AUDIT_BACKENDS = {"manual", "openai"}
REUSABLE_IMAGE_ARTIFACTS: dict[str, str] = {
    "image": "image.png",
    "composite": "composite.png",
    "character_green": "character_green.png",
    "background_clean": "background_clean.png",
}


@dataclass
class AuditWorkflowConfig:
    prompt_text: str
    duration: int
    output_dir: Path
    planner_model: str
    image_backend: str
    audit_image: Path
    compare_image: Path | None
    audit_backend: str
    audit_observed: list[str]
    audit_missing: list[str]
    audit_violation: list[str]
    audit_object_state: list[str]
    audit_contact_graph: list[str]
    audit_viewer_contact_graph: list[str]
    audit_anatomical_contact_graph: list[str]
    audit_pose_intent: str | None
    audit_failure_mode: list[str]
    audit_note: list[str]
    audit_repair: bool
    extract_failure_preset: bool
    audit_regenerate: bool
    from_svp: Path | None
    reuse_run: Path | None
    repair_from_rpe: Path | None
    reference_image: Path | None
    reference_crop: int | None
    separate_character_bg: bool
    failure_preset: list[str]
    cheap: bool
    dry_run: bool
    no_video: bool
    character_lock: bool
    verbose: bool


@dataclass
class AuditWorkflowResults:
    audit_result: ImageAuditResult
    repair_result: RepairResult | None = None
    failure_preset_candidate_path: Path | None = None
    comparison_result: VisualComparisonResult | None = None
    regeneration_result: PipelineResult | None = None
    regeneration_comparison_result: VisualComparisonResult | None = None


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"svp-video {__version__}")
        raise typer.Exit()


@app.command()
def main(
    prompt: list[str] | None = typer.Argument(None, help="Video generation prompt."),
    duration: int = typer.Option(5, "--duration", min=4, max=15, help="Duration in seconds."),
    output: Path | None = typer.Option(None, "--output", help="Output directory."),
    planner_model: str | None = typer.Option(None, "--planner-model", help="Claude model."),
    image_backend: str | None = typer.Option(
        None,
        "--image-backend",
        help="Image backend: gemini or openai.",
    ),
    reference_image: Path | None = typer.Option(
        None,
        "--reference-image",
        help="Optional reference image used by the image backend.",
    ),
    reference_crop: int | None = typer.Option(
        None,
        "--reference-crop",
        min=1,
        max=9,
        help="Crop a 3x3 reference sheet to one panel before image generation.",
    ),
    separate_character_bg: bool = typer.Option(
        False,
        "--separate-character-bg",
        help="Generate character and background separately, then composite before video.",
    ),
    from_svp: Path | None = typer.Option(
        None,
        "--from-svp",
        help="Use an existing SVP JSON file and skip Claude planning.",
    ),
    reuse_run: Path | None = typer.Option(
        None,
        "--reuse-run",
        help="Reuse svp.json and an image artifact from an existing output run.",
    ),
    reuse_image: str | None = typer.Option(
        None,
        "--reuse-image",
        help=(
            "Artifact from --reuse-run to reuse: image, composite, "
            "character_green, background_clean."
        ),
    ),
    repair_from_rpe: Path | None = typer.Option(
        None,
        "--repair-from-rpe",
        help="Observed RPE JSON used to write semantic_diff and target_svp.proposed.json.",
    ),
    failure_preset: list[str] | None = typer.Option(
        None,
        "--failure-preset",
        help="Failure preset id or JSON path to apply before image generation. Repeatable.",
    ),
    audit_image: Path | None = typer.Option(
        None,
        "--audit-image",
        help="Audit an image artifact into observed_rpe.json.",
    ),
    compare_image: Path | None = typer.Option(
        None,
        "--compare-image",
        help="Candidate image to compare against --audit-image inside the audit packet.",
    ),
    audit_backend: str = typer.Option(
        "manual",
        "--audit-backend",
        help="Image audit backend: manual or openai.",
    ),
    audit_observed: list[str] | None = typer.Option(
        None,
        "--audit-observed",
        help="Observed image state proposition. Repeat for multiple entries.",
    ),
    audit_missing: list[str] | None = typer.Option(
        None,
        "--audit-missing",
        help="Missing expected image state proposition. Repeat for multiple entries.",
    ),
    audit_violation: list[str] | None = typer.Option(
        None,
        "--audit-violation",
        help="Forbidden image state observed in the artifact. Repeat for multiple entries.",
    ),
    audit_object_state: list[str] | None = typer.Option(
        None,
        "--audit-object-state",
        help="Observed object role/state, e.g. 'scabbard: separate dark sheath'. Repeatable.",
    ),
    audit_contact_graph: list[str] | None = typer.Option(
        None,
        "--audit-contact-graph",
        help="Observed hand/object contact relation, e.g. 'left_hand -> scabbard'. Repeatable.",
    ),
    audit_viewer_contact_graph: list[str] | None = typer.Option(
        None,
        "--audit-viewer-contact-graph",
        help=(
            "Viewer-relative contact relation, e.g. "
            "'viewer_left_hand -> katana_handle'. Repeatable."
        ),
    ),
    audit_anatomical_contact_graph: list[str] | None = typer.Option(
        None,
        "--audit-anatomical-contact-graph",
        help=(
            "Character-body-side contact relation, e.g. "
            "'character_right_hand -> katana_handle'. Repeatable."
        ),
    ),
    audit_pose_intent: str | None = typer.Option(
        None,
        "--audit-pose-intent",
        help="Observed or desired pose intent, e.g. 'unsheathing / draw-pose'.",
    ),
    audit_failure_mode: list[str] | None = typer.Option(
        None,
        "--audit-failure-mode",
        help="Observed contact/object failure mode. Repeat for multiple entries.",
    ),
    audit_note: list[str] | None = typer.Option(
        None,
        "--audit-note",
        help="Additional audit note. Repeat for multiple entries.",
    ),
    audit_repair: bool = typer.Option(
        False,
        "--audit-repair",
        help="After --audit-image, also write semantic diff and target_svp.proposed.json.",
    ),
    extract_failure_preset: bool = typer.Option(
        False,
        "--extract-failure-preset",
        help="After --audit-repair, also write failure_preset.candidate.json.",
    ),
    audit_regenerate: bool = typer.Option(
        False,
        "--audit-regenerate",
        help=(
            "After --audit-repair, regenerate from target_svp.proposed.json. "
            "When --compare-image is provided, compare the regenerated image to it."
        ),
    ),
    cheap: bool = typer.Option(False, "--cheap", help="Use low-cost generation settings."),
    character_lock: bool = typer.Option(
        True,
        "--character-lock/--no-character-lock",
        help="Preserve literal character traits during SVP planning.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate SVP only."),
    no_video: bool = typer.Option(False, "--no-video", help="Generate image only, no video."),
    archive_drive: bool = typer.Option(
        False,
        "--archive-drive",
        help="Archive generated image/video artifacts to Google Drive after success.",
    ),
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
    prompt_text = _resolve_prompt_text(
        prompt=prompt,
        from_svp=from_svp,
        reuse_run=reuse_run,
        repair_from_rpe=repair_from_rpe,
        audit_image=audit_image,
    )
    output_dir = output or Path(os.environ.get("DEFAULT_OUTPUT_DIR", "./out"))
    planner_model = planner_model or os.environ.get("DEFAULT_PLANNER_MODEL", "claude-opus-4-7")
    image_backend = image_backend or os.environ.get("DEFAULT_IMAGE_BACKEND", "gemini")

    _validate_choice("image audit backend", audit_backend, IMAGE_AUDIT_BACKENDS)
    _check_audit_flags_require_image(
        audit_image=audit_image,
        compare_image=compare_image,
        audit_observed=audit_observed,
        audit_missing=audit_missing,
        audit_violation=audit_violation,
        audit_object_state=audit_object_state,
        audit_contact_graph=audit_contact_graph,
        audit_viewer_contact_graph=audit_viewer_contact_graph,
        audit_anatomical_contact_graph=audit_anatomical_contact_graph,
        audit_pose_intent=audit_pose_intent,
        audit_failure_mode=audit_failure_mode,
        audit_note=audit_note,
        audit_repair=audit_repair,
        audit_regenerate=audit_regenerate,
        extract_failure_preset=extract_failure_preset,
    )
    if audit_image is not None:
        _run_audit_workflow(
            prompt_text=prompt_text,
            duration=duration,
            output_dir=output_dir,
            planner_model=planner_model,
            image_backend=image_backend,
            audit_image=audit_image,
            compare_image=compare_image,
            audit_backend=audit_backend,
            audit_observed=audit_observed or [],
            audit_missing=audit_missing or [],
            audit_violation=audit_violation or [],
            audit_object_state=audit_object_state or [],
            audit_contact_graph=audit_contact_graph or [],
            audit_viewer_contact_graph=audit_viewer_contact_graph or [],
            audit_anatomical_contact_graph=audit_anatomical_contact_graph or [],
            audit_pose_intent=audit_pose_intent,
            audit_failure_mode=audit_failure_mode or [],
            audit_note=audit_note or [],
            audit_repair=audit_repair,
            extract_failure_preset=extract_failure_preset,
            audit_regenerate=audit_regenerate,
            from_svp=from_svp,
            reuse_run=reuse_run,
            repair_from_rpe=repair_from_rpe,
            reference_image=reference_image,
            reference_crop=reference_crop,
            separate_character_bg=separate_character_bg,
            failure_preset=failure_preset or [],
            cheap=cheap,
            dry_run=dry_run,
            no_video=no_video,
            character_lock=character_lock,
            verbose=verbose,
        )
        return
    if repair_from_rpe is not None:
        _run_repair_workflow(
            repair_from_rpe=repair_from_rpe,
            from_svp=from_svp,
            reuse_run=reuse_run,
            output_dir=output_dir,
            verbose=verbose,
        )
        return

    _run_generation_workflow(
        prompt_text=prompt_text,
        duration=duration,
        output_dir=output_dir,
        planner_model=planner_model,
        image_backend=image_backend,
        reference_image=reference_image,
        reference_crop=reference_crop,
        separate_character_bg=separate_character_bg,
        from_svp=from_svp,
        reuse_run=reuse_run,
        reuse_image=reuse_image,
        failure_preset=failure_preset or [],
        cheap=cheap,
        character_lock=character_lock,
        dry_run=dry_run,
        no_video=no_video,
        archive_drive=archive_drive,
        verbose=verbose,
    )


def _resolve_prompt_text(
    prompt: list[str] | None,
    from_svp: Path | None,
    reuse_run: Path | None,
    repair_from_rpe: Path | None,
    audit_image: Path | None,
) -> str:
    prompt_text = " ".join(prompt or []).strip()
    if (
        not prompt_text
        and from_svp is None
        and reuse_run is None
        and repair_from_rpe is None
        and audit_image is None
    ):
        console.print("[red]Prompt must not be empty.[/red]")
        raise typer.Exit(2)
    if prompt_text:
        return prompt_text
    return _prompt_placeholder(from_svp=from_svp, reuse_run=reuse_run)


def _run_audit_workflow(
    prompt_text: str,
    duration: int,
    output_dir: Path,
    planner_model: str,
    image_backend: str,
    audit_image: Path,
    compare_image: Path | None,
    audit_backend: str,
    audit_observed: list[str],
    audit_missing: list[str],
    audit_violation: list[str],
    audit_object_state: list[str],
    audit_contact_graph: list[str],
    audit_viewer_contact_graph: list[str],
    audit_anatomical_contact_graph: list[str],
    audit_pose_intent: str | None,
    audit_failure_mode: list[str],
    audit_note: list[str],
    audit_repair: bool,
    extract_failure_preset: bool,
    audit_regenerate: bool,
    from_svp: Path | None,
    reuse_run: Path | None,
    repair_from_rpe: Path | None,
    reference_image: Path | None,
    reference_crop: int | None,
    separate_character_bg: bool,
    failure_preset: list[str],
    cheap: bool,
    dry_run: bool,
    no_video: bool,
    character_lock: bool,
    verbose: bool,
) -> None:
    config = AuditWorkflowConfig(
        prompt_text=prompt_text,
        duration=duration,
        output_dir=output_dir,
        planner_model=planner_model,
        image_backend=image_backend,
        audit_image=audit_image,
        compare_image=compare_image,
        audit_backend=audit_backend,
        audit_observed=audit_observed,
        audit_missing=audit_missing,
        audit_violation=audit_violation,
        audit_object_state=audit_object_state,
        audit_contact_graph=audit_contact_graph,
        audit_viewer_contact_graph=audit_viewer_contact_graph,
        audit_anatomical_contact_graph=audit_anatomical_contact_graph,
        audit_pose_intent=audit_pose_intent,
        audit_failure_mode=audit_failure_mode,
        audit_note=audit_note,
        audit_repair=audit_repair,
        extract_failure_preset=extract_failure_preset,
        audit_regenerate=audit_regenerate,
        from_svp=from_svp,
        reuse_run=reuse_run,
        repair_from_rpe=repair_from_rpe,
        reference_image=reference_image,
        reference_crop=reference_crop,
        separate_character_bg=separate_character_bg,
        failure_preset=failure_preset,
        cheap=cheap,
        dry_run=dry_run,
        no_video=no_video,
        character_lock=character_lock,
        verbose=verbose,
    )
    _validate_audit_workflow_options(config)
    _check_output_dir(config.output_dir)
    try:
        results = _execute_audit_workflow(config)
    except (SVPPipelineError, ValueError, RuntimeError, OSError) as exc:
        _handle_error(exc, verbose=config.verbose)
        raise typer.Exit(1) from exc
    _print_audit_summary(
        results.audit_result,
        repair_result=results.repair_result,
        failure_preset_candidate_path=results.failure_preset_candidate_path,
        comparison_result=results.comparison_result,
        regeneration_result=results.regeneration_result,
        regeneration_comparison_result=results.regeneration_comparison_result,
    )


def _execute_audit_workflow(config: AuditWorkflowConfig) -> AuditWorkflowResults:
    audit_result = _run_image_audit(
        image_path=config.audit_image,
        backend=config.audit_backend,
        from_svp=config.from_svp,
        reuse_run=config.reuse_run,
        output_dir=config.output_dir,
        observed=config.audit_observed,
        missing=config.audit_missing,
        violations=config.audit_violation,
        object_states=config.audit_object_state,
        contact_graph=config.audit_contact_graph,
        viewer_contact_graph=config.audit_viewer_contact_graph,
        anatomical_contact_graph=config.audit_anatomical_contact_graph,
        pose_intent=config.audit_pose_intent,
        failure_modes=config.audit_failure_mode,
        notes=config.audit_note,
    )
    comparison_result = _maybe_run_audit_visual_comparison(
        audit_result=audit_result,
        compare_image=config.compare_image,
        notes=config.audit_note,
    )
    repair_result, failure_preset_candidate_path = _maybe_run_audit_repair(
        audit_result=audit_result,
        audit_repair=config.audit_repair,
        extract_failure_preset=config.extract_failure_preset,
        from_svp=config.from_svp,
        reuse_run=config.reuse_run,
    )
    regeneration_result = _maybe_run_repair_regeneration(
        audit_regenerate=config.audit_regenerate,
        repair_result=repair_result,
        prompt_text=config.prompt_text,
        duration=config.duration,
        no_video=config.no_video,
        output_dir=audit_result.output_dir / "regenerated",
        planner_model=config.planner_model,
        image_backend=config.image_backend,
        cheap=config.cheap,
        dry_run=config.dry_run,
        character_lock=config.character_lock,
        reference_image=config.reference_image,
        reference_crop=config.reference_crop,
        separate_character_bg=config.separate_character_bg,
        failure_preset=config.failure_preset,
        verbose=config.verbose,
    )
    regeneration_comparison_result = _maybe_run_regeneration_comparison(
        audit_result=audit_result,
        compare_image=config.compare_image,
        regeneration_result=regeneration_result,
        notes=config.audit_note,
    )
    return AuditWorkflowResults(
        audit_result=audit_result,
        repair_result=repair_result,
        failure_preset_candidate_path=failure_preset_candidate_path,
        comparison_result=comparison_result,
        regeneration_result=regeneration_result,
        regeneration_comparison_result=regeneration_comparison_result,
    )


def _maybe_run_audit_repair(
    *,
    audit_result: ImageAuditResult,
    audit_repair: bool,
    extract_failure_preset: bool,
    from_svp: Path | None,
    reuse_run: Path | None,
) -> tuple[RepairResult | None, Path | None]:
    if not audit_repair:
        return None, None
    repair_result = _run_repair_from_rpe(
        observed_rpe=audit_result.observed_rpe_path,
        from_svp=from_svp,
        reuse_run=reuse_run,
        output_dir=audit_result.output_dir,
    )
    if not extract_failure_preset:
        return repair_result, None
    return repair_result, _write_failure_preset_candidate(
        audit_result=audit_result,
        repair_result=repair_result,
    )


def _maybe_run_repair_regeneration(
    *,
    audit_regenerate: bool,
    repair_result: RepairResult | None,
    prompt_text: str,
    duration: int,
    no_video: bool,
    output_dir: Path,
    planner_model: str,
    image_backend: str,
    cheap: bool,
    dry_run: bool,
    character_lock: bool,
    reference_image: Path | None,
    reference_crop: int | None,
    separate_character_bg: bool,
    failure_preset: list[str],
    verbose: bool,
) -> PipelineResult | None:
    if not audit_regenerate:
        return None
    assert repair_result is not None
    return _run_regeneration_from_repair(
        repair_result=repair_result,
        prompt=prompt_text,
        duration=duration,
        no_video=no_video,
        output_dir=output_dir,
        planner_model=planner_model,
        image_backend=image_backend,
        cheap=cheap,
        dry_run=dry_run,
        character_lock=character_lock,
        reference_image=reference_image,
        reference_crop=reference_crop,
        separate_character_bg=separate_character_bg,
        failure_presets=failure_preset,
        verbose=verbose,
    )


def _maybe_run_regeneration_comparison(
    *,
    audit_result: ImageAuditResult,
    compare_image: Path | None,
    regeneration_result: PipelineResult | None,
    notes: list[str],
) -> VisualComparisonResult | None:
    if (
        compare_image is None
        or regeneration_result is None
        or regeneration_result.image_path is None
    ):
        return None
    return _run_visual_comparison(
        source_image=compare_image,
        candidate_image=regeneration_result.image_path,
        output_dir=audit_result.output_dir / "regenerated_comparison",
        notes=[
            *(
                notes
                or [
                    "regenerated image compared against improvement target",
                ]
            ),
        ],
    )


def _validate_audit_workflow_options(config: AuditWorkflowConfig) -> None:
    if config.audit_regenerate and not config.audit_repair:
        console.print("[red]--audit-regenerate requires --audit-repair[/red]")
        raise typer.Exit(1)
    if config.extract_failure_preset and not config.audit_repair:
        console.print("[red]--extract-failure-preset requires --audit-repair[/red]")
        raise typer.Exit(1)
    _check_audit_options(
        image_path=config.audit_image,
        compare_image=config.compare_image,
        backend=config.audit_backend,
        from_svp=config.from_svp,
        reuse_run=config.reuse_run,
        repair_from_rpe=config.repair_from_rpe,
    )
    if config.audit_regenerate:
        _validate_regeneration_options(
            planner_model=config.planner_model,
            image_backend=config.image_backend,
            reference_image=config.reference_image,
            reference_crop=config.reference_crop,
            separate_character_bg=config.separate_character_bg,
            dry_run=config.dry_run,
            no_video=config.no_video,
        )


def _validate_regeneration_options(
    planner_model: str,
    image_backend: str,
    reference_image: Path | None,
    reference_crop: int | None,
    separate_character_bg: bool,
    dry_run: bool,
    no_video: bool,
) -> None:
    _validate_choice("planner model", planner_model, PLANNER_MODELS)
    _validate_choice("image backend", image_backend, IMAGE_BACKENDS)
    _check_reference_image(reference_image, reference_crop)
    _check_separate_character_bg(
        separate_character_bg=separate_character_bg,
        image_backend=image_backend,
        reference_image=reference_image,
    )
    _check_api_keys(
        image_backend=image_backend,
        require_planner=False,
        require_image=not dry_run,
        require_video=not (no_video or dry_run),
    )


def _maybe_run_audit_visual_comparison(
    audit_result: ImageAuditResult,
    compare_image: Path | None,
    notes: list[str],
) -> VisualComparisonResult | None:
    if compare_image is None:
        return None
    return _run_visual_comparison(
        source_image=audit_result.source_image_path,
        candidate_image=compare_image,
        output_dir=audit_result.output_dir,
        notes=notes,
    )


def _run_repair_workflow(
    repair_from_rpe: Path,
    from_svp: Path | None,
    reuse_run: Path | None,
    output_dir: Path,
    verbose: bool,
) -> None:
    _check_repair_options(
        observed_rpe=repair_from_rpe,
        from_svp=from_svp,
        reuse_run=reuse_run,
    )
    _check_output_dir(output_dir)
    try:
        repair_result = _run_repair_from_rpe(
            observed_rpe=repair_from_rpe,
            from_svp=from_svp,
            reuse_run=reuse_run,
            output_dir=output_dir,
        )
    except (SVPPipelineError, ValueError, RuntimeError, OSError) as exc:
        _handle_error(exc, verbose=verbose)
        raise typer.Exit(1) from exc
    _print_repair_summary(repair_result)


def _run_generation_workflow(
    prompt_text: str,
    duration: int,
    output_dir: Path,
    planner_model: str,
    image_backend: str,
    reference_image: Path | None,
    reference_crop: int | None,
    separate_character_bg: bool,
    from_svp: Path | None,
    reuse_run: Path | None,
    reuse_image: str | None,
    failure_preset: list[str],
    cheap: bool,
    character_lock: bool,
    dry_run: bool,
    no_video: bool,
    archive_drive: bool,
    verbose: bool,
) -> None:
    _validate_choice("planner model", planner_model, PLANNER_MODELS)
    _validate_choice("image backend", image_backend, IMAGE_BACKENDS)
    _check_reuse_options(
        from_svp=from_svp,
        reuse_run=reuse_run,
        reuse_image=reuse_image,
        reference_image=reference_image,
        reference_crop=reference_crop,
        separate_character_bg=separate_character_bg,
    )
    _check_reference_image(reference_image, reference_crop)
    _check_separate_character_bg(
        separate_character_bg=separate_character_bg,
        image_backend=image_backend,
        reference_image=reference_image,
    )
    _check_api_keys(
        image_backend=image_backend,
        require_planner=from_svp is None and reuse_run is None,
        require_image=reuse_run is None and not (dry_run and from_svp is not None),
        require_video=not (no_video or dry_run),
    )
    _check_output_dir(output_dir)

    logger = setup_verbose_logger() if verbose else None
    try:
        pipeline = Pipeline(
            output_dir=output_dir,
            planner_model=planner_model,  # type: ignore[arg-type]
            image_backend=image_backend,
            cheap_mode=cheap,
            dry_run=dry_run,
            character_lock=character_lock,
        )
        result = _run_with_progress(
            pipeline=pipeline,
            prompt=prompt_text,
            duration=duration,
            no_video=no_video,
            reference_image=reference_image,
            reference_crop=reference_crop,
            separate_character_bg=separate_character_bg,
            from_svp=from_svp,
            reuse_run=reuse_run,
            reuse_image=reuse_image,
            failure_presets=failure_preset,
            verbose=verbose,
        )
        if logger is not None:
            logger.info("pipeline_completed", extra={"extra_data": _result_payload(result)})
        _print_summary(result, dry_run=dry_run, no_video=no_video)
        if archive_drive:
            if dry_run:
                console.print("\n[yellow]--archive-drive skipped for --dry-run.[/yellow]")
            else:
                archive_result = _archive_outputs_to_drive(result)
                _print_archive_summary(archive_result)
    except (SVPPipelineError, ValueError, RuntimeError, FileNotFoundError) as exc:
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


def _check_api_keys(
    image_backend: str,
    require_planner: bool,
    require_image: bool,
    require_video: bool,
) -> None:
    missing: list[str] = []
    if require_planner and not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if (
        require_image
        and image_backend == "gemini"
        and not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    ):
        missing.append("GOOGLE_API_KEY or GEMINI_API_KEY")
    if require_image and image_backend == "openai" and not os.getenv("OPENAI_API_KEY"):
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


def _check_reference_image(reference_image: Path | None, reference_crop: int | None) -> None:
    if reference_image is None:
        if reference_crop is not None:
            console.print("[red]--reference-crop requires --reference-image[/red]")
            raise typer.Exit(1)
        return

    if not reference_image.exists():
        console.print(f"[red]Reference image not found: {reference_image}[/red]")
        raise typer.Exit(1)
    if not reference_image.is_file():
        console.print(f"[red]Reference image is not a file: {reference_image}[/red]")
        raise typer.Exit(1)


def _check_separate_character_bg(
    separate_character_bg: bool,
    image_backend: str,
    reference_image: Path | None,
) -> None:
    if not separate_character_bg:
        return
    if image_backend != "openai":
        console.print(
            "[red]--separate-character-bg currently requires --image-backend openai[/red]"
        )
        raise typer.Exit(1)
    if reference_image is None:
        console.print("[red]--separate-character-bg requires --reference-image[/red]")
        raise typer.Exit(1)


def _check_reuse_options(
    from_svp: Path | None,
    reuse_run: Path | None,
    reuse_image: str | None,
    reference_image: Path | None,
    reference_crop: int | None,
    separate_character_bg: bool,
) -> None:
    allowed_reuse_images = set(REUSABLE_IMAGE_ARTIFACTS)
    if from_svp is not None and not from_svp.is_file():
        console.print(f"[red]SVP file not found: {from_svp}[/red]")
        raise typer.Exit(1)
    if reuse_image is not None and reuse_image not in allowed_reuse_images:
        allowed = ", ".join(sorted(allowed_reuse_images))
        console.print(
            f"[red]Invalid --reuse-image: {reuse_image!r}. Choose one of: {allowed}[/red]"
        )
        raise typer.Exit(2)
    if reuse_image is not None and reuse_run is None:
        console.print("[red]--reuse-image requires --reuse-run[/red]")
        raise typer.Exit(1)
    if reuse_run is None:
        return
    if not reuse_run.is_dir():
        console.print(f"[red]Reuse run directory not found: {reuse_run}[/red]")
        raise typer.Exit(1)
    if not (reuse_run / "svp.json").is_file() and from_svp is None:
        console.print(f"[red]Reusable SVP not found: {reuse_run / 'svp.json'}[/red]")
        raise typer.Exit(1)
    artifact = REUSABLE_IMAGE_ARTIFACTS[reuse_image or "image"]
    if not (reuse_run / artifact).is_file():
        console.print(f"[red]Reusable image artifact not found: {reuse_run / artifact}[/red]")
        raise typer.Exit(1)
    if reference_image is not None or reference_crop is not None or separate_character_bg:
        console.print(
            "[red]--reuse-run reuses an existing image artifact; do not combine it with "
            "--reference-image, --reference-crop, or --separate-character-bg[/red]"
        )
        raise typer.Exit(1)


def _check_repair_options(
    observed_rpe: Path,
    from_svp: Path | None,
    reuse_run: Path | None,
) -> None:
    if not observed_rpe.is_file():
        console.print(f"[red]Observed RPE file not found: {observed_rpe}[/red]")
        raise typer.Exit(1)
    if from_svp is None and reuse_run is None:
        console.print("[red]--repair-from-rpe requires --from-svp or --reuse-run[/red]")
        raise typer.Exit(1)
    if from_svp is not None and not from_svp.is_file():
        console.print(f"[red]SVP file not found: {from_svp}[/red]")
        raise typer.Exit(1)
    if reuse_run is not None:
        if not reuse_run.is_dir():
            console.print(f"[red]Reuse run directory not found: {reuse_run}[/red]")
            raise typer.Exit(1)
        if from_svp is None and not (reuse_run / "svp.json").is_file():
            console.print(f"[red]Reusable SVP not found: {reuse_run / 'svp.json'}[/red]")
            raise typer.Exit(1)


def _check_audit_options(
    image_path: Path,
    compare_image: Path | None,
    backend: str,
    from_svp: Path | None,
    reuse_run: Path | None,
    repair_from_rpe: Path | None,
) -> None:
    if repair_from_rpe is not None:
        console.print("[red]Use either --audit-image or --repair-from-rpe, not both.[/red]")
        raise typer.Exit(1)
    if not image_path.is_file():
        console.print(f"[red]Audit image not found: {image_path}[/red]")
        raise typer.Exit(1)
    _check_readable_image(image_path, label="Audit image")
    if compare_image is not None and not compare_image.is_file():
        console.print(f"[red]Compare image not found: {compare_image}[/red]")
        raise typer.Exit(1)
    if compare_image is not None:
        _check_readable_image(compare_image, label="Compare image")
    if from_svp is None and reuse_run is None:
        console.print("[red]--audit-image requires --from-svp or --reuse-run[/red]")
        raise typer.Exit(1)
    if from_svp is not None and not from_svp.is_file():
        console.print(f"[red]SVP file not found: {from_svp}[/red]")
        raise typer.Exit(1)
    if reuse_run is not None:
        if not reuse_run.is_dir():
            console.print(f"[red]Reuse run directory not found: {reuse_run}[/red]")
            raise typer.Exit(1)
        if from_svp is None and not (reuse_run / "svp.json").is_file():
            console.print(f"[red]Reusable SVP not found: {reuse_run / 'svp.json'}[/red]")
            raise typer.Exit(1)
    if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY is required for --audit-backend openai[/red]")
        raise typer.Exit(1)


def _check_audit_flags_require_image(
    audit_image: Path | None,
    compare_image: Path | None,
    audit_observed: list[str] | None,
    audit_missing: list[str] | None,
    audit_violation: list[str] | None,
    audit_object_state: list[str] | None,
    audit_contact_graph: list[str] | None,
    audit_viewer_contact_graph: list[str] | None,
    audit_anatomical_contact_graph: list[str] | None,
    audit_pose_intent: str | None,
    audit_failure_mode: list[str] | None,
    audit_note: list[str] | None,
    audit_repair: bool,
    audit_regenerate: bool,
    extract_failure_preset: bool,
) -> None:
    if audit_image is not None:
        return
    if (
        any(
            (
                compare_image,
                audit_observed,
                audit_missing,
                audit_violation,
                audit_object_state,
                audit_contact_graph,
                audit_viewer_contact_graph,
                audit_anatomical_contact_graph,
                audit_pose_intent,
                audit_failure_mode,
                audit_note,
            )
        )
        or audit_repair
        or audit_regenerate
        or extract_failure_preset
    ):
        console.print("[red]Audit options require --audit-image[/red]")
        raise typer.Exit(1)


def _check_readable_image(path: Path, label: str) -> None:
    try:
        validate_image_file(path)
    except OSError as exc:
        console.print(f"[red]{label} is not a readable image: {path}[/red]")
        raise typer.Exit(1) from exc


def _prompt_placeholder(from_svp: Path | None, reuse_run: Path | None) -> str:
    if from_svp is not None:
        return f"regenerate from SVP: {from_svp}"
    if reuse_run is not None:
        return f"regenerate from run: {reuse_run}"
    return "regenerate"


def _run_repair_from_rpe(
    observed_rpe: Path,
    from_svp: Path | None,
    reuse_run: Path | None,
    output_dir: Path,
) -> RepairResult:
    if from_svp is not None:
        base_svp = from_svp
    else:
        if reuse_run is None:
            raise RuntimeError("either from_svp or reuse_run must be set")
        base_svp = reuse_run / "svp.json"
    return repair_svp_from_observed_rpe(
        base_svp_path=base_svp,
        observed_rpe_path=observed_rpe,
        output_root=output_dir,
    )


def _write_failure_preset_candidate(
    *,
    audit_result: ImageAuditResult,
    repair_result: RepairResult,
) -> Path:
    preset = extract_failure_preset_candidate(
        observed=audit_result.observed_rpe,
        diff=repair_result.diff,
        source={
            "observed_rpe": str(audit_result.observed_rpe_path),
            "semantic_diff": str(repair_result.semantic_diff_path),
            "proposed_svp": str(repair_result.proposed_svp_path),
        },
        preset_id="candidate",
    )
    candidate_path = repair_result.output_dir / "failure_preset.candidate.json"
    write_failure_preset_candidate(candidate_path, preset)
    return candidate_path


def _run_image_audit(
    image_path: Path,
    backend: str,
    from_svp: Path | None,
    reuse_run: Path | None,
    output_dir: Path,
    observed: list[str],
    missing: list[str],
    violations: list[str],
    object_states: list[str],
    contact_graph: list[str],
    viewer_contact_graph: list[str],
    anatomical_contact_graph: list[str],
    pose_intent: str | None,
    failure_modes: list[str],
    notes: list[str],
) -> ImageAuditResult:
    if from_svp is not None:
        base_svp = from_svp
    else:
        if reuse_run is None:
            raise RuntimeError("either from_svp or reuse_run must be set")
        base_svp = reuse_run / "svp.json"
    return audit_image_to_observed_rpe(
        svp_path=base_svp,
        image_path=image_path,
        output_root=output_dir,
        backend=backend,
        observed=observed,
        missing=missing,
        violations=violations,
        object_states=object_states,
        contact_graph=contact_graph,
        viewer_contact_graph=viewer_contact_graph,
        anatomical_contact_graph=anatomical_contact_graph,
        pose_intent=pose_intent,
        failure_modes=failure_modes,
        notes=notes,
    )


def _run_visual_comparison(
    source_image: Path,
    candidate_image: Path,
    output_dir: Path,
    notes: list[str],
) -> VisualComparisonResult:
    return write_visual_comparison(
        source_image_path=source_image,
        candidate_image_path=candidate_image,
        output_dir=output_dir,
        notes=notes,
    )


def _run_regeneration_from_repair(
    *,
    repair_result: RepairResult,
    prompt: str,
    duration: int,
    no_video: bool,
    output_dir: Path,
    planner_model: str,
    image_backend: str,
    cheap: bool,
    dry_run: bool,
    character_lock: bool,
    reference_image: Path | None,
    reference_crop: int | None,
    separate_character_bg: bool,
    failure_presets: list[str],
    verbose: bool,
) -> PipelineResult:
    pipeline = Pipeline(
        output_dir=output_dir,
        planner_model=planner_model,  # type: ignore[arg-type]
        image_backend=image_backend,
        cheap_mode=cheap,
        dry_run=dry_run,
        character_lock=character_lock,
    )
    return _run_with_progress(
        pipeline=pipeline,
        prompt=prompt or f"regenerate from repaired SVP: {repair_result.proposed_svp_path}",
        duration=duration,
        no_video=no_video,
        reference_image=reference_image,
        reference_crop=reference_crop,
        separate_character_bg=separate_character_bg,
        from_svp=repair_result.proposed_svp_path,
        reuse_run=None,
        reuse_image=None,
        failure_presets=failure_presets,
        verbose=verbose,
    )


def _print_audit_summary(
    result: ImageAuditResult,
    repair_result: RepairResult | None,
    failure_preset_candidate_path: Path | None,
    comparison_result: VisualComparisonResult | None,
    regeneration_result: PipelineResult | None = None,
    regeneration_comparison_result: VisualComparisonResult | None = None,
) -> None:
    console.print("\nImage RPE audit written.")
    console.print(f"Output dir: {result.output_dir}")
    console.print(f"Source image: {result.source_image_path}")
    console.print(f"Expected RPE: {result.expected_rpe_path}")
    console.print(f"Observed RPE: {result.observed_rpe_path}")
    console.print(f"Observed: {len(result.observed_rpe.observed)}")
    console.print(f"Missing: {len(result.observed_rpe.missing)}")
    console.print(f"Violations: {len(result.observed_rpe.violations)}")
    if repair_result is not None:
        console.print(f"Semantic diff: {repair_result.semantic_diff_path}")
        console.print(f"Repair proposal: {repair_result.proposal_path}")
        console.print(f"Proposed SVP: {repair_result.proposed_svp_path}")
        console.print(f"Gate status: {repair_result.diff.gate_status}")
        console.print(f"Issues: {len(repair_result.diff.issues)}")
    if failure_preset_candidate_path is not None:
        console.print(f"Failure preset candidate: {failure_preset_candidate_path}")
    if comparison_result is not None:
        console.print(f"Candidate image: {comparison_result.candidate_image_path}")
        console.print(f"Visual comparison: {comparison_result.comparison_path}")
    if regeneration_result is not None:
        console.print(f"Regenerated output dir: {regeneration_result.output_dir}")
        if regeneration_result.image_path is not None:
            console.print(f"Regenerated image: {regeneration_result.image_path}")
        if regeneration_result.video_path is not None:
            console.print(f"Regenerated video: {regeneration_result.video_path}")
        console.print(f"Regenerated log: {regeneration_result.log_path}")
    if regeneration_comparison_result is not None:
        console.print(
            f"Regenerated visual comparison: {regeneration_comparison_result.comparison_path}"
        )


def _print_repair_summary(result: RepairResult) -> None:
    console.print("\nSemantic repair proposal written.")
    console.print(f"Output dir: {result.output_dir}")
    console.print(f"Semantic diff: {result.semantic_diff_path}")
    console.print(f"Repair proposal: {result.proposal_path}")
    console.print(f"Proposed SVP: {result.proposed_svp_path}")
    console.print(f"Gate status: {result.diff.gate_status}")
    console.print(f"Issues: {len(result.diff.issues)}")


def _run_with_progress(
    pipeline: Pipeline,
    prompt: str,
    duration: int,
    no_video: bool,
    reference_image: Path | None,
    reference_crop: int | None,
    separate_character_bg: bool,
    from_svp: Path | None,
    reuse_run: Path | None,
    reuse_image: str | None,
    failure_presets: list[str],
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
            reference_image_path=reference_image,
            reference_crop=reference_crop,
            separate_character_bg=separate_character_bg,
            from_svp_path=from_svp,
            reuse_run_dir=reuse_run,
            reuse_image=reuse_image,
            failure_presets=failure_presets,
            progress_callback=_progress,
        )


def _print_progress_event(event: str, payload: dict[str, Any]) -> None:
    if event == "planner_start":
        if payload.get("reused"):
            console.print(f"[1/3] Reusing SVP ({payload['source']})...")
        else:
            console.print(f"[1/3] Generating SVP with Claude ({payload['model']})...")
    elif event == "planner_done":
        verb = "reused" if payload.get("reused") else "generated"
        console.print(
            f"[1/3] SVP {verb} ({payload['elapsed_sec']:.1f}s, ${payload['cost_usd']:.3f})"
        )
    elif event == "image_start":
        if payload.get("reused"):
            console.print(f"[2/3] Reusing image artifact ({payload['source']})...")
        else:
            console.print(f"[2/3] Generating image with {payload['backend']}...")
    elif event == "image_done":
        verb = "reused" if payload.get("reused") else "generated"
        console.print(
            f"[2/3] Image {verb} ({payload['elapsed_sec']:.1f}s, ${payload['cost_usd']:.3f})"
        )
    elif event == "video_start":
        console.print(
            "[3/3] Generating video with Seedance 2.0 "
            f"({payload['tier']}, {payload['resolution']}, {payload['duration']}s)..."
        )
        console.print("      This may take 1-3 minutes.")
    elif event == "video_done":
        console.print(
            f"[3/3] Video generated ({payload['elapsed_sec']:.1f}s, ${payload['cost_usd']:.3f})"
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


def _archive_outputs_to_drive(result: PipelineResult) -> Any:
    try:
        from .tools.archive_to_drive import archive_run
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Drive archive dependencies are missing. Install them with: pip install -e ".[drive]"'
        ) from exc

    console.print("\nArchiving outputs to Google Drive...")
    try:
        with console.status("[bold]Uploading artifacts to Google Drive...[/bold]"):
            return archive_run(result.output_dir)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Drive archive dependencies are missing. Install them with: pip install -e ".[drive]"'
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Archive to Drive failed: {exc}. Local outputs remain at: {result.output_dir}"
        ) from exc


def _print_archive_summary(archive_result: Any) -> None:
    if archive_result.already_archived:
        console.print("Drive archive: already archived; nothing to upload.")
    elif archive_result.uploaded_files:
        console.print(f"Drive archive: uploaded {len(archive_result.uploaded_files)} file(s).")
        for key, url in archive_result.uploaded_files.items():
            console.print(f"  {key}: {url}")
    else:
        console.print("Drive archive: no new files to upload.")

    if archive_result.skipped_files:
        console.print(f"Drive archive skipped: {', '.join(archive_result.skipped_files)}")
    console.print(f"Drive folder: {archive_result.drive_folder_url}")
    console.print(f"Updated log: {archive_result.log_path}")


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
