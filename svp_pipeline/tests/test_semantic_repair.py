"""Tests for RPE-based semantic repair."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from svp_pipeline.schema import SVPVideo
from svp_pipeline.semantic.image_audit import audit_image_to_observed_rpe
from svp_pipeline.semantic.models import ObservedRPE
from svp_pipeline.semantic.repair import repair_svp_from_observed_rpe
from svp_pipeline.semantic.rpe import diff_rpe, extract_expected_rpe
from svp_pipeline.semantic.visual_compare import write_visual_comparison

SAMPLES_DIR = Path(__file__).parent / "samples"


def _write_png(path: Path, color: tuple[int, int, int] = (0, 0, 0)) -> None:
    Image.new("RGB", (2, 2), color=color).save(path)


def _load(name: str) -> SVPVideo:
    return SVPVideo.model_validate_json((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def test_extract_expected_rpe_includes_core_constraints() -> None:
    svp = _load("shibuya_dusk.json")

    expected = extract_expected_rpe(svp)
    required_text = "\n".join(prop.text for prop in expected.required)
    forbidden_text = "\n".join(prop.text for prop in expected.forbidden)

    assert svp.por_core[0] in required_text
    assert svp.grv_anchor[0] in required_text
    assert forbidden_text


def test_diff_rpe_creates_repair_issues() -> None:
    svp = _load("shibuya_dusk.json")
    expected = extract_expected_rpe(svp)
    observed = ObservedRPE(
        missing=["single character remains centered"],
        violations=["sword-like background reflection"],
        notes=["manual visual check"],
    )

    report = diff_rpe(
        expected=expected,
        observed=observed,
    )

    assert report.gate_status == "repair"
    assert [issue.code for issue in report.issues] == [
        "missing_required",
        "forbidden_violation",
    ]


def test_repair_svp_from_observed_rpe_writes_proposal_files(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    base_svp = tmp_path / "svp.json"
    observed_rpe = tmp_path / "observed_rpe.json"
    base_svp.write_text(svp.model_dump_json(indent=2), encoding="utf-8")
    observed_rpe.write_text(
        json.dumps(
            {
                "missing": ["single character remains centered"],
                "violations": ["sword-like background reflection"],
                "notes": ["manual visual check"],
            }
        ),
        encoding="utf-8",
    )

    result = repair_svp_from_observed_rpe(
        base_svp_path=base_svp,
        observed_rpe_path=observed_rpe,
        output_root=tmp_path,
    )

    assert result.semantic_diff_path.exists()
    assert result.proposal_path.exists()
    assert result.proposed_svp_path.exists()
    proposed = SVPVideo.model_validate_json(result.proposed_svp_path.read_text(encoding="utf-8"))
    assert "single character remains centered" in proposed.c3.constraints.required
    assert "sword-like background reflection" in proposed.c3.constraints.forbidden
    assert "sword-like background reflection" in proposed.pose_layer.constraints.forbidden


def test_repair_promotes_object_graph_and_contact_failures(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    svp = svp.model_copy(
        update={
            "pose_layer": svp.pose_layer.model_copy(
                update={
                    "contact_points": [
                        "left_hand -> sword grip",
                        "right_hand -> sword grip",
                    ]
                }
            )
        }
    )
    base_svp = tmp_path / "svp.json"
    observed_rpe = tmp_path / "observed_rpe.json"
    base_svp.write_text(svp.model_dump_json(indent=2), encoding="utf-8")
    observed_rpe.write_text(
        json.dumps(
            {
                "state": {
                    "object_graph": [
                        "katana_blade: single glowing drawn blade",
                        "scabbard: separate dark sheath at upper-left",
                    ],
                    "contact_graph": [
                        "right_hand -> katana_handle",
                        "left_hand -> scabbard",
                    ],
                    "viewer_contact_graph": [
                        "viewer_left_hand -> katana_handle",
                        "viewer_right_hand -> scabbard",
                    ],
                    "anatomical_contact_graph": [
                        "character_right_hand -> katana_handle",
                        "character_left_hand -> scabbard",
                    ],
                    "pose_intent": "unsheathing / draw-pose",
                    "failure_modes": [
                        "left hand grips the wrong object",
                        "blade and scabbard are fused",
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    result = repair_svp_from_observed_rpe(
        base_svp_path=base_svp,
        observed_rpe_path=observed_rpe,
        output_root=tmp_path,
    )

    proposed = result.proposed_svp
    assert result.diff.gate_status == "repair"
    assert [issue.layer for issue in result.diff.issues] == [
        "observed_state.failure_modes",
        "observed_state.failure_modes",
    ]
    assert (
        "Object state must be preserved: scabbard: separate dark sheath at upper-left"
        in proposed.c3.constraints.required
    )
    assert "left_hand -> sword grip" not in proposed.pose_layer.contact_points
    assert "right_hand -> sword grip" not in proposed.pose_layer.contact_points
    assert "RPE contact graph: left_hand -> scabbard" in proposed.pose_layer.contact_points
    assert "RPE contact graph: right_hand -> katana_handle" in proposed.pose_layer.contact_points
    assert (
        "RPE viewer contact graph: viewer_left_hand -> katana_handle"
        in proposed.pose_layer.contact_points
    )
    assert (
        "RPE anatomical contact graph: character_right_hand -> katana_handle"
        in proposed.pose_layer.contact_points
    )
    assert "RPE pose intent: unsheathing / draw-pose" in proposed.pose_layer.contact_points
    assert any(
        "left hand grips the wrong object" in item
        for item in proposed.c3.evaluation_criteria.critical_fail_conditions
    )
    assert any(
        "blade and scabbard are fused" in item
        for item in proposed.pose_layer.constraints.forbidden
    )


def test_image_audit_writes_observed_rpe_state_scaffold(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    base_svp = tmp_path / "svp.json"
    image = tmp_path / "image.png"
    base_svp.write_text(svp.model_dump_json(indent=2), encoding="utf-8")
    _write_png(image)

    result = audit_image_to_observed_rpe(
        svp_path=base_svp,
        image_path=image,
        output_root=tmp_path,
        observed=["silver ponytail is visible"],
        missing=["red eye color is weak"],
        violations=["sword-like reflection on wet floor"],
        object_states=["scabbard: separate dark sheath at upper-left"],
        contact_graph=["left_hand -> scabbard"],
        viewer_contact_graph=["viewer_left_hand -> katana_handle"],
        anatomical_contact_graph=["character_right_hand -> katana_handle"],
        pose_intent="unsheathing / draw-pose",
        failure_modes=["left hand grips the wrong object"],
        notes=["manual visual check"],
    )

    assert result.expected_rpe_path.exists()
    assert result.observed_rpe_path.exists()
    observed = ObservedRPE.model_validate_json(
        result.observed_rpe_path.read_text(encoding="utf-8")
    )
    assert observed.artifact == str(result.source_image_path)
    assert observed.observed == ["silver ponytail is visible"]
    assert observed.missing == ["red eye color is weak"]
    assert observed.violations == ["sword-like reflection on wet floor"]
    assert observed.state["object_graph"] == ["scabbard: separate dark sheath at upper-left"]
    assert observed.state["contact_graph"] == ["left_hand -> scabbard"]
    assert observed.state["viewer_contact_graph"] == ["viewer_left_hand -> katana_handle"]
    assert observed.state["anatomical_contact_graph"] == [
        "character_right_hand -> katana_handle"
    ]
    assert observed.state["pose_intent"] == "unsheathing / draw-pose"
    assert observed.state["failure_modes"] == ["left hand grips the wrong object"]
    assert "expected_identity" in observed.state
    assert result.source_image_path.name == "source_image.png"
    assert result.source_image_path.exists()


def test_image_audit_openai_backend_parses_observed_rpe(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    base_svp = tmp_path / "svp.json"
    image = tmp_path / "image.png"
    base_svp.write_text(svp.model_dump_json(indent=2), encoding="utf-8")
    _write_png(image)

    class _FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["response_format"] == {"type": "json_object"}
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "artifact": "model-supplied-wrong-path.png",
                                    "observed": ["single character is visible"],
                                    "missing": ["red eye color is weak"],
                                    "violations": ["background blade residue"],
                                    "notes": ["mock vision audit"],
                                    "state": {"identity": ["silver ponytail"]},
                                }
                            )
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions()),
    )

    result = audit_image_to_observed_rpe(
        svp_path=base_svp,
        image_path=image,
        output_root=tmp_path,
        backend="openai",
        client=fake_client,
    )

    assert result.observed_rpe.observed == ["single character is visible"]
    assert result.observed_rpe.missing == ["red eye color is weak"]
    assert result.observed_rpe.violations == ["background blade residue"]
    assert result.observed_rpe.artifact == str(result.source_image_path)
    assert result.observed_rpe.state["identity"] == ["silver ponytail"]
    assert result.observed_rpe.state["target_svp"] == str(base_svp)


def test_repair_pass_diff_does_not_request_regeneration(tmp_path: Path) -> None:
    svp = _load("shibuya_dusk.json")
    base_svp = tmp_path / "svp.json"
    observed_rpe = tmp_path / "observed_rpe.json"
    base_svp.write_text(svp.model_dump_json(indent=2), encoding="utf-8")
    observed_rpe.write_text(
        json.dumps({"observed": ["image matches the target SVP"]}),
        encoding="utf-8",
    )

    result = repair_svp_from_observed_rpe(
        base_svp_path=base_svp,
        observed_rpe_path=observed_rpe,
        output_root=tmp_path,
    )

    proposal = json.loads(result.proposal_path.read_text(encoding="utf-8"))
    assert result.diff.gate_status == "pass"
    assert proposal["regeneration_plan"] == {
        "reuse_reference": True,
        "reuse_image": True,
        "regenerate_image": False,
        "regenerate_video": False,
    }


def test_visual_comparison_writes_packet_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    candidate = tmp_path / "candidate.png"
    _write_png(source)
    _write_png(candidate)

    result = write_visual_comparison(
        source_image_path=source,
        candidate_image_path=candidate,
        output_dir=tmp_path / "packet",
        notes=["compare source and regenerated image"],
    )

    assert result.source_image_path.name == "source_image.png"
    assert result.candidate_image_path.name == "candidate_image.png"
    assert result.comparison_path.exists()
    payload = json.loads(result.comparison_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "SVP.visual_comparison.v1"
    assert payload["dimensions_equal"] is True
    assert payload["pixel_rms"] == 0.0
    assert payload["source_image"]["sha256"] == payload["candidate_image"]["sha256"]
