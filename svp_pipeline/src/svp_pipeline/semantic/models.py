"""Data models for RPE-based semantic repair."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RPEProposition(BaseModel):
    """A normalized proposition extracted from SVP or observation."""

    layer: str
    text: str
    kind: Literal["required", "forbidden", "observed", "missing", "violation"]

    model_config = ConfigDict(extra="forbid")


class ExpectedRPE(BaseModel):
    """Expected propositions derived from a target SVP."""

    required: list[RPEProposition] = Field(default_factory=list)
    forbidden: list[RPEProposition] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ObservedRPE(BaseModel):
    """Observed propositions from human notes or a future visual-RPE model."""

    artifact: str | None = None
    observed: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class SemanticIssue(BaseModel):
    """A semantic gap between target SVP expectations and observed RPE."""

    code: Literal["missing_required", "forbidden_violation"]
    severity: Literal["warn", "fail"]
    layer: str
    expected: str
    observed: str
    repair_hint: str

    model_config = ConfigDict(extra="forbid")


class SemanticDiffReport(BaseModel):
    """Semantic diff produced from Expected RPE and Observed RPE."""

    schema_version: Literal["SVP.semantic_diff.v1"] = "SVP.semantic_diff.v1"
    gate_status: Literal["pass", "repair"] = "pass"
    issues: list[SemanticIssue] = Field(default_factory=list)
    observed_notes: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class RepairProposal(BaseModel):
    """Repair proposal that can be used as the next target SVP input."""

    schema_version: Literal["SVP.repair_proposal.v1"] = "SVP.repair_proposal.v1"
    base_svp: str
    observed_rpe: str
    proposed_svp: str
    semantic_diff: str
    rationale: list[str] = Field(default_factory=list)
    regeneration_plan: dict[str, bool] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
