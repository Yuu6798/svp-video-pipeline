# Learned Audio Annotation Layer Policy

Status: design / pre-implementation
Scope: audio annotation models considered for the SVP pipeline
Audience: contributors adding learned-model backends

## 1. Goal

Add a learned-model audio annotation layer that augments — but does not
replace — the existing rule-based RPE layers, using only OSS components
under MIT / Apache-style licenses.

The existing `ExpectedRPE` / `ObservedRPE` extraction and the future
`PhysicalRPE` / `SemanticRPE` rule layers remain the deterministic,
evidence-bearing path. Learned-model output is isolated in a separate
`LearnedAudioAnnotations` container and is never folded back into
rule-derived evidence fields.

## 2. Core Principle

Learned-model output MUST NOT be written into `SemanticRPE.por_surface`,
`semantic_hypothesis`, `ExpectedRPE.required` / `forbidden`, or any other
rule-derived evidence field.

Rationale: rule-based propositions and external model estimates serve
different epistemic roles. Mixing them obscures which fields are measured
constraints and which are model-conditioned guesses, which breaks the
auditability of the semantic-diff loop.

A learned model's output is a tag with a confidence score and provenance.
It is never ground truth.

## 3. Adoption Matrix

### 3.1 Adopt

#### `beat_this`

- Use case: beat / downbeat detection
- License: MIT-compatible
- Constraints:
  - `dbn=False` MUST be fixed; we do not pull madmom DBN as a transitive
    dependency.
  - Ship behind an `optional` extra so the default install stays light.
  - Provide a deterministic fallback path (e.g., the current librosa
    backend) so pipelines without the extra still produce beat output.

#### `panns_inference`

- Use case: AudioSet 527-class acoustic tags and embeddings
- Constraints:
  - Output is treated as `external acoustic tags`, not as mood / genre /
    music-quality ground truth.
  - Top-k labels and an optional embedding only; no full posterior dump.
  - Never written into `SemanticRPE` evidence layers.

#### `basic-pitch`

- Use case: polyphonic note events and melody contour reinforcement
- Constraints:
  - Additive only. The current `pyin` melody contour MUST NOT be replaced
    in the same change set.
  - Ship behind an `optional` extra.

### 3.2 Reject

#### `Essentia` / `essentia-tensorflow`

Reason: AGPL and several non-commercial model weights conflict with the
distribution policy of this pipeline.

#### `madmom`

Reason: not compatible with Python 3.11+; bundled models / data are under
non-commercial terms.

#### `BeatNet`

Reason: depends on `madmom` and inherits its compatibility and license
issues.

### 3.3 Hold

#### `openl3` / `torchopenl3`

Reason: Python 3.11 compatibility, maintenance status, and weights
licensing all need re-verification before adoption.

#### `autochord`

Reason: package itself is MIT, but it depends on the NNLS-Chroma VAMP
plugin (GPL-2.0) and is unsupported on Windows. Not adoptable as is.

#### `EfficientAT`

Reason: promising but requires a spike to confirm packaging,
dependencies, and weights license terms.

## 4. License Policy

- Only MIT, Apache-2.0, BSD-class, and equivalently permissive licenses
  are acceptable for new runtime dependencies.
- Model weights distributed by an upstream project must be inspected
  separately from the code license. A permissive code license does not
  imply permissive weights.
- Each learned-annotation output record MUST carry `source_model`,
  `source_version`, and a `license_metadata` map so downstream consumers
  can audit provenance without reloading the model.

## 5. Optional Dependency Policy

All learned-model backends are gated behind opt-in `pyproject.toml`
extras:

| Extra        | Pulls in                |
|--------------|-------------------------|
| `beat`       | `beat_this`             |
| `learned-tags` | `panns_inference`     |
| `pitch`      | `basic-pitch`           |

The default install MUST remain green without any of these extras. Each
backend module performs a guarded import and falls back gracefully (or
omits the corresponding annotations) when the extra is not installed.

## 6. Output Isolation

Learned annotations live in a single dedicated container, attached to
the run bundle as a sibling field — not as a sub-field of any
rule-derived RPE.

Required metadata on every learned-annotation payload:

- `schema_version`
- `enabled_models` — which adapters were actually invoked for this run
- `labels[].source_model` / `source_version`
- `inference_config` — model-specific knobs that affected the output
- `license_metadata`
- `estimation_disclaimer` — a static string asserting that the contents
  are model estimates, not production-quality truth labels

## 7. Non-Goals

- Replacing librosa beat tracking, `pyin` melody extraction, or any
  current deterministic backend in the same change set that introduces
  the learned variant.
- Using learned tags to score or gate semantic repair decisions.
- Bundling pretrained weights inside this repository.

## 8. Implementation Order

The implementation is split into independent PRs so each step is small
and reviewable:

1. **PR1 — docs only** (this document).
2. **PR2 — schema only.** Add `LearnedAudioAnnotations` and an optional
   field on the run bundle. No backend code, no new runtime deps.
3. **PR3 — `beat_this` backend spike.** Optional extra `beat`,
   `dbn=False`, fallback to the current librosa path, fake-backend
   tests.
4. **PR4 — `panns_inference` backend spike.** Optional extra
   `learned-tags`, top-k tags + optional embedding into
   `LearnedAudioAnnotations`. No write-through into `SemanticRPE`.
5. **PR5 — `basic-pitch` backend spike.** Optional extra `pitch`,
   note-event output, additive next to the existing `pyin` contour.

## 9. Acceptance Criteria

A change set in this track is acceptable only if all of the following
hold:

- `Essentia`, `essentia-tensorflow`, `madmom`, and `BeatNet` are not
  introduced as direct or transitive runtime dependencies.
- Learned-model output is confined to `LearnedAudioAnnotations`.
- `ExpectedRPE` / `ObservedRPE` (and the future `PhysicalRPE` /
  `SemanticRPE`) evidence layers are not modified to consume learned
  output.
- The default install (no optional extras) still passes the existing
  pipeline test suite.
- Every learned annotation record carries model name, model version,
  and license metadata.
- This document is updated whenever the adopt / reject / hold lists
  change.

## 10. Notes On Existing Code

The brief that drives this policy uses the namespace `svp_rpe` and
references `PhysicalRPE` / `SemanticRPE`. The current code lives under
`svp_pipeline.semantic` and exposes `ExpectedRPE` / `ObservedRPE` /
`RPEProposition`. The isolation rule applies to whichever namespace the
learned backends end up in; the schema PR (PR2) will pin the concrete
attachment point on the run bundle.
