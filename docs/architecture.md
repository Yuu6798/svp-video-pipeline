# Architecture

パイプラインの段階構成と、各補助フラグ（`--reference-image` /
`--separate-character-bg` / `--from-svp` / `--reuse-run` /
`--repair-from-rpe` / `--audit-image` 系）が何を行うかの詳細。

```text
Natural language prompt
        |
        v
Planner (Claude)
        |
        v
SVP.v4x-five-layer.video JSON
        |
        v
Image backend (Gemini or OpenAI)
        |
        v
Reference PNG
        |
        v
Seedance 2.0 reference-to-video
        |
        v
MP4 video
```

The image prompt uses the composition, face, style, and pose layers. The motion
prompt uses `motion_layer`, `por_core`, `grv_anchor`, and the motion-specific
constraints, while referring to the generated image as `@Image1`.
`generator/planner.py`'s 4 post-processing scenarios (character lock,
background noise control, umbrella/katana object contact, katana reflection
policy) are declared as data in `generator/planner_rules.py` (`LayerRule`
tuples per scenario) and folded onto the SVP by a single `apply_scenario`
engine, instead of one hand-written method per touched layer.
When `--reference-image` is provided, the image backend uses that file as an
additional visual reference; SVP text remains the primary semantic control.
`--reference-crop` is intended for 3x3 character sheets. It avoids passing the
entire collage/grid to the image model, which can otherwise reproduce duplicate
characters or panel layouts.
Reference images are treated as character/style references only: the prompt tells
the image backend not to copy reference backgrounds, panel layouts, duplicate
poses, weapon trails, compression artifacts, or texture noise.
`--separate-character-bg` goes further by generating a green-screen character
plate and a background plate separately, compositing them into `image.png`, and
then passing that composite to Seedance. It also saves `character_green.png`,
`background_clean.png`, and `composite.png` for inspection.
`--from-svp` and `--reuse-run` support a code-development-like regeneration
loop: edit or repair the SVP, reuse known-good image/composite artifacts when
appropriate, and rerun only the stages that still need generation. Reuse metadata
is written to `log.json.inputs`; reused planner/image stages are recorded as
`status: "reused"`.
`--repair-from-rpe` closes the first semantic repair loop. It reads an
`observed_rpe.json` file, compares its `missing` / `violations` against the
target SVP's expected RPE, and writes `semantic_diff.json`,
`repair_proposal.json`, and `target_svp.proposed.json` without calling any
generation API. A minimal observation file looks like:

```json
{
  "missing": ["single character remains centered"],
  "violations": ["sword-like background reflection"],
  "notes": ["manual visual check"]
}
```

`--audit-image` is the image-side entry point for that loop. It creates an audit
packet, copies the audited image into it as `source_image.*`, and writes
`expected_rpe.json` / `observed_rpe.json` from a target SVP plus that image
artifact. In `manual` mode, the caller supplies state observations with
`--audit-observed`, `--audit-missing`, and `--audit-violation`; this is useful
when Codex or a human has inspected the image and identified concrete state
variables such as contact-graph failures, background residue, or identity drift.
`--compare-image` adds a candidate image (for example a Codex repair preview or a
pipeline regenerated image) to the same packet as `candidate_image.*` and writes
`visual_comparison.json` with dimensions, hashes, file sizes, and pixel metrics
when dimensions match. With `--audit-repair`, the same command immediately
produces `target_svp.proposed.json` for the next regeneration pass.
For tool-heavy poses, prefer structured state variables over generic critique:
`--audit-object-state`, `--audit-contact-graph`, `--audit-pose-intent`, and
`--audit-failure-mode` distinguish object roles such as blade vs scabbard and
contact failures such as the left hand gripping the wrong object. These fields
are promoted into pose constraints, contact points, object rules, and critical
failure conditions during repair.
`--audit-regenerate` completes that pass in one command: it takes the repaired
SVP, runs the normal pipeline from `target_svp.proposed.json`, stores the
regenerated run under the audit packet, and when `--compare-image` is present
writes a second visual comparison against that target preview. This makes Codex
or human repair previews usable as intermediate improvement targets while the
final judgment remains the pipeline backend's regenerated image.
`--audit-backend openai` is reserved for vision-model auditing and requires
`OPENAI_API_KEY`.
