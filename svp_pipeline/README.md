# SVP Video Pipeline

SVP Video Pipeline turns a natural-language prompt into a structured video
prompt, a reference image, and an MP4 video.

The pipeline is built around `SVP.v4x-five-layer.video`: a five-layer schema that
keeps the semantic core of a scene explicit across planning, image generation,
and reference-to-video generation. This makes each stage inspectable and
repeatable instead of hiding the full creative brief inside one prompt string.

The current implementation supports Claude for planning, Gemini or OpenAI for
image generation, and Seedance 2.0 reference-to-video through fal.ai for video.

## Installation

Python 3.11 or newer is required.

```bash
git clone https://github.com/Yuu6798/svp-video-pipeline.git
cd svp-video-pipeline/svp_pipeline
pip install -e ".[dev]"
```

Configure API keys with environment variables or a local `.env` file:

```bash
cp .env.example .env
```

Required keys:

| Key | Required when | Console |
|---|---|---|
| `ANTHROPIC_API_KEY` | Planner runs (`--from-svp` / `--reuse-run` skip this) | https://console.anthropic.com/ |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Gemini image generation (`--reuse-run` skips this) | https://aistudio.google.com/apikey |
| `OPENAI_API_KEY` | OpenAI image generation (`--reuse-run` skips this) | https://platform.openai.com/api-keys |
| `FAL_KEY` | Full video runs | https://fal.ai/dashboard/keys |

Optional defaults:

```bash
DEFAULT_PLANNER_MODEL=claude-opus-4-7
DEFAULT_IMAGE_BACKEND=gemini
DEFAULT_OUTPUT_DIR=./out
```

## Usage

```bash
# Full run: Claude -> image -> Seedance video
svp-video "夕暮れの渋谷で少女が傘を畳む"

# Low-cost mode: Gemini 1K or OpenAI low, Seedance fast tier, 480p
svp-video "朝の窓辺の白バラ" --cheap

# Generate and archive binary artifacts to Google Drive
svp-video "sample prompt" --cheap --archive-drive

# OpenAI image backend
svp-video "朝の窓辺の白バラ" --image-backend openai --cheap

# Optional character/style reference image
svp-video "cyberpunk rain city, silver ponytail woman with red eyes" \
  --reference-image ./refs/character.png \
  --reference-crop 1 \
  --image-backend openai \
  --cheap

# Experimental split character/background generation before video
svp-video "cyberpunk rain city, silver ponytail woman with red eyes" \
  --image-backend openai \
  --reference-image ./refs/character_sheet.jpg \
  --reference-crop 1 \
  --separate-character-bg \
  --cheap

# SVP only
svp-video "アクションシーン" --dry-run

# Stop after image generation
svp-video "静物のマクロ撮影" --no-video

# Regenerate from an edited / repaired SVP without calling Claude
svp-video --from-svp ./out/20260425-140453/target_svp.proposed.json --no-video

# Build a repaired SVP proposal from observed RPE / semantic failures
svp-video --from-svp ./out/20260425-140453/svp.json \
  --repair-from-rpe ./out/20260425-140453/observed_rpe.json

# Audit a generated image into observed RPE, then write a repaired SVP proposal
svp-video --from-svp ./out/20260425-140453/svp.json \
  --audit-image ./out/20260425-140453/image.png \
  --compare-image ./out/20260425-150000/image.png \
  --audit-violation "sword-like reflection appears in the background" \
  --audit-missing "red eyes are weak" \
  --audit-repair

# Audit, repair, regenerate from the repaired SVP, then compare against a target preview
svp-video --from-svp ./out/20260425-140453/svp.json \
  --audit-image ./out/20260425-140453/image.png \
  --compare-image ./out/codex_repair_preview.png \
  --audit-object-state "katana_blade: single glowing drawn blade" \
  --audit-object-state "scabbard: separate dark sheath at upper-left" \
  --audit-contact-graph "right_hand -> katana_handle" \
  --audit-contact-graph "left_hand -> scabbard" \
  --audit-pose-intent "unsheathing / draw-pose" \
  --audit-failure-mode "left hand grips the wrong object" \
  --audit-failure-mode "blade and scabbard are fused" \
  --audit-missing "glowing lavender eyes from the target preview" \
  --audit-repair \
  --audit-regenerate \
  --no-video

# Reuse an existing intermediate image artifact and regenerate later stages
svp-video --reuse-run ./out/20260425-140453 --reuse-image composite --cheap

# Verbose JSON logs on stdout
svp-video "雨の夜の路地" --verbose
```

### CLI Options

| Option | Description | Default |
|---|---|---|
| `PROMPT` | Natural-language video prompt; optional with `--from-svp` or `--reuse-run` | Required unless reusing |
| `--duration INTEGER` | Video duration in seconds, 4-15 | `5` |
| `--output PATH` | Output directory | `./out` |
| `--planner-model TEXT` | `claude-opus-4-7` or `claude-haiku-4-5` | `claude-opus-4-7` |
| `--image-backend TEXT` | `gemini` or `openai` | `gemini` |
| `--reference-image PATH` | Optional image reference for image generation | Off |
| `--reference-crop 1-9` | Crop a 3x3 reference sheet to one panel | Off |
| `--separate-character-bg` | OpenAI-only experimental route: generate character/background separately, then composite | Off |
| `--from-svp PATH` | Use an existing SVP JSON and skip Claude planning | Off |
| `--reuse-run PATH` | Reuse `svp.json` and an image artifact from an existing output run | Off |
| `--reuse-image TEXT` | Artifact from `--reuse-run`: `image`, `composite`, `character_green`, `background_clean` | `image` |
| `--repair-from-rpe PATH` | Compare observed RPE against target SVP and write `target_svp.proposed.json` | Off |
| `--audit-image PATH` | Convert an image artifact into `observed_rpe.json` state variables | Off |
| `--compare-image PATH` | Candidate/preview/regenerated image to store and compare in the audit packet | Off |
| `--audit-backend TEXT` | `manual` for supplied findings, or `openai` for vision-model audit | `manual` |
| `--audit-observed TEXT` | Observed image state proposition; repeatable | Off |
| `--audit-missing TEXT` | Missing expected image state proposition; repeatable | Off |
| `--audit-violation TEXT` | Forbidden image state observed in the artifact; repeatable | Off |
| `--audit-object-state TEXT` | Structured object role/state, e.g. `scabbard: separate dark sheath`; repeatable | Off |
| `--audit-contact-graph TEXT` | Structured hand/object relation, e.g. `left_hand -> scabbard`; repeatable | Off |
| `--audit-pose-intent TEXT` | Intended pose/action state, e.g. `unsheathing / draw-pose` | Off |
| `--audit-failure-mode TEXT` | Explicit visual-state failure, e.g. `blade and scabbard are fused`; repeatable | Off |
| `--audit-repair` | After `--audit-image`, also write semantic diff and repaired SVP proposal | Off |
| `--audit-regenerate` | After `--audit-repair`, regenerate from `target_svp.proposed.json`; with `--compare-image`, compare the regenerated image against the target preview | Off |
| `--character-lock` / `--no-character-lock` | Preserve literal character traits in SVP planning | On |
| `--cheap` | Low-cost image/video settings | Off |
| `--dry-run` | Generate SVP only, with estimated downstream cost | Off |
| `--no-video` | Generate SVP + image, skip video | Off |
| `--archive-drive` | Upload generated binary artifacts to Google Drive after a successful non-dry run | Off |
| `--verbose`, `-v` | Print verbose JSON logs and tracebacks | Off |
| `--version` | Print package version | |
| `--help` | Print CLI help | |

Output is written to a timestamped directory:

```text
out/20260425-123456/
  svp.json
  image.png
  video.mp4
  log.json
```

## Architecture

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

## Known Limitations

- Full video generation is paid. A typical 5-second standard run is roughly
  `$1.6`; `--cheap` is roughly `$0.5`, depending on the selected image backend.
- Seedance 2.0 currently supports 4-15 second videos.
- Gemini supports the SVP aspect ratio values directly except `auto`, which is
  resolved to `16:9`.
- OpenAI `gpt-image-2` supports only three native sizes plus `auto`; `21:9` and
  `4:3` are rounded to the nearest landscape size.
- Reference images improve character/style reproducibility but do not guarantee
  pixel-level identity preservation. Keep critical traits in the text prompt and
  SVP forbidden constraints as well.
- Character sheets should be cropped to a single panel with `--reference-crop`;
  passing an entire grid can cause duplicate background figures or collage-like
  layouts.
- `--separate-character-bg` reduces reference-background bleed, but the current
  chroma-key compositing can still leave edge artifacts or a mild composited look.
- C-group visual risks such as reversed hands, thin linear objects, and
  soft-body deformation still require manual observation.
- The pipeline does not yet support batch mode, Web UI, external object storage,
  or automated video Delta-E scoring. Existing SVP JSON input is supported with
  `--from-svp`; existing run artifact reuse is supported with `--reuse-run`.

## Backend Comparison Snapshot (2026-04-25)

Qualitative observations from manual comparison runs of the reference-image
+ split-composite path (PR #8). Sample basis: a recurring cyberpunk character
prompt with a 3x3 reference sheet input.

| Backend mode | Character fidelity | Background quality | Notable artifacts |
|---|---|---|---|
| `--image-backend gemini` | weak | clean composition | character traits drift from reference |
| `--image-backend openai` | high | grainy / noisy | reference-background bleed (e.g., katana traces) |
| `--image-backend openai --separate-character-bg` | high | clean | mild edge halo from chroma-key composite |

- `--separate-character-bg` produced the best end-to-end output in this
  comparison and is the recommended path for character-preservation work,
  despite the additional generation step (two OpenAI image calls + composite)
  and the modest edge-artifact tradeoff.
- The Gemini backend remains the default for prompts that prioritize scene
  composition over literal character preservation.
- This snapshot is qualitative (manual visual inspection), not a scored
  benchmark. C-group risk items (reverse grip, linear-object handling, soft-
  body deformation) remain on the deferred list.

This observation also closes the M3-deferred `gpt-image-2` forbidden-effect
evaluation: forbidden constraints were respected in the OpenAI runs, but
reference-background bleed dominated other concerns and motivated the
split-composite route.

## Development

```bash
pytest tests/ -v
ruff check src/ tests/
```

Focused test runs:

```bash
pytest tests/test_cli.py
pytest tests/test_image.py
pytest tests/test_image_openai.py
pytest tests/test_video.py
```

Manual smoke tests:

```bash
svp-video "テストプロンプト" --dry-run
svp-video "テストプロンプト" --no-video --cheap
svp-video "テストプロンプト" --cheap
```

## Archive to Google Drive

Optional helper for archiving generated image / video artifacts to a personal
Google Drive folder. Use `--archive-drive` on `svp-video` for the normal
one-step workflow, or run the helper manually for existing output directories.

```bash
pip install -e ".[drive]"
# Create an OAuth Desktop client in Google Cloud Console and save credentials as:
# ~/.config/svp-pipeline/google-credentials.json

svp-video "test prompt" --cheap --archive-drive

python -m svp_pipeline.tools.archive_to_drive out/20260425-140453
```

See `python -m svp_pipeline.tools.archive_to_drive --help` for options.
