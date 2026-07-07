# CLI Reference

`svp-video` の全ワークフロー例と `CLI Options` 全項目。入口の最短実行例は
[`svp_pipeline/README.md`](../svp_pipeline/README.md) の Quick Start を参照。

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
