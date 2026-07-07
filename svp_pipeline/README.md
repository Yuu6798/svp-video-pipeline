# SVP Video Pipeline

SVP Video Pipeline turns a natural-language prompt into a structured video
prompt, a reference image, and an MP4 video.

The pipeline is built around `SVP.v4x-five-layer.video`: a five-layer schema that
keeps the semantic core of a scene explicit across planning, image generation,
and reference-to-video generation. This makes each stage inspectable and
repeatable instead of hiding the full creative brief inside one prompt string.

The current implementation supports Claude for planning, Gemini or OpenAI for
image generation, and Seedance 2.0 reference-to-video through fal.ai for video.

## 設計ドキュメント

| ドキュメント | 内容 |
|---|---|
| [`docs/cli.md`](../docs/cli.md) | `svp-video` の全ワークフロー例と CLI Options 全項目 |
| [`docs/architecture.md`](../docs/architecture.md) | パイプライン段階構成と各補助フラグ（reference-image / split-composite / from-svp / reuse-run / repair-from-rpe / audit-image 系）の詳細 |
| [`docs/backend_comparison.md`](../docs/backend_comparison.md) | Gemini/OpenAI/split-composite の定性比較スナップショット (2026-04-25) |

## Installation

Python 3.11 or newer is required.

```bash
cd svp_pipeline
pip install -e ".[dev]"
cp .env.example .env   # fill in API keys
```

Required keys: `ANTHROPIC_API_KEY` (planner), `GOOGLE_API_KEY` or
`GEMINI_API_KEY` (Gemini image backend, default), `OPENAI_API_KEY`
(`--image-backend openai` only), `FAL_KEY` (full video runs only). See
`.env.example` for optional defaults (`DEFAULT_PLANNER_MODEL`,
`DEFAULT_IMAGE_BACKEND`, `DEFAULT_OUTPUT_DIR`).

## Usage

```bash
# Full run: Claude -> image -> Seedance video
svp-video "夕暮れの渋谷で少女が傘を畳む"

# Low-cost mode: Gemini 1K or OpenAI low, Seedance fast tier, 480p
svp-video "朝の窓辺の白バラ" --cheap

# SVP only
svp-video "アクションシーン" --dry-run
```

Output is written to a timestamped directory (`svp.json`, `image.png`,
`video.mp4`, `log.json`). Full command catalog (reference images,
split-composite generation, repair/audit loops, reuse) and the complete CLI
options table live in [`docs/cli.md`](../docs/cli.md).

## Architecture

```text
Prompt -> Planner (Claude) -> SVP JSON -> Image backend (Gemini/OpenAI)
       -> Reference PNG -> Seedance 2.0 -> MP4 video
```

`--reference-image` / `--separate-character-bg` control image-generation
fidelity; `--from-svp` / `--reuse-run` support a code-development-like
regeneration loop; `--repair-from-rpe` and `--audit-image` close a semantic
repair loop from observed image state back to a proposed SVP. Full stage
breakdown and data-flow details: [`docs/architecture.md`](../docs/architecture.md).

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

Qualitative Gemini/OpenAI/split-composite comparison data:
[`docs/backend_comparison.md`](../docs/backend_comparison.md).

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
