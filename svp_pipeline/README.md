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
| `ANTHROPIC_API_KEY` | Always, for the planner | https://console.anthropic.com/ |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `--image-backend gemini` | https://aistudio.google.com/apikey |
| `OPENAI_API_KEY` | `--image-backend openai` | https://platform.openai.com/api-keys |
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

# OpenAI image backend
svp-video "朝の窓辺の白バラ" --image-backend openai --cheap

# SVP only
svp-video "アクションシーン" --dry-run

# Stop after image generation
svp-video "静物のマクロ撮影" --no-video

# Verbose JSON logs on stdout
svp-video "雨の夜の路地" --verbose
```

### CLI Options

| Option | Description | Default |
|---|---|---|
| `PROMPT` | Natural-language video prompt | Required |
| `--duration INTEGER` | Video duration in seconds, 4-15 | `5` |
| `--output PATH` | Output directory | `./out` |
| `--planner-model TEXT` | `claude-opus-4-7` or `claude-haiku-4-5` | `claude-opus-4-7` |
| `--image-backend TEXT` | `gemini` or `openai` | `gemini` |
| `--cheap` | Low-cost image/video settings | Off |
| `--dry-run` | Generate SVP only, with estimated downstream cost | Off |
| `--no-video` | Generate SVP + image, skip video | Off |
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

## Known Limitations

- Full video generation is paid. A typical 5-second standard run is roughly
  `$1.6`; `--cheap` is roughly `$0.5`, depending on the selected image backend.
- Seedance 2.0 currently supports 4-15 second videos.
- Gemini supports the SVP aspect ratio values directly except `auto`, which is
  resolved to `16:9`.
- OpenAI `gpt-image-2` supports only three native sizes plus `auto`; `21:9` and
  `4:3` are rounded to the nearest landscape size.
- C-group visual risks such as reversed hands, thin linear objects, and
  soft-body deformation still require manual observation.
- The pipeline does not yet support batch mode, existing SVP JSON input, Web UI,
  external object storage, or automated video Delta-E scoring.

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
