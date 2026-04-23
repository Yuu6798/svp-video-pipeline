# SVP Video Pipeline

## Overview
（M5 で執筆）

## Installation
（M5 で執筆）

## Usage
（M5 で執筆）

## Architecture
（M5 で執筆）

## Known Limitations
- M3 image backend is `gemini-3-pro-image-preview`.
- OpenAI `gpt-image-2` backend is deferred until organization verification is available.
- `auto` aspect ratio is mapped to `16:9` in M3 because Gemini image API does not accept `auto`.
- M3 only supports planner -> image. Video stage remains out of scope until M4.
- JSON-structured prompt sections are preserved to keep the same "JSON Supremacy" behavior observed in prior experiments.

## Development
（M5 で執筆）

## Manual Gemini Verification (M3)
Run the following command locally after setting API keys:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...

python -c "
from pathlib import Path
from svp_pipeline.pipeline import Pipeline

p = Pipeline(
    output_dir=Path('./out'),
    planner_model='claude-haiku-4-5',
)
result = p.run(
    '夕暮れの渋谷で少女が傘を畳む',
    duration=5,
    no_video=True,
)
print(f'Image saved: {result.image_path}')
print(f'Total cost: ${result.total_cost_usd:.4f}')
"
```

Recommended observation prompts:
1. still_life case (no human subject)
2. action_ninja case (`21:9`)
3. shibuya_dusk case (urban portrait)
4. interaction-bias suppression prompt
5. forbidden enforcement prompt
