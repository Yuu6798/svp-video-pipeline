# SVP Video Pipeline

Natural-language prompt → SVP five-layer plan → reference image → MP4 video.
Uses Claude (planner), Gemini/OpenAI (image), Seedance 2.0 via fal.ai (video).

詳細ドキュメント (Usage / CLI options / Architecture / Backend Comparison
Snapshot 等) は [`svp_pipeline/README.md`](svp_pipeline/README.md) を参照。

## Requirements

- Python 3.11+
- API keys:
  - `ANTHROPIC_API_KEY` — planner、必須
  - `GOOGLE_API_KEY` — image、デフォルトの Gemini backend に必須
  - `OPENAI_API_KEY` — OpenAI backend を使う場合のみ。実行時に
    `--image-backend openai` を併用する必要あり
  - `FAL_KEY` — video full run のみ

## Quick Start

```bash
git clone https://github.com/Yuu6798/svp-video-pipeline.git
cd svp-video-pipeline/svp_pipeline
pip install -e ".[dev]"
cp .env.example .env   # API keys を記入
svp-video "夕暮れの渋谷で少女が傘を畳む"
```

上記コマンドはデフォルトの Gemini backend で動作する (したがって
`GOOGLE_API_KEY` が必要)。OpenAI backend を使う場合は
`svp-video "..." --image-backend openai` を実行する。

CLI 全オプション、アーキテクチャ、既知の制約は
[`svp_pipeline/README.md`](svp_pipeline/README.md) を参照してください。

## License

MIT — [LICENSE](LICENSE) 参照。
