# Measurement: A/A Noise Floor (PROBE-0)

`svp-video probe-noise` measures how much a single image backend's own
regeneration noise moves pixel metrics, using the same fixed input each time
(an "A/A" test — same SVP against itself, not A vs. B). This is the
denominator every later effect claim (repair effectiveness, backend
comparison, `control_profile` grip) has to be checked against before it can
be called a real effect. The command and its `noise_floor.json` report are
an **instrument**: they report numbers, not verdicts. There is no pass/fail
threshold, no "acceptable noise" constant, and no `verdict` field anywhere in
this pipeline. Callers interpret the numbers in context.

## Why this exists: the n=1 lesson

The sibling audio project (`ugh-prompt-engine`) hit this repeatedly in its
K-series grip experiments: a single before/after comparison (`n=1`) cannot
distinguish a real effect from ordinary regeneration noise, and several
early "effects" it recorded later turned out to be noise-floor artifacts
once measured against a proper baseline. The operating rule it settled on
applies here unchanged:

- **Do not trust an n=1 conclusion.** One regeneration pair proves nothing
  about noise vs. signal.
- **An effect only counts once it clears the noise floor.** Before claiming
  a repair loop, a `control_profile` field, or a backend switch moved
  something, compare the observed delta against this backend's own A/A
  noise floor for a comparable SVP. If the delta is inside the noise floor's
  range, it is not (yet) a demonstrated effect.

`probe-noise` exists to make that baseline cheap to produce and reuse,
instead of re-deriving it ad hoc for every comparison.

## What it measures

`svp-video probe-noise --from-svp <svp.json> --backend <gemini|openai> -n <N> --output-dir <dir>`:

1. Reads the SVP JSON directly (no planner call — this isolates the image
   backend's own noise from planner variance).
2. Calls the image backend `N` times with the same SVP, saving
   `probe_00.png` .. `probe_{N-1:02d}.png` under `--output-dir`.
3. Computes the existing pixel metrics (`pixel_rms`, `mean_rgb_delta` from
   `semantic/visual_compare.py`, exposed as the public `pixel_metrics()`
   wrapper) for every pair of the `N` images (`N * (N-1) / 2` pairs).
4. Aggregates `mean`/`min`/`max` across all pairs and writes
   `noise_floor.json`.

`--corpus <manifest.json>` runs the same probe over several SVP files under
one shared `backend`/`n`, writing one sub-directory (named after each SVP
file's stem) with its own `noise_floor.json` under `--output-dir`.

## `noise_floor.json` schema

```json
{
  "schema_version": "SVP.noise_floor.v1",
  "svp_path": "tests/samples/action_ninja.json",
  "svp_sha256": "…64 hex chars…",
  "backend": "gemini",
  "model": "gemini-3-pro-image-preview",
  "n": 3,
  "timestamp_utc": "2026-07-07T12:00:00Z",
  "per_image_cost_usd": [0.08, 0.08, 0.08],
  "total_cost_usd": 0.24,
  "pairs": [
    {"index_a": 0, "index_b": 1, "pixel_rms": 4.1234, "mean_rgb_delta": 2.05},
    {"index_a": 0, "index_b": 2, "pixel_rms": 3.98, "mean_rgb_delta": 1.87},
    {"index_a": 1, "index_b": 2, "pixel_rms": 4.5, "mean_rgb_delta": 2.2}
  ],
  "pixel_rms_mean": 4.1978,
  "pixel_rms_min": 3.98,
  "pixel_rms_max": 4.5,
  "mean_rgb_delta_mean": 2.04,
  "mean_rgb_delta_min": 1.87,
  "mean_rgb_delta_max": 2.2
}
```

All floats are rounded to 4 decimal places per the repo-wide float
convention. `svp_sha256` pins the exact SVP content the noise floor was
measured against (compare against the same SVP file's sha256 before reusing
a cached noise floor for a new comparison). Every field above is either an
identifier, a raw measurement, or an aggregate of measurements — deliberately
no `status`/`verdict`/`threshold` field.

### Manifest format is JSON, not YAML

The original design sketch called `--corpus` manifests YAML. `svp_pipeline`'s
`pyproject.toml` has no PyYAML dependency anywhere (checked with
`grep -i yaml svp_pipeline/pyproject.toml`, zero hits), and adding one is out
of scope for this feature, so the manifest is JSON instead:

```json
{
  "backend": "gemini",
  "n": 3,
  "svp_files": ["action_ninja.json", "shibuya_dusk.json", "still_life_macro.json"]
}
```

`svp_files` entries are resolved relative to the manifest file's own
directory (absolute paths are used as-is). Unknown top-level keys, a missing
required key, or `n < 2` all fail fast with a `ValueError` before any image
is generated.

Before any generation starts, `run_corpus_noise_probe` preflights the whole
corpus (`probe/corpus.py::preflight_corpus`): every `svp_files` entry must
exist, be readable, and pass `SVPVideo` schema validation, and no two entries
may share an output-directory stem (two different paths both named
`scene.json` would otherwise collide on `output_dir/scene`, with the later
run silently overwriting the earlier one's images/`noise_floor.json`). A
single bad or duplicate-stem entry anywhere in the corpus aborts the run with
zero backend calls and zero partial reports written, rather than failing
partway through after incurring cost on earlier entries.

## Runbook (real API calls — billing applies)

`probe-noise` calls a real, billed image backend `N` times per SVP (or
`N * len(svp_files)` times for `--corpus`). There is no dry-run/estimate
mode for actual generation cost beyond the `per_image_cost_usd` values
reported after the fact — budget before running.

Recommended starting point: `n=3` across a small corpus of 3-5 SVP files.
`n=3` gives 3 pairs per SVP (enough to see a min/max spread, not just a
single pair), and 3-5 SVPs across different scene types gives some signal on
whether the noise floor is scene-dependent. The existing
`svp_pipeline/tests/samples/*.json` fixtures (`action_ninja.json`,
`shibuya_dusk.json`, `still_life_macro.json`) are already a usable initial
corpus — see `svp_pipeline/tests/samples/probe_corpus.json` for a
ready-to-run manifest referencing all three.

```bash
cd svp_pipeline
svp-video probe-noise --corpus tests/samples/probe_corpus.json --output-dir out/noise-floor/2026-07-07-gemini
```

Run this once per backend/model you plan to compare against or measure grip
for (a Gemini noise floor cannot bound an OpenAI comparison). Re-run
periodically — a backend's own noise characteristics can drift across model
version updates, so a noise floor measured against an old model version is
not automatically valid evidence for a new one.

## Case study record template

Record every real `probe-noise` run so noise floors are reusable evidence,
not one-off numbers. Required fields, matching the K-series case-study
convention this was ported from:

```markdown
### <date> — <backend> <model/version> noise floor

- Date: YYYY-MM-DD
- Backend / model version: <e.g. gemini-3-pro-image-preview>
- n: <N>
- Corpus: <manifest path or SVP file list>
- noise_floor.json path(s): <output dir(s)>
- pixel_rms: mean=<x> min=<x> max=<x>
- Comparison against this noise floor: <what was being checked, and whether
  its delta cleared min/max or fell inside it — REQUIRED, this is the whole
  point of measuring the floor>
```

## Connection to GRIP-1

PROBE-0's noise floor is the denominator for GRIP-1 (SVP single-field
perturbation grip measurement, queued in `STATUS.md`): a grip measurement
perturbs one `control_profile`-declared field and measures how much the
output moves, but that movement is only a real "grip" signal once it is
shown to exceed this backend's A/A noise floor for a comparable SVP.
GRIP-1's harness is expected to reuse `run_noise_probe`/`pixel_metrics`
directly rather than re-implementing pairwise pixel comparison.
