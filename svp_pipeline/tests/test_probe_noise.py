"""Tests for the PROBE-0 A/A noise-floor harness (probe/noise.py, probe/corpus.py, CLI).

All backends here are fakes; no real image API is ever called.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import svp_pipeline.cli as cli_mod
from svp_pipeline.cli import app
from svp_pipeline.generator.image import ImageResult
from svp_pipeline.probe.corpus import (
    NoiseCorpusManifest,
    load_corpus_manifest,
    run_corpus_noise_probe,
)
from svp_pipeline.probe.noise import (
    NoiseFloorReport,
    run_noise_probe,
    write_noise_floor_report,
)
from svp_pipeline.semantic.visual_compare import pixel_metrics
from tests.fixtures.helpers import load_sample as _load
from tests.fixtures.helpers import valid_png_bytes

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class _FakeNoiseImageBackend:
    """Fake image backend returning distinct solid-color PNGs per call.

    Cycles through ``colors`` in order so tests can assert on known,
    non-trivial pairwise pixel distances (a backend that always returns the
    same bytes would make every pair distance 0, which cannot distinguish
    "aggregation math is correct" from "aggregation math is a no-op").
    """

    def __init__(
        self,
        colors: list[str],
        model: str = "fake-noise-model",
        cost_usd: float = 0.01,
    ) -> None:
        self._colors = colors
        self.model = model
        self.cost_usd = cost_usd
        self.calls = 0

    def generate(
        self,
        svp: Any,
        quality_mode: str = "normal",
        reference_image_path: Path | None = None,
    ) -> ImageResult:
        color = self._colors[self.calls % len(self._colors)]
        self.calls += 1
        return ImageResult(
            png_bytes=valid_png_bytes(color),
            cost_usd=self.cost_usd,
            elapsed_sec=0.01,
            raw_prompt="fake prompt",
            model=self.model,
            backend="fake",
            aspect_ratio="16:9",
            native_size_or_resolution="1K",
            was_aspect_coerced=False,
        )


def _write_svp(tmp_path: Path, name: str = "svp.json") -> Path:
    svp_path = tmp_path / name
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    return svp_path


# ---------------------------------------------------------------------------
# noise.run_noise_probe: aggregation math
# ---------------------------------------------------------------------------


def test_n2_produces_exactly_one_pair(tmp_path: Path) -> None:
    svp_path = _write_svp(tmp_path)
    backend = _FakeNoiseImageBackend(["white", "black"])
    output_dir = tmp_path / "out"

    report = run_noise_probe(
        svp_path=svp_path, backend="gemini", n=2, output_dir=output_dir, image_backend=backend
    )

    assert report.n == 2
    assert len(report.pairs) == 1
    assert (report.pairs[0].index_a, report.pairs[0].index_b) == (0, 1)
    assert report.pixel_rms_mean == report.pixel_rms_min == report.pixel_rms_max


def test_n3_produces_three_pairs_with_correct_indices(tmp_path: Path) -> None:
    svp_path = _write_svp(tmp_path)
    backend = _FakeNoiseImageBackend(["white", "black", "red"])
    output_dir = tmp_path / "out"

    report = run_noise_probe(
        svp_path=svp_path, backend="gemini", n=3, output_dir=output_dir, image_backend=backend
    )

    assert report.n == 3
    assert len(report.pairs) == 3
    pair_indices = {(pair.index_a, pair.index_b) for pair in report.pairs}
    assert pair_indices == {(0, 1), (0, 2), (1, 2)}


def test_aggregates_match_independent_pixel_metrics_recomputation(tmp_path: Path) -> None:
    svp_path = _write_svp(tmp_path)
    backend = _FakeNoiseImageBackend(["white", "black", "red"])
    output_dir = tmp_path / "out"

    report = run_noise_probe(
        svp_path=svp_path, backend="gemini", n=3, output_dir=output_dir, image_backend=backend
    )

    expected = [
        pixel_metrics(output_dir / f"probe_{i:02d}.png", output_dir / f"probe_{j:02d}.png")
        for i, j in ((0, 1), (0, 2), (1, 2))
    ]
    expected_rms = [m["pixel_rms"] for m in expected]
    expected_delta = [m["mean_rgb_delta"] for m in expected]

    assert report.pixel_rms_mean == pytest.approx(sum(expected_rms) / 3, abs=1e-4)
    assert report.pixel_rms_min == pytest.approx(min(expected_rms), abs=1e-4)
    assert report.pixel_rms_max == pytest.approx(max(expected_rms), abs=1e-4)
    assert report.mean_rgb_delta_mean == pytest.approx(sum(expected_delta) / 3, abs=1e-4)
    assert report.mean_rgb_delta_min == pytest.approx(min(expected_delta), abs=1e-4)
    assert report.mean_rgb_delta_max == pytest.approx(max(expected_delta), abs=1e-4)
    # non-trivial: distinct colors must not collapse to a zero noise floor
    assert report.pixel_rms_mean > 0


def test_identical_images_yield_zero_noise_floor(tmp_path: Path) -> None:
    svp_path = _write_svp(tmp_path)
    backend = _FakeNoiseImageBackend(["white"])  # every call returns the same bytes
    output_dir = tmp_path / "out"

    report = run_noise_probe(
        svp_path=svp_path, backend="gemini", n=3, output_dir=output_dir, image_backend=backend
    )

    assert report.pixel_rms_mean == 0.0
    assert report.pixel_rms_max == 0.0
    assert report.mean_rgb_delta_mean == 0.0


# ---------------------------------------------------------------------------
# cost aggregation
# ---------------------------------------------------------------------------


def test_cost_aggregation(tmp_path: Path) -> None:
    svp_path = _write_svp(tmp_path)
    backend = _FakeNoiseImageBackend(["white", "black"], cost_usd=0.08)
    output_dir = tmp_path / "out"

    report = run_noise_probe(
        svp_path=svp_path, backend="gemini", n=4, output_dir=output_dir, image_backend=backend
    )

    assert report.per_image_cost_usd == [0.08, 0.08, 0.08, 0.08]
    assert report.total_cost_usd == pytest.approx(0.32, abs=1e-4)
    assert backend.calls == 4


# ---------------------------------------------------------------------------
# metadata / fail-fast at the function level
# ---------------------------------------------------------------------------


def test_metadata_fields(tmp_path: Path) -> None:
    svp_path = _write_svp(tmp_path)
    backend = _FakeNoiseImageBackend(["white", "black"])
    output_dir = tmp_path / "out"

    report = run_noise_probe(
        svp_path=svp_path, backend="openai", n=2, output_dir=output_dir, image_backend=backend
    )

    assert report.backend == "openai"
    assert report.model == "fake-noise-model"
    assert report.svp_path == str(svp_path)
    assert len(report.svp_sha256) == 64  # sha256 hex digest length
    assert report.timestamp_utc.endswith("Z")


def test_n_below_2_raises_value_error(tmp_path: Path) -> None:
    svp_path = _write_svp(tmp_path)
    with pytest.raises(ValueError, match="n must be >= 2"):
        run_noise_probe(svp_path=svp_path, backend="gemini", n=1, output_dir=tmp_path / "out")


def test_missing_svp_file_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SVP file not found"):
        run_noise_probe(
            svp_path=tmp_path / "does-not-exist.json",
            backend="gemini",
            n=2,
            output_dir=tmp_path / "out",
        )


# ---------------------------------------------------------------------------
# noise_floor.json schema / verdict-free contract
# ---------------------------------------------------------------------------


def test_write_noise_floor_report_schema_has_no_verdict(tmp_path: Path) -> None:
    svp_path = _write_svp(tmp_path)
    backend = _FakeNoiseImageBackend(["white", "black"])
    output_dir = tmp_path / "out"

    report = run_noise_probe(
        svp_path=svp_path, backend="gemini", n=2, output_dir=output_dir, image_backend=backend
    )
    report_path = write_noise_floor_report(report, output_dir)

    assert report_path == output_dir / "noise_floor.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))

    for required_key in (
        "schema_version",
        "svp_path",
        "svp_sha256",
        "backend",
        "model",
        "n",
        "timestamp_utc",
        "per_image_cost_usd",
        "total_cost_usd",
        "pairs",
        "pixel_rms_mean",
        "pixel_rms_min",
        "pixel_rms_max",
        "mean_rgb_delta_mean",
        "mean_rgb_delta_min",
        "mean_rgb_delta_max",
    ):
        assert required_key in data

    assert len(data["pairs"]) == 1
    assert set(data["pairs"][0]) == {"index_a", "index_b", "pixel_rms", "mean_rgb_delta"}

    # verdict/threshold/pass-fail language must not appear anywhere (AC3: instrument, no verdict)
    forbidden_keys = {"verdict", "pass", "fail", "passed", "failed", "threshold", "status"}
    assert forbidden_keys.isdisjoint(data)


# ---------------------------------------------------------------------------
# corpus manifest loading
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path: Path, payload: dict[str, Any], name: str = "corpus.json") -> Path:
    manifest_path = tmp_path / name
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def test_load_corpus_manifest_resolves_relative_svp_paths(tmp_path: Path) -> None:
    svp_a = _write_svp(tmp_path, "a.json")
    svp_b = _write_svp(tmp_path, "b.json")
    manifest_path = _write_manifest(
        tmp_path,
        {"backend": "gemini", "n": 2, "svp_files": ["a.json", "b.json"]},
    )

    manifest = load_corpus_manifest(manifest_path)

    assert isinstance(manifest, NoiseCorpusManifest)
    assert manifest.backend == "gemini"
    assert manifest.n == 2
    assert manifest.svp_files == [svp_a, svp_b]


def test_load_corpus_manifest_rejects_unknown_key(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        {"backend": "gemini", "n": 2, "svp_files": ["a.json"], "extra_key": "nope"},
    )
    with pytest.raises(ValueError, match="unknown corpus manifest key"):
        load_corpus_manifest(manifest_path)


def test_load_corpus_manifest_rejects_missing_key(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, {"backend": "gemini", "n": 2})
    with pytest.raises(ValueError, match="missing required key"):
        load_corpus_manifest(manifest_path)


def test_load_corpus_manifest_rejects_n_below_2(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path, {"backend": "gemini", "n": 1, "svp_files": ["a.json"]}
    )
    with pytest.raises(ValueError, match="'n' must be an integer >= 2"):
        load_corpus_manifest(manifest_path)


def test_load_corpus_manifest_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="corpus manifest not found"):
        load_corpus_manifest(tmp_path / "missing.json")


def test_real_sample_corpus_manifest_loads(tmp_path: Path) -> None:
    samples_dir = Path(__file__).resolve().parent / "samples"
    manifest = load_corpus_manifest(samples_dir / "probe_corpus.json")

    assert manifest.backend == "gemini"
    assert manifest.n == 3
    assert len(manifest.svp_files) == 3
    for svp_file in manifest.svp_files:
        assert svp_file.is_file()


# ---------------------------------------------------------------------------
# corpus.run_corpus_noise_probe: batch loop
# ---------------------------------------------------------------------------


def test_run_corpus_noise_probe_writes_one_report_per_svp(tmp_path: Path) -> None:
    svp_a = _write_svp(tmp_path, "a.json")
    svp_b = _write_svp(tmp_path, "b.json")
    manifest = NoiseCorpusManifest(backend="gemini", n=2, svp_files=[svp_a, svp_b])
    backend = _FakeNoiseImageBackend(["white", "black"])
    output_dir = tmp_path / "out"

    reports = run_corpus_noise_probe(
        manifest=manifest, output_dir=output_dir, image_backend=backend
    )

    assert len(reports) == 2
    assert all(isinstance(report, NoiseFloorReport) for report in reports)
    assert (output_dir / "a" / "noise_floor.json").is_file()
    assert (output_dir / "b" / "noise_floor.json").is_file()
    # both SVPs generated through the same shared backend instance
    assert backend.calls == 4


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _set_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_mod, "load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("FAL_KEY", "fal-test")


def _clear_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_mod, "load_dotenv", lambda: None)
    for key in (
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "FAL_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_cli_probe_noise_single_svp_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    svp_path = _write_svp(tmp_path)
    output_dir = tmp_path / "out"
    calls: list[dict[str, Any]] = []

    def _fake_run_noise_probe(*, svp_path, backend, n, output_dir):
        calls.append({"svp_path": svp_path, "backend": backend, "n": n, "output_dir": output_dir})
        fake_backend = _FakeNoiseImageBackend(["white", "black"])
        return run_noise_probe(
            svp_path=svp_path,
            backend=backend,
            n=n,
            output_dir=output_dir,
            image_backend=fake_backend,
        )

    monkeypatch.setattr(cli_mod, "run_noise_probe", _fake_run_noise_probe)

    result = runner.invoke(
        app,
        [
            "probe-noise",
            "--from-svp",
            str(svp_path),
            "--backend",
            "openai",
            "-n",
            "2",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["backend"] == "openai"
    assert calls[0]["n"] == 2
    assert "A/A noise floor probe complete" in result.output
    assert (output_dir / "noise_floor.json").is_file()


def test_cli_probe_noise_requires_from_svp_or_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_keys(monkeypatch)
    result = runner.invoke(app, ["probe-noise"])
    assert result.exit_code == 1
    assert "requires --from-svp or --corpus" in result.output


def test_cli_probe_noise_rejects_both_from_svp_and_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_required_keys(monkeypatch)
    svp_path = _write_svp(tmp_path)
    manifest_path = _write_manifest(
        tmp_path, {"backend": "gemini", "n": 2, "svp_files": ["svp.json"]}
    )

    result = runner.invoke(
        app,
        [
            "probe-noise",
            "--from-svp",
            str(svp_path),
            "--corpus",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 1
    assert "not both" in result.output


def test_cli_probe_noise_n_below_2_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_required_keys(monkeypatch)
    svp_path = _write_svp(tmp_path)

    result = runner.invoke(
        app,
        ["probe-noise", "--from-svp", str(svp_path), "-n", "1"],
    )

    assert result.exit_code == 2
    assert "must be >= 2" in result.output


def test_cli_probe_noise_missing_svp_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_required_keys(monkeypatch)
    result = runner.invoke(
        app,
        ["probe-noise", "--from-svp", str(tmp_path / "missing.json")],
    )
    assert result.exit_code == 1
    assert "SVP file not found" in result.output


def test_cli_probe_noise_missing_image_key_fails_gracefully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = _write_svp(tmp_path)

    result = runner.invoke(
        app,
        ["probe-noise", "--from-svp", str(svp_path), "--backend", "gemini"],
    )

    assert result.exit_code == 1
    assert "GOOGLE_API_KEY" in result.output


def test_cli_probe_noise_corpus_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_keys(monkeypatch)
    svp_a = _write_svp(tmp_path, "a.json")
    svp_b = _write_svp(tmp_path, "b.json")
    manifest_path = _write_manifest(
        tmp_path, {"backend": "gemini", "n": 2, "svp_files": ["a.json", "b.json"]}
    )
    output_dir = tmp_path / "out"
    calls: list[dict[str, Any]] = []

    def _fake_run_corpus_noise_probe(*, manifest, output_dir):
        calls.append({"manifest": manifest, "output_dir": output_dir})
        fake_backend = _FakeNoiseImageBackend(["white", "black"])
        return run_corpus_noise_probe(
            manifest=manifest, output_dir=output_dir, image_backend=fake_backend
        )

    monkeypatch.setattr(cli_mod, "run_corpus_noise_probe", _fake_run_corpus_noise_probe)

    result = runner.invoke(
        app,
        ["probe-noise", "--corpus", str(manifest_path), "--output-dir", str(output_dir)],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["manifest"].svp_files == [svp_a, svp_b]
    assert "corpus probe complete: 2 SVP file(s)" in result.output
    assert (output_dir / "a" / "noise_floor.json").is_file()
    assert (output_dir / "b" / "noise_floor.json").is_file()


def test_cli_probe_noise_corpus_unknown_key_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_required_keys(monkeypatch)
    manifest_path = _write_manifest(
        tmp_path,
        {"backend": "gemini", "n": 2, "svp_files": ["a.json"], "bogus": True},
    )

    result = runner.invoke(app, ["probe-noise", "--corpus", str(manifest_path)])

    assert result.exit_code == 1
    assert "unknown corpus manifest key" in result.output


def test_cli_probe_noise_help_lists_options() -> None:
    result = runner.invoke(app, ["probe-noise", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    for option in ("--from-svp", "--corpus", "--backend", "--output-dir"):
        assert option in output
