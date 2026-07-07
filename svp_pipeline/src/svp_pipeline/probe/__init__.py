"""A/A noise-floor probe harness (PROBE-0).

Measures per-SVP regeneration noise (image backend non-determinism) by
generating N images from a single fixed SVP and comparing every pair with
the existing pixel metrics. This is an instrument, not a verdict: it
reports numbers (mean/min/max pairwise distance) without judging whether a
result is "good" or "bad". See ``docs/measurement.md`` for methodology.
"""

from __future__ import annotations
