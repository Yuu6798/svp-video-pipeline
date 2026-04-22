"""Shared pytest fixtures for the svp_pipeline schema test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def samples_dir() -> Path:
    return Path(__file__).parent / "samples"
