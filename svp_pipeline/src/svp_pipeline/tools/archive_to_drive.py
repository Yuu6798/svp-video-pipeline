"""Compatibility wrapper for `python -m svp_pipeline.tools.archive_to_drive`."""

from __future__ import annotations

from tools.archive_to_drive import *  # noqa: F403
from tools.archive_to_drive import main

if __name__ == "__main__":
    raise SystemExit(main())
