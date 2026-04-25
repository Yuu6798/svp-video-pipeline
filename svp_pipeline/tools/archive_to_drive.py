"""Repository-local wrapper for the packaged Drive archive helper."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from svp_pipeline.tools.archive_to_drive import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
