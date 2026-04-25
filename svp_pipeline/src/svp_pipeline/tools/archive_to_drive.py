"""Compatibility wrapper for `python -m svp_pipeline.tools.archive_to_drive`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HELPER_PATH = Path(__file__).resolve().parents[3] / "tools" / "archive_to_drive.py"
_SPEC = importlib.util.spec_from_file_location("_svp_archive_to_drive", _HELPER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load archive helper from {_HELPER_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ArchiveResult = _MODULE.ArchiveResult
archive_run = _MODULE.archive_run
authenticate = _MODULE.authenticate
detect_artifacts = _MODULE.detect_artifacts
ensure_drive_folder = _MODULE.ensure_drive_folder
filter_unuploaded = _MODULE.filter_unuploaded
load_log = _MODULE.load_log
main = _MODULE.main
run_id_to_date = _MODULE.run_id_to_date
update_log_with_urls = _MODULE.update_log_with_urls
upload_file = _MODULE.upload_file


if __name__ == "__main__":
    raise SystemExit(main())
