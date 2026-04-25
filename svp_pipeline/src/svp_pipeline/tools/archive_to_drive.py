"""Archive generated SVP image/video artifacts to Google Drive.

This helper keeps text artifacts (`svp.json`, `log.json`) in the repository and
uploads only binary outputs to Drive. After upload, Drive URLs are written back
to `log.json` under the flat `drive_urls` field.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import mimetypes
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "svp-pipeline"
DEFAULT_CREDENTIALS_PATH = DEFAULT_CONFIG_DIR / "google-credentials.json"
DEFAULT_TOKEN_PATH = DEFAULT_CONFIG_DIR / "google-token.json"
DEFAULT_DRIVE_ROOT = "SVP Archive"
RUN_ID_RE = re.compile(r"^(?P<timestamp>\d{8}-\d{6})(?:-\d{2})?$")

UPLOAD_ARTIFACTS: dict[str, str] = {
    "image": "image.png",
    "video": "video.mp4",
    "character_green": "character_green.png",
    "background_clean": "background_clean.png",
    "composite": "composite.png",
}


@dataclass
class ArchiveResult:
    """Archive run result."""

    uploaded_files: dict[str, str]
    skipped_files: list[str]
    drive_folder_url: str
    log_path: Path
    already_archived: bool


def load_log(run_dir: Path) -> dict[str, Any]:
    """Load `<run_dir>/log.json`."""

    log_path = run_dir / "log.json"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not log_path.exists():
        raise FileNotFoundError(f"log.json not found: {log_path}")
    return json.loads(log_path.read_text(encoding="utf-8"))


def detect_artifacts(run_dir: Path) -> dict[str, Path]:
    """Detect local binary artifacts that should be uploaded to Drive."""

    artifacts: dict[str, Path] = {}
    for key, filename in UPLOAD_ARTIFACTS.items():
        path = run_dir / filename
        if path.is_file():
            artifacts[key] = path
    return artifacts


def filter_unuploaded(artifacts: dict[str, Path], log: dict[str, Any]) -> dict[str, Path]:
    """Return artifacts not yet present in `log["drive_urls"]`."""

    drive_urls = log.get("drive_urls") or {}
    return {key: path for key, path in artifacts.items() if key not in drive_urls}


def authenticate(credentials_path: Path, token_path: Path):
    """Authenticate through OAuth2 desktop flow and return Google credentials."""

    if not credentials_path.exists():
        raise FileNotFoundError(
            "Google OAuth credentials not found: "
            f"{credentials_path}. Create an OAuth Desktop client in Google Cloud "
            "Console, enable Drive API, and save it at this path."
        )

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if credentials and credentials.valid:
        return credentials

    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
        except Exception as exc:  # pragma: no cover - concrete type is library-specific
            raise RuntimeError(
                "Failed to refresh Google OAuth token. Delete "
                f"{token_path} and re-run the command to authenticate again."
            ) from exc
    else:
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
        credentials = flow.run_local_server(port=0)

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    return credentials


def build_drive_service(credentials):
    """Build a Google Drive v3 service."""

    from googleapiclient.discovery import build

    return build("drive", "v3", credentials=credentials)


def ensure_drive_folder(service, drive_root: str, run_id: str) -> tuple[str, str]:
    """Ensure `<drive_root>/by-date/YYYY-MM-DD/<run_id>/` exists in Drive."""

    date_name = run_id_to_date(run_id)
    root_id, _ = _ensure_child_folder(service, drive_root, parent_id=None)
    by_date_id, _ = _ensure_child_folder(service, "by-date", parent_id=root_id)
    date_id, _ = _ensure_child_folder(service, date_name, parent_id=by_date_id)
    return _ensure_child_folder(service, run_id, parent_id=date_id)


def upload_file(service, local_path: Path, parent_folder_id: str) -> str:
    """Upload a file and return its Drive `webViewLink`."""

    from googleapiclient.http import MediaFileUpload

    mime_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
    body = {"name": local_path.name, "parents": [parent_folder_id]}
    request = service.files().create(
        body=body,
        media_body=media,
        fields="id,webViewLink",
    )
    result = _execute_with_retry(request, f"upload {local_path.name}")
    return result["webViewLink"]


def update_log_with_urls(
    log_path: Path,
    new_urls: dict[str, str],
    drive_folder_url: str,
) -> None:
    """Merge Drive URLs into `log.json` using an atomic replace."""

    log = json.loads(log_path.read_text(encoding="utf-8"))
    drive_urls = dict(log.get("drive_urls") or {})
    drive_urls.update(new_urls)
    if drive_folder_url:
        drive_urls["drive_folder"] = drive_folder_url
    drive_urls["uploaded_at"] = _utc_now_iso()
    log["drive_urls"] = drive_urls

    tmp_path = log_path.with_name(f"{log_path.name}.tmp")
    tmp_path.write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, log_path)


def archive_run(
    run_dir: Path,
    credentials_path: Path = DEFAULT_CREDENTIALS_PATH,
    token_path: Path = DEFAULT_TOKEN_PATH,
    drive_root: str = DEFAULT_DRIVE_ROOT,
    dry_run: bool = False,
) -> ArchiveResult:
    """Archive a single SVP run directory to Google Drive."""

    run_dir = run_dir.resolve()
    log = load_log(run_dir)
    artifacts = detect_artifacts(run_dir)
    pending = filter_unuploaded(artifacts, log)
    existing_drive_urls = log.get("drive_urls") or {}
    skipped = [key for key in artifacts if key in existing_drive_urls]
    log_path = run_dir / "log.json"

    if artifacts and not pending and "drive_folder" in existing_drive_urls:
        return ArchiveResult(
            uploaded_files={},
            skipped_files=skipped,
            drive_folder_url=existing_drive_urls["drive_folder"],
            log_path=log_path,
            already_archived=True,
        )

    run_id = run_dir.name
    if dry_run:
        return ArchiveResult(
            uploaded_files={},
            skipped_files=skipped,
            drive_folder_url=_format_drive_path(drive_root, run_id),
            log_path=log_path,
            already_archived=False,
        )

    credentials = authenticate(credentials_path, token_path)
    service = build_drive_service(credentials)
    folder_id, folder_url = ensure_drive_folder(service, drive_root, run_id)

    uploaded: dict[str, str] = {}
    for key, path in pending.items():
        url = upload_file(service, path, folder_id)
        uploaded[key] = url
        update_log_with_urls(log_path, {key: url}, folder_url)

    if not pending:
        update_log_with_urls(log_path, {}, folder_url)

    return ArchiveResult(
        uploaded_files=uploaded,
        skipped_files=skipped,
        drive_folder_url=folder_url,
        log_path=log_path,
        already_archived=False,
    )


def run_id_to_date(run_id: str) -> str:
    """Convert a pipeline run directory name to `YYYY-MM-DD`.

    The pipeline may append `-NN` when multiple runs start in the same second,
    for example `20260425-140453-01`.
    """

    match = RUN_ID_RE.fullmatch(run_id)
    if not match:
        raise ValueError(
            f"Invalid run directory name {run_id!r}; expected YYYYMMDD-HHMMSS "
            "or YYYYMMDD-HHMMSS-NN."
        )
    try:
        return dt.datetime.strptime(
            match.group("timestamp"),
            "%Y%m%d-%H%M%S",
        ).date().isoformat()
    except ValueError as exc:
        raise ValueError(
            f"Invalid run directory date in {run_id!r}; expected a real calendar date."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = archive_run(
            run_dir=args.run_dir,
            credentials_path=args.credentials,
            token_path=args.token,
            drive_root=args.drive_root,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        if args.verbose:
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_summary(result, dry_run=args.dry_run)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archive_to_drive",
        description="Archive SVP image/video artifacts to Google Drive.",
    )
    parser.add_argument("run_dir", type=Path, help="Run directory such as out/20260425-140453")
    parser.add_argument(
        "--credentials",
        type=Path,
        default=DEFAULT_CREDENTIALS_PATH,
        help=f"OAuth credentials JSON (default: {DEFAULT_CREDENTIALS_PATH})",
    )
    parser.add_argument(
        "--token",
        type=Path,
        default=DEFAULT_TOKEN_PATH,
        help=f"OAuth token JSON (default: {DEFAULT_TOKEN_PATH})",
    )
    parser.add_argument(
        "--drive-root",
        default=DEFAULT_DRIVE_ROOT,
        help=f"Drive root folder name (default: {DEFAULT_DRIVE_ROOT!r})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    parser.add_argument("--verbose", action="store_true", help="Show traceback on errors")
    return parser


def _print_summary(result: ArchiveResult, dry_run: bool) -> None:
    if result.already_archived:
        print("Already archived; nothing to upload.")
        print(f"Drive folder: {result.drive_folder_url}")
        return

    if dry_run:
        print("Dry run: no files uploaded and log.json was not modified.")
    elif result.uploaded_files:
        print(f"Uploaded {len(result.uploaded_files)} file(s).")
    else:
        print("No new files to upload.")

    if result.uploaded_files:
        for key, url in result.uploaded_files.items():
            print(f"- {key}: {url}")
    if result.skipped_files:
        print(f"Skipped already uploaded: {', '.join(result.skipped_files)}")
    print(f"Drive folder: {result.drive_folder_url}")
    print(f"Updated log: {result.log_path}")


def _ensure_child_folder(service, name: str, parent_id: str | None) -> tuple[str, str]:
    existing = _find_folder(service, name, parent_id)
    if existing:
        folder_id = existing["id"]
        return folder_id, existing.get("webViewLink") or _folder_url(folder_id)

    body: dict[str, Any] = {"name": name, "mimeType": FOLDER_MIME_TYPE}
    if parent_id:
        body["parents"] = [parent_id]

    request = service.files().create(body=body, fields="id,webViewLink")
    created = _execute_with_retry(request, f"create folder {name}")
    folder_id = created["id"]
    return folder_id, created.get("webViewLink") or _folder_url(folder_id)


def _find_folder(service, name: str, parent_id: str | None) -> dict[str, Any] | None:
    conditions = [
        f"name = '{_escape_drive_query_value(name)}'",
        f"mimeType = '{FOLDER_MIME_TYPE}'",
        "trashed = false",
    ]
    if parent_id:
        conditions.append(f"'{parent_id}' in parents")
    else:
        conditions.append("'root' in parents")
    query = " and ".join(conditions)
    request = service.files().list(
        q=query,
        fields="files(id,name,webViewLink)",
        pageSize=1,
    )
    result = _execute_with_retry(request, f"find folder {name}")
    files = result.get("files", [])
    return files[0] if files else None


def _execute_with_retry(request, description: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            return request.execute()
        except Exception as exc:  # pragma: no cover - concrete type is library-specific
            last_error = exc
            if attempt == 0:
                time.sleep(1)
    raise RuntimeError(f"Drive API failed during {description}") from last_error


def _escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _folder_url(folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}"


def _format_drive_path(drive_root: str, run_id: str) -> str:
    return f"{drive_root}/by-date/{run_id_to_date(run_id)}/{run_id}"


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
