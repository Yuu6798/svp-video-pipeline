"""Tests for Google Drive archive helper."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import svp_pipeline.tools.archive_to_drive as archive_mod
from tests.fixtures.mock_drive import (
    MockDriveService,
    patch_authenticate,
    patch_drive_service_build,
)


def _make_run_dir(
    tmp_path: Path,
    *,
    artifacts: tuple[str, ...] = ("image.png", "video.mp4"),
    drive_urls: dict[str, str] | None = None,
) -> Path:
    run_dir = tmp_path / "20260425-140453"
    run_dir.mkdir()
    for filename in artifacts:
        (run_dir / filename).write_bytes(b"data")
    log: dict[str, object] = {
        "user_prompt": "test",
        "outputs": {"svp": "svp.json", "image": "image.png"},
    }
    if drive_urls is not None:
        log["drive_urls"] = drive_urls
    (run_dir / "log.json").write_text(json.dumps(log), encoding="utf-8")
    (run_dir / "svp.json").write_text("{}", encoding="utf-8")
    return run_dir


def test_detect_image_video_only(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)

    artifacts = archive_mod.detect_artifacts(run_dir)

    assert set(artifacts) == {"image", "video"}
    assert artifacts["image"].name == "image.png"
    assert artifacts["video"].name == "video.mp4"


def test_detect_with_split_composite(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path,
        artifacts=(
            "image.png",
            "video.mp4",
            "character_green.png",
            "background_clean.png",
            "composite.png",
        ),
    )

    assert set(archive_mod.detect_artifacts(run_dir)) == {
        "image",
        "video",
        "character_green",
        "background_clean",
        "composite",
    }


def test_detect_no_video(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, artifacts=("image.png",))

    assert set(archive_mod.detect_artifacts(run_dir)) == {"image"}


def test_filter_partial(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, drive_urls={"image": "https://drive/image"})
    artifacts = archive_mod.detect_artifacts(run_dir)
    log = archive_mod.load_log(run_dir)

    assert set(archive_mod.filter_unuploaded(artifacts, log)) == {"video"}


def test_filter_all_uploaded(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path,
        drive_urls={
            "image": "https://drive/image",
            "video": "https://drive/video",
        },
    )

    assert archive_mod.filter_unuploaded(
        archive_mod.detect_artifacts(run_dir),
        archive_mod.load_log(run_dir),
    ) == {}


def test_run_id_date_extraction() -> None:
    assert archive_mod.run_id_to_date("20260425-140453") == "2026-04-25"


def test_creates_new_folder_tree() -> None:
    service = MockDriveService()

    folder_id, folder_url = archive_mod.ensure_drive_folder(
        service,
        "SVP Archive",
        "20260425-140453",
    )

    assert folder_id.startswith("mock-id-")
    assert folder_url.startswith("https://drive.google.com/drive/folders/")
    assert [folder["name"] for folder in service.created_folders] == [
        "SVP Archive",
        "by-date",
        "2026-04-25",
        "20260425-140453",
    ]


def test_reuses_existing_folder() -> None:
    service = MockDriveService()
    root_id = service.register_folder("SVP Archive", None, "root-id")
    by_date_id = service.register_folder("by-date", root_id, "by-date-id")
    date_id = service.register_folder("2026-04-25", by_date_id, "date-id")
    service.register_folder("20260425-140453", date_id, "run-id")

    folder_id, _ = archive_mod.ensure_drive_folder(service, "SVP Archive", "20260425-140453")

    assert folder_id == "run-id"
    assert service.created_folders == []


def test_root_folder_lookup_is_limited_to_drive_root_parent() -> None:
    service = MockDriveService()
    nested_parent_id = service.register_folder("Other", None, "other-id")
    service.register_folder("SVP Archive", nested_parent_id, "nested-archive-id")

    folder_id, _ = archive_mod.ensure_drive_folder(service, "SVP Archive", "20260425-140453")

    assert folder_id != "nested-archive-id"
    assert service.created_folders[0]["name"] == "SVP Archive"


def test_adds_drive_urls_field(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    log_path = run_dir / "log.json"

    archive_mod.update_log_with_urls(log_path, {"image": "https://drive/image"}, "folder-url")

    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log["drive_urls"]["image"] == "https://drive/image"
    assert log["drive_urls"]["drive_folder"] == "folder-url"
    assert log["drive_urls"]["uploaded_at"].endswith("Z")


def test_merges_with_existing_drive_urls(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, drive_urls={"image": "https://drive/image"})
    log_path = run_dir / "log.json"

    archive_mod.update_log_with_urls(log_path, {"video": "https://drive/video"}, "folder-url")

    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log["drive_urls"]["image"] == "https://drive/image"
    assert log["drive_urls"]["video"] == "https://drive/video"


def test_atomic_write_preserves_original_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _make_run_dir(tmp_path)
    log_path = run_dir / "log.json"
    original = log_path.read_text(encoding="utf-8")

    def fail_replace(_src: Path, _dst: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(archive_mod.os, "replace", fail_replace)

    with pytest.raises(OSError):
        archive_mod.update_log_with_urls(log_path, {"image": "url"}, "folder-url")

    assert log_path.read_text(encoding="utf-8") == original


def test_full_archive_image_and_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _make_run_dir(tmp_path)
    service = MockDriveService()
    patch_authenticate(monkeypatch, archive_mod)
    patch_drive_service_build(monkeypatch, archive_mod, service)
    monkeypatch.setattr(
        archive_mod,
        "upload_file",
        lambda _service, path, _folder_id: f"https://drive/{path.name}",
    )

    result = archive_mod.archive_run(run_dir)

    assert result.uploaded_files == {
        "image": "https://drive/image.png",
        "video": "https://drive/video.mp4",
    }
    log = json.loads((run_dir / "log.json").read_text(encoding="utf-8"))
    assert log["drive_urls"]["image"] == "https://drive/image.png"
    assert log["drive_urls"]["video"] == "https://drive/video.mp4"
    assert "svp" not in log["drive_urls"]
    assert "log" not in log["drive_urls"]


def test_idempotent_already_archived(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _make_run_dir(
        tmp_path,
        drive_urls={
            "image": "https://drive/image",
            "video": "https://drive/video",
            "drive_folder": "https://drive/folder",
        },
    )
    auth = MagicMock()
    monkeypatch.setattr(archive_mod, "authenticate", auth)

    result = archive_mod.archive_run(run_dir)

    assert result.already_archived is True
    assert result.uploaded_files == {}
    auth.assert_not_called()


def test_resume_partial_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _make_run_dir(tmp_path, drive_urls={"image": "https://drive/image"})
    service = MockDriveService()
    patch_authenticate(monkeypatch, archive_mod)
    patch_drive_service_build(monkeypatch, archive_mod, service)
    uploaded_names: list[str] = []

    def fake_upload(_service, path: Path, _folder_id: str) -> str:
        uploaded_names.append(path.name)
        return f"https://drive/{path.name}"

    monkeypatch.setattr(archive_mod, "upload_file", fake_upload)

    result = archive_mod.archive_run(run_dir)

    assert uploaded_names == ["video.mp4"]
    assert result.skipped_files == ["image"]


def test_dry_run_does_not_upload_or_modify_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _make_run_dir(tmp_path)
    original = (run_dir / "log.json").read_text(encoding="utf-8")
    auth = MagicMock()
    monkeypatch.setattr(archive_mod, "authenticate", auth)

    result = archive_mod.archive_run(run_dir, dry_run=True)

    assert result.uploaded_files == {}
    assert (run_dir / "log.json").read_text(encoding="utf-8") == original
    auth.assert_not_called()


def test_missing_log_json_errors(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260425-140453"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        archive_mod.load_log(run_dir)


def test_authenticate_handles_missing_credentials(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Google OAuth credentials"):
        archive_mod.authenticate(tmp_path / "missing.json", tmp_path / "token.json")


def test_authenticate_uses_existing_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_google_auth_modules(monkeypatch, valid=True)
    credentials = tmp_path / "credentials.json"
    token = tmp_path / "token.json"
    credentials.write_text("{}", encoding="utf-8")
    token.write_text("{}", encoding="utf-8")

    result = archive_mod.authenticate(credentials, token)

    assert result.valid is True


def test_authenticate_refreshes_expired_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_credentials = _install_fake_google_auth_modules(monkeypatch, valid=False, expired=True)
    credentials = tmp_path / "credentials.json"
    token = tmp_path / "token.json"
    credentials.write_text("{}", encoding="utf-8")
    token.write_text("{}", encoding="utf-8")

    archive_mod.authenticate(credentials, token)

    assert fake_credentials.refresh_called is True


def test_module_help_runs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        archive_mod.main(["--help"])

    assert exc_info.value.code == 0
    assert "Archive SVP image/video artifacts" in capsys.readouterr().out


def _install_fake_google_auth_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    valid: bool,
    expired: bool = False,
):
    class FakeCredentials:
        def __init__(self) -> None:
            self.valid = valid
            self.expired = expired
            self.refresh_token = "refresh-token"
            self.refresh_called = False

        @classmethod
        def from_authorized_user_file(cls, _path: str, _scopes: list[str]):
            return fake_credentials

        def refresh(self, _request) -> None:
            self.refresh_called = True
            self.valid = True
            self.expired = False

        def to_json(self) -> str:
            return "{}"

    class FakeRequest:
        pass

    fake_credentials = FakeCredentials()

    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.oauth2", types.ModuleType("google.oauth2"))
    monkeypatch.setitem(
        sys.modules,
        "google.oauth2.credentials",
        types.SimpleNamespace(Credentials=FakeCredentials),
    )
    monkeypatch.setitem(sys.modules, "google.auth", types.ModuleType("google.auth"))
    monkeypatch.setitem(
        sys.modules,
        "google.auth.transport",
        types.ModuleType("google.auth.transport"),
    )
    monkeypatch.setitem(
        sys.modules,
        "google.auth.transport.requests",
        types.SimpleNamespace(Request=FakeRequest),
    )
    monkeypatch.setitem(
        sys.modules,
        "google_auth_oauthlib",
        types.ModuleType("google_auth_oauthlib"),
    )
    monkeypatch.setitem(
        sys.modules,
        "google_auth_oauthlib.flow",
        types.SimpleNamespace(InstalledAppFlow=object),
    )
    return fake_credentials
