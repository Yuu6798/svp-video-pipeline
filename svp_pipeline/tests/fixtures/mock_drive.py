"""Google Drive API test doubles for archive_to_drive tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


class MockDriveService:
    """Minimal Drive API service mock."""

    def __init__(self) -> None:
        self.uploaded_files: list[dict[str, Any]] = []
        self.created_folders: list[dict[str, Any]] = []
        self.existing_folders: dict[tuple[str | None, str], dict[str, str]] = {}
        self._next_id = 1000
        self.fail_next_execute = False

    def _new_id(self) -> str:
        self._next_id += 1
        return f"mock-id-{self._next_id}"

    def files(self) -> MockFilesResource:
        return MockFilesResource(self)

    def register_folder(
        self,
        name: str,
        parent_id: str | None,
        folder_id: str | None = None,
    ) -> str:
        folder_id = folder_id or self._new_id()
        self.existing_folders[(parent_id, name)] = {
            "id": folder_id,
            "name": name,
            "webViewLink": f"https://drive.google.com/drive/folders/{folder_id}",
        }
        return folder_id


class MockFilesResource:
    """Minimal `files()` resource mock."""

    def __init__(self, service: MockDriveService) -> None:
        self.service = service

    def create(self, body: dict[str, Any], media_body=None, fields: str = "id") -> MockExecute:
        def run() -> dict[str, Any]:
            if self.service.fail_next_execute:
                self.service.fail_next_execute = False
                raise RuntimeError("transient")

            file_id = self.service._new_id()
            name = body["name"]
            parent_id = (body.get("parents") or [None])[0]
            web_link = (
                f"https://drive.google.com/drive/folders/{file_id}"
                if body.get("mimeType") == "application/vnd.google-apps.folder"
                else f"https://drive.google.com/file/d/{file_id}/view"
            )
            payload = {"id": file_id, "name": name, "webViewLink": web_link}
            if body.get("mimeType") == "application/vnd.google-apps.folder":
                self.service.created_folders.append(body)
                self.service.existing_folders[(parent_id, name)] = payload
            else:
                self.service.uploaded_files.append(
                    {"body": body, "media_body": media_body, "fields": fields}
                )
            return {"id": file_id, "webViewLink": web_link}

        return MockExecute(run)

    def list(
        self,
        q: str = "",
        fields: str = "files(id,name)",
        pageSize: int | None = None,
    ) -> MockExecute:
        def run() -> dict[str, Any]:
            name = _extract_quoted_after(q, "name = ")
            parent_id = _extract_parent(q)
            folder = self.service.existing_folders.get((parent_id, name))
            return {"files": [folder] if folder else []}

        return MockExecute(run)

    def get(self, fileId: str, fields: str = "webViewLink") -> MockExecute:
        return MockExecute(lambda: {"webViewLink": f"https://drive.google.com/file/d/{fileId}/view"})


class MockExecute:
    """Object exposing the Drive client's `.execute()` call shape."""

    def __init__(self, callback) -> None:
        self.callback = callback

    def execute(self):
        return self.callback()


def build_mock_credentials() -> MagicMock:
    credentials = MagicMock()
    credentials.valid = True
    credentials.expired = False
    credentials.refresh_token = "refresh-token"
    credentials.to_json.return_value = "{}"
    return credentials


def patch_authenticate(monkeypatch, module, return_credentials=None):
    credentials = return_credentials or build_mock_credentials()
    monkeypatch.setattr(module, "authenticate", lambda *_args, **_kwargs: credentials)
    return credentials


def patch_drive_service_build(monkeypatch, module, mock_service: MockDriveService):
    monkeypatch.setattr(module, "build_drive_service", lambda _credentials: mock_service)
    return mock_service


def _extract_quoted_after(query: str, prefix: str) -> str:
    start = query.index(prefix) + len(prefix)
    if query[start] != "'":
        raise ValueError(query)
    end = query.index("'", start + 1)
    return query[start + 1 : end]


def _extract_parent(query: str) -> str | None:
    marker = "' in parents"
    if marker not in query:
        return None
    end = query.index(marker)
    start = query.rfind("'", 0, end)
    return query[start + 1 : end]
