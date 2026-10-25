"""Router functions called directly, same philosophy as test_api_review.py.
Async endpoints are driven with asyncio.run() rather than pulling in
pytest-asyncio for a single file's worth of async tests."""

import asyncio
import io
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

from edukg.api.routers import pipeline
from edukg.core.exceptions import ExternalServiceError
from edukg.infrastructure.filesystem.document_repository import FilesystemDocumentRepository


class _FakePostgresStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.saved: list[tuple[str, bytes, str | None]] = []

    def save(self, filename: str, content: bytes, content_type: str | None) -> None:
        if self.fail:
            raise ExternalServiceError("simulated Postgres outage")
        self.saved.append((filename, content, content_type))


@dataclass
class _FakeResources:
    document_repo: FilesystemDocumentRepository | None = None
    postgres_store: _FakePostgresStore = field(default_factory=_FakePostgresStore)


def _upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(content))


def test_upload_document_saves_into_raw_dir_and_postgres(tmp_path: Path) -> None:
    resources = _FakeResources()
    result = asyncio.run(pipeline.upload_document(_upload("notes.pdf", b"%PDF-1.4 fake"), resources, raw_dir=tmp_path))

    assert result.name == "notes.pdf"
    assert (tmp_path / "notes.pdf").read_bytes() == b"%PDF-1.4 fake"
    assert resources.postgres_store.saved == [("notes.pdf", b"%PDF-1.4 fake", None)]


def test_upload_document_strips_path_traversal(tmp_path: Path) -> None:
    result = asyncio.run(
        pipeline.upload_document(_upload("../../etc/evil.pdf", b"content"), _FakeResources(), raw_dir=tmp_path)
    )

    assert result.name == "evil.pdf"
    assert (tmp_path / "evil.pdf").exists()  # written directly inside raw_dir, no directory traversal honored


def test_upload_document_rejects_unsupported_extension(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pipeline.upload_document(_upload("script.exe", b"content"), _FakeResources(), raw_dir=tmp_path))
    assert exc_info.value.status_code == 400


def test_upload_document_rejects_empty_file(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pipeline.upload_document(_upload("empty.pdf", b""), _FakeResources(), raw_dir=tmp_path))
    assert exc_info.value.status_code == 400


def test_upload_document_rejects_missing_filename(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pipeline.upload_document(_upload("", b"content"), _FakeResources(), raw_dir=tmp_path))
    assert exc_info.value.status_code == 400


def test_upload_document_surfaces_postgres_failure_but_keeps_local_copy(tmp_path: Path) -> None:
    resources = _FakeResources(postgres_store=_FakePostgresStore(fail=True))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pipeline.upload_document(_upload("notes.pdf", b"content"), resources, raw_dir=tmp_path))

    assert exc_info.value.status_code == 502
    assert (tmp_path / "notes.pdf").read_bytes() == b"content"  # local write already happened, not rolled back


def test_list_available_documents_reflects_raw_dir_and_ingestion_state(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "a.pdf").write_bytes(b"content-a")
    (raw_dir / "b.docx").write_bytes(b"content-b")
    (raw_dir / "ignore.txt").write_text("not a supported type")

    document_repo = FilesystemDocumentRepository(base_dir=tmp_path / "documents")
    from edukg.core.models import Document, DocumentType

    saved = document_repo.save(Document(file_path=raw_dir / "a.pdf", file_hash="a" * 64, document_type=DocumentType.PDF))

    docs = pipeline.list_available_documents(_FakeResources(document_repo=document_repo), raw_dir=raw_dir)

    names = {d.name: d.already_ingested for d in docs}
    assert names == {"a.pdf": True, "b.docx": False}

    by_name = {d.name: d for d in docs}
    assert by_name["a.pdf"].document_id == saved.id  # ingested document exposes its real id for linking
    assert by_name["b.docx"].document_id is None  # never-ingested document has no id to link to


def test_list_available_documents_on_missing_raw_dir_returns_empty(tmp_path: Path) -> None:
    document_repo = FilesystemDocumentRepository(base_dir=tmp_path / "documents")
    docs = pipeline.list_available_documents(_FakeResources(document_repo=document_repo), raw_dir=tmp_path / "nonexistent")
    assert docs == []
