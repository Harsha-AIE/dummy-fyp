from pathlib import Path
from uuid import uuid4

import pytest

from edukg.core.models import Document
from edukg.core.models.enums import DocumentType
from edukg.infrastructure.cache.filesystem_cache import FilesystemCacheRepository
from edukg.infrastructure.filesystem.document_repository import FilesystemDocumentRepository
from edukg.modules.ingestion.ingestion_service import DocumentIngestionService
from edukg.pipeline.context import PipelineContext


class _StubOCRProvider:
    def extract_text(self, image_path: Path, *, layout_aware: bool = True) -> list[dict]:
        return []


def _make_docx(path: Path) -> None:
    import docx

    doc = docx.Document()
    doc.add_paragraph("Gradient descent minimizes the loss function.")
    doc.add_paragraph("Backpropagation computes gradients efficiently.")
    doc.save(str(path))


def _make_pdf(path: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as f:
        writer.write(f)


def _build_service(tmp_path: Path) -> DocumentIngestionService:
    doc_repo = FilesystemDocumentRepository(base_dir=tmp_path / "documents")
    cache_repo = FilesystemCacheRepository(base_dir=tmp_path / "cache")
    return DocumentIngestionService(
        document_repository=doc_repo,
        cache_repository=cache_repo,
        ocr_provider=_StubOCRProvider(),
    ), doc_repo


def test_ingestion_service_processes_docx(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    _make_docx(docx_path)

    service, doc_repo = _build_service(tmp_path)
    placeholder = Document(file_path=docx_path, file_hash="0" * 64, document_type=DocumentType.DOCX)
    context = PipelineContext(document=placeholder)

    result = service.run(context)

    assert result.document.document_type == DocumentType.DOCX
    assert result.document.file_hash != "0" * 64
    assert len(result.chunks) >= 1
    assert any("gradient descent" in c.text.lower() for c in result.chunks)

    persisted = doc_repo.get(result.document.id)
    assert persisted is not None
    assert doc_repo.get_chunks(result.document.id) == result.chunks


def test_ingestion_service_is_idempotent_via_hash_cache(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    _make_docx(docx_path)

    service, _ = _build_service(tmp_path)
    placeholder = Document(file_path=docx_path, file_hash="0" * 64, document_type=DocumentType.DOCX)

    first = service.run(PipelineContext(document=placeholder))
    second = service.run(PipelineContext(document=placeholder.model_copy(update={"id": uuid4()})))

    assert first.document.id == second.document.id  # second run hit the hash cache


def test_ingestion_service_rejects_missing_file(tmp_path: Path) -> None:
    from edukg.core.exceptions import DocumentProcessingError

    service, _ = _build_service(tmp_path)
    placeholder = Document(file_path=tmp_path / "missing.docx", file_hash="0" * 64, document_type=DocumentType.DOCX)

    with pytest.raises(DocumentProcessingError):
        service.run(PipelineContext(document=placeholder))


def test_ingestion_service_rejects_unsupported_extension(tmp_path: Path) -> None:
    from edukg.core.exceptions import DocumentProcessingError

    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("plain text file")

    service, _ = _build_service(tmp_path)
    placeholder = Document(file_path=bad_file, file_hash="0" * 64, document_type=DocumentType.DOCX)

    with pytest.raises(DocumentProcessingError):
        service.run(PipelineContext(document=placeholder))
