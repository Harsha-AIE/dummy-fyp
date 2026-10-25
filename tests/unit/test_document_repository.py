from pathlib import Path

from edukg.core.models import Document, DocumentType
from edukg.infrastructure.filesystem.document_repository import FilesystemDocumentRepository


def test_list_all_documents_returns_every_saved_document(tmp_path: Path) -> None:
    repo = FilesystemDocumentRepository(base_dir=tmp_path / "documents")
    doc_a = Document(file_path=tmp_path / "a.docx", file_hash="a" * 64, document_type=DocumentType.DOCX)
    doc_b = Document(file_path=tmp_path / "b.pdf", file_hash="b" * 64, document_type=DocumentType.PDF)
    repo.save(doc_a)
    repo.save(doc_b)

    documents = repo.list_all_documents()

    assert {d.id for d in documents} == {doc_a.id, doc_b.id}


def test_list_all_documents_on_empty_repository(tmp_path: Path) -> None:
    repo = FilesystemDocumentRepository(base_dir=tmp_path / "documents")
    assert repo.list_all_documents() == []
