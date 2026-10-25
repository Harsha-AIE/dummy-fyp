from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from edukg.core.config import get_settings
from edukg.core.exceptions import CacheError
from edukg.core.interfaces import DocumentRepository
from edukg.core.logging_setup import get_logger
from edukg.core.models import Chunk, Document

logger = get_logger(__name__)


class FilesystemDocumentRepository(DocumentRepository):
    """Persists Document + Chunk records as JSON under data/processed/documents.
    One folder per document: <document_id>/document.json, <document_id>/chunks.json.
    A hash index (<hash>.id) enables fast get_by_hash lookups for ingestion caching."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or get_settings().paths.processed_dir / "documents"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._hash_index_dir = self._base_dir / "_hash_index"
        self._hash_index_dir.mkdir(parents=True, exist_ok=True)

    def _doc_dir(self, document_id: UUID) -> Path:
        path = self._base_dir / str(document_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, document: Document) -> Document:
        doc_dir = self._doc_dir(document.id)
        try:
            (doc_dir / "document.json").write_text(
                json.dumps(document.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            (self._hash_index_dir / f"{document.file_hash}.id").write_text(str(document.id), encoding="utf-8")
        except OSError as exc:
            raise CacheError(f"Failed saving document {document.id}") from exc
        logger.info("document_saved", document_id=str(document.id), file_hash=document.file_hash)
        return document

    def get(self, document_id: UUID) -> Document | None:
        path = self._base_dir / str(document_id) / "document.json"
        if not path.exists():
            return None
        return Document.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def get_by_hash(self, file_hash: str) -> Document | None:
        index_path = self._hash_index_dir / f"{file_hash}.id"
        if not index_path.exists():
            return None
        document_id = UUID(index_path.read_text(encoding="utf-8").strip())
        return self.get(document_id)

    def save_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        document_id = chunks[0].document_id
        doc_dir = self._doc_dir(document_id)
        try:
            (doc_dir / "chunks.json").write_text(
                json.dumps([c.model_dump(mode="json") for c in chunks], indent=2), encoding="utf-8"
            )
        except OSError as exc:
            raise CacheError(f"Failed saving chunks for document {document_id}") from exc
        logger.info("chunks_saved", document_id=str(document_id), chunk_count=len(chunks))

    def get_chunks(self, document_id: UUID) -> list[Chunk]:
        path = self._base_dir / str(document_id) / "chunks.json"
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [Chunk.model_validate(c) for c in raw]

    def list_all_documents(self) -> list[Document]:
        documents: list[Document] = []
        for doc_dir in sorted(self._base_dir.iterdir()):
            if doc_dir.name == "_hash_index" or not doc_dir.is_dir():
                continue
            doc_path = doc_dir / "document.json"
            if doc_path.exists():
                documents.append(Document.model_validate(json.loads(doc_path.read_text(encoding="utf-8"))))
        return documents
