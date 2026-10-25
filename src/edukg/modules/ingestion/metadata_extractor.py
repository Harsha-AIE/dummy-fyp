from __future__ import annotations

from pathlib import Path

from edukg.core.logging_setup import get_logger
from edukg.core.models import DocumentMetadata
from edukg.core.models.enums import DocumentType

logger = get_logger(__name__)


class MetadataExtractor:
    """Extracts title/author/page_count from a document's native metadata where
    available. Never raises — a missing/corrupt metadata block degrades to defaults
    rather than failing ingestion, since metadata is enrichment, not a hard requirement."""

    def extract(self, file_path: Path, document_type: DocumentType) -> DocumentMetadata:
        if document_type == DocumentType.PDF:
            return self._extract_pdf(file_path)
        if document_type == DocumentType.DOCX:
            return self._extract_docx(file_path)
        return DocumentMetadata()

    def _extract_pdf(self, file_path: Path) -> DocumentMetadata:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            info = reader.metadata or {}
            return DocumentMetadata(
                title=(info.title if hasattr(info, "title") else info.get("/Title")) or None,
                author=(info.author if hasattr(info, "author") else info.get("/Author")) or None,
                page_count=len(reader.pages),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdf_metadata_extraction_failed", file_name=file_path.name, error=str(exc))
            return DocumentMetadata()

    def _extract_docx(self, file_path: Path) -> DocumentMetadata:
        try:
            import docx

            document = docx.Document(str(file_path))
            props = document.core_properties
            return DocumentMetadata(title=props.title or None, author=props.author or None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("docx_metadata_extraction_failed", file_name=file_path.name, error=str(exc))
            return DocumentMetadata()
