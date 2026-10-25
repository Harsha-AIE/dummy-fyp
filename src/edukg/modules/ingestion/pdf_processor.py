from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from edukg.core.exceptions import DocumentProcessingError
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import ExtractionMethod

logger = get_logger(__name__)

# A page with fewer than this many native characters is assumed to be a scanned
# image (or near-empty) and is routed to OCR instead of being trusted as text.
_MIN_NATIVE_CHARS_PER_PAGE = 20


@dataclass(slots=True)
class ExtractedPage:
    page_number: int
    text: str
    extraction_method: ExtractionMethod
    needs_ocr: bool


class PDFProcessor:
    """Extracts native text per page via pypdf. Pages with little/no extractable text
    (scanned pages) are marked needs_ocr=True for the OCR provider to fill in —
    this is the layer that decides *which* pages need OCR; it never performs OCR itself."""

    def extract(self, file_path: Path) -> list[ExtractedPage]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentProcessingError("pypdf is not installed") from exc

        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:  # noqa: BLE001 - any pypdf failure becomes a typed error
            raise DocumentProcessingError(f"Failed to open PDF '{file_path.name}': {exc}") from exc

        pages: list[ExtractedPage] = []
        for index, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("pdf_page_extract_failed", page=index, error=str(exc))
                text = ""

            needs_ocr = len(text.strip()) < _MIN_NATIVE_CHARS_PER_PAGE
            pages.append(
                ExtractedPage(
                    page_number=index,
                    text=text,
                    extraction_method=ExtractionMethod.NATIVE_TEXT,
                    needs_ocr=needs_ocr,
                )
            )

        logger.info(
            "pdf_extracted",
            file_name=file_path.name,
            page_count=len(pages),
            pages_needing_ocr=sum(1 for p in pages if p.needs_ocr),
        )
        return pages

    def page_count(self, file_path: Path) -> int:
        try:
            from pypdf import PdfReader

            return len(PdfReader(str(file_path)).pages)
        except Exception as exc:  # noqa: BLE001
            raise DocumentProcessingError(f"Failed to read page count for '{file_path.name}': {exc}") from exc
