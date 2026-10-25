from __future__ import annotations

from pathlib import Path

from edukg.core.exceptions import DocumentProcessingError
from edukg.core.logging_setup import get_logger

logger = get_logger(__name__)


class PDFRasterizer:
    """Converts specific PDF pages to temporary PNG images for OCR input. Only
    invoked for pages PDFProcessor flagged as needs_ocr — never for the whole document."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def rasterize_page(self, pdf_path: Path, page_number: int, *, dpi: int = 300) -> Path:
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise DocumentProcessingError("pdf2image is not installed") from exc

        try:
            images = convert_from_path(
                str(pdf_path), dpi=dpi, first_page=page_number + 1, last_page=page_number + 1
            )
        except Exception as exc:  # noqa: BLE001
            raise DocumentProcessingError(
                f"Failed to rasterize page {page_number} of '{pdf_path.name}': {exc}"
            ) from exc

        if not images:
            raise DocumentProcessingError(f"Rasterization produced no image for page {page_number} of '{pdf_path.name}'")

        image_path = self._output_dir / f"{pdf_path.stem}_p{page_number}.png"
        images[0].save(image_path)
        logger.info("pdf_page_rasterized", file_name=pdf_path.name, page=page_number, image_path=str(image_path))
        return image_path
