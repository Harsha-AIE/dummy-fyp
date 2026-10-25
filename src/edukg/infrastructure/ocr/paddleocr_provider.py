from __future__ import annotations

from pathlib import Path

from edukg.core.config import OCRSettings, get_settings
from edukg.core.exceptions import ExternalServiceError
from edukg.core.interfaces import OCRProvider
from edukg.core.logging_setup import get_logger

logger = get_logger(__name__)


class PaddleOCRProvider(OCRProvider):
    """
    OCRProvider implementation backed by PaddleOCR 3.x. Returns layout blocks
    (text + bounding box + confidence) in reading order so callers can do
    layout-aware assembly rather than treating a page as one opaque string.
    The PaddleOCR engine is created lazily and cached on first use.
    """

    def __init__(self, settings: OCRSettings | None = None) -> None:
        self._settings = settings or get_settings().ocr
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ExternalServiceError("The 'paddleocr' package is not installed") from exc
            self._engine = PaddleOCR(
                lang=self._settings.lang,
                use_angle_cls=True,
                use_gpu=self._settings.use_gpu,
                show_log=False,
            )
        return self._engine

    def extract_text(self, image_path: Path, *, layout_aware: bool = True) -> list[dict]:
        engine = self._get_engine()
        try:
            raw_result = engine.ocr(str(image_path), cls=True)
        except Exception as exc:  # noqa: BLE001 - any PaddleOCR failure becomes a typed error
            raise ExternalServiceError(f"PaddleOCR failed on '{image_path.name}': {exc}") from exc

        blocks: list[dict] = []
        for page_result in raw_result or []:
            for reading_order, line in enumerate(page_result or []):
                bbox, (text, confidence) = line
                blocks.append(
                    {
                        "text": text,
                        "bbox": bbox,
                        "confidence": float(confidence),
                        "reading_order": reading_order,
                    }
                )

        if layout_aware:
            blocks.sort(key=lambda b: (min(p[1] for p in b["bbox"]), min(p[0] for p in b["bbox"])))

        logger.info("ocr_extracted", file_name=image_path.name, block_count=len(blocks))
        return blocks
