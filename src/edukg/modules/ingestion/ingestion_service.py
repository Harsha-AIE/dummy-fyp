from __future__ import annotations

from pathlib import Path

from edukg.core.config import get_settings
from edukg.core.exceptions import DocumentProcessingError
from edukg.core.interfaces import CacheRepository, DocumentRepository, OCRProvider, PipelineStageHandler
from edukg.core.logging_setup import get_logger
from edukg.core.models import DocumentMetadata
from edukg.core.models.enums import DocumentType, ExtractionMethod, PipelineStage
from edukg.modules.ingestion.chunker import ChunkGenerator
from edukg.modules.ingestion.docx_processor import DOCXProcessor
from edukg.modules.ingestion.hashing import compute_file_hash
from edukg.modules.ingestion.layout_assembly import assemble_layout_text
from edukg.modules.ingestion.metadata_extractor import MetadataExtractor
from edukg.modules.ingestion.pdf_processor import PDFProcessor
from edukg.modules.ingestion.pdf_rasterizer import PDFRasterizer
from edukg.modules.ingestion.text_cleaner import TextCleaner
from edukg.pipeline.context import PipelineContext

logger = get_logger(__name__)

_EXTENSION_TO_TYPE = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
}


class DocumentIngestionService(PipelineStageHandler[PipelineContext, PipelineContext]):
    """
    INGESTION stage: hash the raw file (for caching/dedup) -> detect document type ->
    extract text (native PDF text, falling back page-by-page to PaddleOCR for scanned
    pages; DOCX via python-docx) -> clean -> chunk -> extract metadata -> persist
    Document + Chunks via DocumentRepository. If a Document with the same file_hash
    already exists, ingestion is skipped and the cached result is reused.
    """

    stage = PipelineStage.INGESTION

    def __init__(
        self,
        document_repository: DocumentRepository,
        cache_repository: CacheRepository,
        ocr_provider: OCRProvider,
        pdf_processor: PDFProcessor | None = None,
        docx_processor: DOCXProcessor | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        text_cleaner: TextCleaner | None = None,
        chunk_generator: ChunkGenerator | None = None,
        rasterizer: PDFRasterizer | None = None,
    ) -> None:
        self._documents = document_repository
        self._cache = cache_repository
        self._ocr = ocr_provider
        self._pdf_processor = pdf_processor or PDFProcessor()
        self._docx_processor = docx_processor or DOCXProcessor()
        self._metadata_extractor = metadata_extractor or MetadataExtractor()
        self._cleaner = text_cleaner or TextCleaner()
        self._chunker = chunk_generator or ChunkGenerator()
        self._rasterizer = rasterizer or PDFRasterizer(get_settings().paths.cache_dir / "ocr_pages")

    def run(self, input_data: PipelineContext) -> PipelineContext:
        context = input_data
        file_path = context.document.file_path

        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")

        file_hash = compute_file_hash(file_path)
        cached = self._documents.get_by_hash(file_hash)
        if cached is not None:
            logger.info("ingestion_cache_hit", file_hash=file_hash, document_id=str(cached.id))
            context.document = cached
            context.chunks = self._documents.get_chunks(cached.id)
            context.mark_complete(self.stage)
            return context

        document_type = _EXTENSION_TO_TYPE.get(file_path.suffix.lower())
        if document_type is None:
            raise DocumentProcessingError(f"Unsupported file type: {file_path.suffix}")

        page_texts = self._extract(file_path, document_type)
        metadata = self._metadata_extractor.extract(file_path, document_type)

        document = context.document.model_copy(
            update={
                "file_hash": file_hash,
                "document_type": document_type,
                "metadata": self._merge_metadata(metadata, len(page_texts)),
            }
        )

        chunks = self._chunker.generate(document.id, page_texts)
        if not chunks:
            raise DocumentProcessingError(f"No extractable text produced from '{file_path.name}'")

        self._documents.save(document)
        self._documents.save_chunks(chunks)

        context.document = document
        context.chunks = chunks
        context.mark_complete(self.stage)

        logger.info(
            "stage_ingestion_done",
            document_id=str(document.id),
            document_type=document_type.value,
            chunk_count=len(chunks),
        )
        return context

    def _merge_metadata(self, metadata: DocumentMetadata, page_count: int) -> DocumentMetadata:
        if metadata.page_count is None:
            return metadata.model_copy(update={"page_count": page_count})
        return metadata

    def _extract(
        self, file_path: Path, document_type: DocumentType
    ) -> list[tuple[int | None, str, ExtractionMethod]]:
        if document_type == DocumentType.DOCX:
            text = self._docx_processor.extract(file_path)
            cleaned = self._cleaner.clean(text)
            return [(None, cleaned, self._docx_processor.extraction_method())]

        return self._extract_pdf(file_path)

    def _extract_pdf(self, file_path: Path) -> list[tuple[int | None, str, ExtractionMethod]]:
        pages = self._pdf_processor.extract(file_path)
        results: list[tuple[int | None, str, ExtractionMethod]] = []

        for page in pages:
            if not page.needs_ocr:
                results.append((page.page_number, self._cleaner.clean(page.text), ExtractionMethod.NATIVE_TEXT))
                continue

            try:
                image_path = self._rasterizer.rasterize_page(file_path, page.page_number)
                blocks = self._ocr.extract_text(image_path, layout_aware=True)
                ocr_text = assemble_layout_text(blocks)
            except DocumentProcessingError as exc:
                logger.warning("pdf_page_ocr_failed", page=page.page_number, error=str(exc))
                ocr_text = page.text  # fall back to whatever native extraction produced, even if sparse

            method = ExtractionMethod.HYBRID if page.text.strip() else ExtractionMethod.OCR_PADDLE
            results.append((page.page_number, self._cleaner.clean(ocr_text), method))

        return results
