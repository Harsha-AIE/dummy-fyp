from __future__ import annotations

from edukg.core.exceptions import EntityExtractionError
from edukg.core.interfaces import PipelineStageHandler
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import PipelineStage
from edukg.modules.entity_extraction.mention_extractor import MentionExtractor
from edukg.pipeline.context import PipelineContext

logger = get_logger(__name__)


class EntityExtractionStageHandler(PipelineStageHandler[PipelineContext, PipelineContext]):
    stage = PipelineStage.ENTITY_EXTRACTION

    def __init__(self, extractor: MentionExtractor) -> None:
        self._extractor = extractor

    def run(self, input_data: PipelineContext) -> PipelineContext:
        context = input_data
        if not context.chunks:
            raise EntityExtractionError("No chunks available — ingestion must run before entity extraction")

        mentions = []
        for chunk in context.chunks:
            mentions.extend(self._extractor.extract(chunk))

        context.mentions = mentions
        context.mark_complete(self.stage)
        logger.info("stage_entity_extraction_done", document_id=str(context.document.id), mention_count=len(mentions))
        return context
