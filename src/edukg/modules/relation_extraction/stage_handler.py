from __future__ import annotations

from edukg.core.exceptions import RelationExtractionError
from edukg.core.interfaces import PipelineStageHandler
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import PipelineStage
from edukg.modules.ontology.ontology_manager import OntologyManager
from edukg.modules.relation_extraction.relation_extraction_service import RelationExtractionService
from edukg.pipeline.context import PipelineContext

logger = get_logger(__name__)


class RelationExtractionStageHandler(PipelineStageHandler[PipelineContext, PipelineContext]):
    stage = PipelineStage.RELATION_EXTRACTION

    def __init__(self, service: RelationExtractionService, ontology_manager: OntologyManager) -> None:
        self._service = service
        self._manager = ontology_manager

    def run(self, input_data: PipelineContext) -> PipelineContext:
        context = input_data
        if not context.entities:
            raise RelationExtractionError("No entities available — entity linking must run before relation extraction")

        active = self._manager.get_active_ontology(context.document.detected_subject)
        relations = self._service.extract(context.document.id, context.chunks, context.mentions, context.entities, active)

        context.relations = relations
        context.mark_complete(self.stage)
        logger.info("stage_relation_extraction_done", document_id=str(context.document.id), relation_count=len(relations))
        return context
