from __future__ import annotations

from edukg.core.exceptions import EntityLinkingError
from edukg.core.interfaces import PipelineStageHandler
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import PipelineStage
from edukg.modules.entity_linking.linking_service import EntityLinkingService
from edukg.modules.ontology.ontology_manager import OntologyManager
from edukg.pipeline.context import PipelineContext

logger = get_logger(__name__)


class EntityLinkingStageHandler(PipelineStageHandler[PipelineContext, PipelineContext]):
    stage = PipelineStage.ENTITY_LINKING

    def __init__(self, service: EntityLinkingService, ontology_manager: OntologyManager) -> None:
        self._service = service
        self._manager = ontology_manager

    def run(self, input_data: PipelineContext) -> PipelineContext:
        context = input_data
        if not context.mentions:
            raise EntityLinkingError("No mentions available — entity extraction must run before entity linking")

        active = self._manager.get_active_ontology(context.document.detected_subject)
        entities = self._service.link(context.mentions, context.document.id, active)

        context.entities = entities
        context.mark_complete(self.stage)
        logger.info("stage_entity_linking_done", document_id=str(context.document.id), entity_count=len(entities))
        return context
