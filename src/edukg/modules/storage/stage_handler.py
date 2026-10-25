from __future__ import annotations

from edukg.core.exceptions import GraphStorageError
from edukg.core.interfaces import PipelineStageHandler
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import PipelineStage
from edukg.modules.storage.graph_storage_service import GraphStorageService
from edukg.pipeline.context import PipelineContext

logger = get_logger(__name__)


class GraphStorageStageHandler(PipelineStageHandler[PipelineContext, PipelineContext]):
    stage = PipelineStage.GRAPH_STORAGE

    def __init__(self, service: GraphStorageService) -> None:
        self._service = service

    def run(self, input_data: PipelineContext) -> PipelineContext:
        context = input_data
        if not context.entities:
            raise GraphStorageError("No entities available — entity linking must run before graph storage")

        self._service.store(context.entities, context.triples)

        context.mark_complete(self.stage)
        logger.info(
            "stage_graph_storage_done",
            document_id=str(context.document.id),
            entity_count=len(context.entities),
            triple_count=len(context.triples),
        )
        return context
