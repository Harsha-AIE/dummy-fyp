from __future__ import annotations

from edukg.core.exceptions import KnowledgeFusionError
from edukg.core.interfaces import PipelineStageHandler
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import PipelineStage
from edukg.modules.fusion.fusion_service import KnowledgeFusionService
from edukg.modules.fusion.pending_triple_store import PendingTripleStore
from edukg.pipeline.context import PipelineContext

logger = get_logger(__name__)


class KnowledgeFusionStageHandler(PipelineStageHandler[PipelineContext, PipelineContext]):
    stage = PipelineStage.KNOWLEDGE_FUSION

    def __init__(self, service: KnowledgeFusionService, pending_store: PendingTripleStore | None = None) -> None:
        self._service = service
        self._pending_store = pending_store or PendingTripleStore()

    def run(self, input_data: PipelineContext) -> PipelineContext:
        context = input_data
        if not context.triples:
            raise KnowledgeFusionError("No triples available — triple validation must run before knowledge fusion")

        accepted, pending = self._service.fuse(context.triples)
        self._pending_store.save_all(pending)

        context.triples = accepted
        context.mark_complete(self.stage)
        logger.info(
            "stage_knowledge_fusion_done", document_id=str(context.document.id), accepted=len(accepted), pending=len(pending)
        )
        return context
