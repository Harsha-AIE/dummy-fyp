from __future__ import annotations

from edukg.core.exceptions import TripleValidationError
from edukg.core.interfaces import PipelineStageHandler
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import PipelineStage
from edukg.modules.ontology.ontology_manager import OntologyManager
from edukg.modules.triple_validation.domain_range_validator import TripleValidationService
from edukg.pipeline.context import PipelineContext

logger = get_logger(__name__)


class TripleValidationStageHandler(PipelineStageHandler[PipelineContext, PipelineContext]):
    stage = PipelineStage.TRIPLE_VALIDATION

    def __init__(self, validator: TripleValidationService, ontology_manager: OntologyManager) -> None:
        self._validator = validator
        self._manager = ontology_manager

    def run(self, input_data: PipelineContext) -> PipelineContext:
        context = input_data
        if not context.triples:
            raise TripleValidationError("No triples available — triple generation must run before triple validation")

        active = self._manager.get_active_ontology(context.document.detected_subject)
        entities_by_id = {e.id: e for e in context.entities}
        results = self._validator.validate_batch(context.triples, entities_by_id, active)

        context.validation_results.extend(results)
        context.triples = [
            triple.model_copy(update={"validation_status": result.status})
            for triple, result in zip(context.triples, results, strict=True)
        ]
        context.mark_complete(self.stage)
        logger.info(
            "stage_triple_validation_done",
            document_id=str(context.document.id),
            triple_count=len(context.triples),
            invalid_count=sum(1 for r in results if not r.is_valid),
        )
        return context
