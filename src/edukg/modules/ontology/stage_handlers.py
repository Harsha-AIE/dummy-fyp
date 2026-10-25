from __future__ import annotations

from edukg.core.exceptions import OntologyError
from edukg.core.interfaces import PipelineStageHandler
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import PipelineStage
from edukg.modules.ontology.ontology_extension_service import OntologyExtensionService
from edukg.modules.ontology.ontology_manager import OntologyManager
from edukg.modules.ontology.ontology_review_service import OntologyReviewService
from edukg.modules.ontology.ontology_validator import OntologyValidationService
from edukg.pipeline.context import PipelineContext

logger = get_logger(__name__)


class OntologyLoadStageHandler(PipelineStageHandler[PipelineContext, PipelineContext]):
    """Loads core + domain ontology for the document's detected subject into the context."""

    stage = PipelineStage.ONTOLOGY_LOAD

    def __init__(self, ontology_manager: OntologyManager) -> None:
        self._manager = ontology_manager

    def run(self, input_data: PipelineContext) -> PipelineContext:
        context = input_data
        active = self._manager.get_active_ontology(context.document.detected_subject)
        context.ontology_classes = active.classes
        context.ontology_relations = active.relations
        context.mark_complete(self.stage)
        logger.info(
            "stage_ontology_load_done",
            document_id=str(context.document.id),
            class_count=len(active.classes),
            relation_count=len(active.relations),
        )
        return context


class OntologyExtensionStageHandler(PipelineStageHandler[PipelineContext, PipelineContext]):
    """Proposes ontology extensions from the document's chunks via the LLM. Proposals are
    persisted for audit but NOT merged here — that happens in the validation stage's
    auto-review pass, or later via a human reviewer using OntologyReviewService."""

    stage = PipelineStage.ONTOLOGY_EXTENSION

    def __init__(self, extension_service: OntologyExtensionService, ontology_manager: OntologyManager, repository) -> None:
        self._extension_service = extension_service
        self._manager = ontology_manager
        self._repository = repository

    def run(self, input_data: PipelineContext) -> PipelineContext:
        context = input_data
        active = self._manager.get_active_ontology(context.document.detected_subject)

        proposals = self._extension_service.propose_extensions(context.document, context.chunks, active)
        for proposal in proposals:
            self._repository.save_extension_proposal(proposal)

        context.proposed_extensions = proposals
        context.mark_complete(self.stage)
        logger.info(
            "stage_ontology_extension_done", document_id=str(context.document.id), proposal_count=len(proposals)
        )
        return context


class OntologyValidationStageHandler(PipelineStageHandler[PipelineContext, PipelineContext]):
    """Validates every proposed extension, then runs the automated (SYSTEM-role) review
    pass so offline pipeline runs can proceed without blocking on a human reviewer.
    Accepted extensions are merged into the ontology and reflected back into the context;
    everything else remains pending for a future human reviewer."""

    stage = PipelineStage.ONTOLOGY_VALIDATION

    def __init__(
        self,
        validator: OntologyValidationService,
        review_service: OntologyReviewService,
        ontology_manager: OntologyManager,
    ) -> None:
        self._validator = validator
        self._review_service = review_service
        self._manager = ontology_manager

    def run(self, input_data: PipelineContext) -> PipelineContext:
        context = input_data
        active = self._manager.get_active_ontology(context.document.detected_subject)

        validation_results = self._validator.validate_batch(context.proposed_extensions, active)
        context.validation_results.extend(validation_results)

        accepted = []
        for extension, result in zip(context.proposed_extensions, validation_results, strict=True):
            try:
                was_approved, _ = self._review_service.auto_review(extension, result, active)
            except OntologyError as exc:
                logger.warning("ontology_auto_review_merge_failed", extension_id=str(extension.id), error=str(exc))
                continue
            if was_approved:
                accepted.append(extension)

        context.accepted_extensions = accepted
        if accepted:
            active = self._manager.get_active_ontology(context.document.detected_subject, force_reload=True)
            context.ontology_classes = active.classes
            context.ontology_relations = active.relations

        context.mark_complete(self.stage)
        logger.info(
            "stage_ontology_validation_done",
            document_id=str(context.document.id),
            proposed=len(context.proposed_extensions),
            accepted=len(accepted),
        )
        return context
