from __future__ import annotations

from edukg.core.config import get_settings
from edukg.core.exceptions import OntologyError
from edukg.core.interfaces import OntologyRepository
from edukg.core.logging_setup import get_logger
from edukg.core.models import (
    OntologyClass,
    OntologyExtension,
    OntologyRelation,
    ReviewAction,
    ReviewDecision,
    ReviewerRole,
    ValidationResult,
    ValidationStatus,
)
from edukg.modules.ontology.active_ontology import ActiveOntology
from edukg.modules.ontology.ontology_merge_service import OntologyMergeService

logger = get_logger(__name__)


class OntologyReviewService:
    """
    Backend service layer for the ontology human-in-the-loop review workflow.
    A future UI (teacher/admin) calls approve/reject/modify/merge; this service
    is the only thing allowed to turn a ReviewDecision into a call on
    OntologyMergeService. Every decision, regardless of outcome, is persisted
    via OntologyRepository.record_review_decision for full auditability.
    """

    def __init__(self, repository: OntologyRepository, merge_service: OntologyMergeService) -> None:
        self._repository = repository
        self._merge_service = merge_service

    def approve(
        self,
        extension: OntologyExtension,
        active_ontology: ActiveOntology,
        *,
        reviewer_id: str,
        reviewer_role: ReviewerRole = ReviewerRole.TEACHER,
        comment: str | None = None,
    ) -> OntologyClass | OntologyRelation:
        decision = ReviewDecision(
            target_id=extension.id,
            target_type="OntologyExtension",
            action=ReviewAction.APPROVE,
            reviewer_role=reviewer_role,
            reviewer_id=reviewer_id,
            comment=comment,
        )
        self._repository.record_review_decision(decision)
        self._forget_pending(extension)
        merged = self._merge_service.merge(extension, active_ontology)
        logger.info("ontology_extension_approved", extension_id=str(extension.id), reviewer=reviewer_id)
        return merged

    def reject(
        self,
        extension: OntologyExtension,
        *,
        reviewer_id: str,
        reviewer_role: ReviewerRole = ReviewerRole.TEACHER,
        comment: str | None = None,
    ) -> ReviewDecision:
        decision = ReviewDecision(
            target_id=extension.id,
            target_type="OntologyExtension",
            action=ReviewAction.REJECT,
            reviewer_role=reviewer_role,
            reviewer_id=reviewer_id,
            comment=comment,
        )
        saved = self._repository.record_review_decision(decision)
        self._forget_pending(extension)
        logger.info("ontology_extension_rejected", extension_id=str(extension.id), reviewer=reviewer_id)
        return saved

    def modify(
        self,
        extension: OntologyExtension,
        active_ontology: ActiveOntology,
        *,
        modified_fields: dict,
        reviewer_id: str,
        reviewer_role: ReviewerRole = ReviewerRole.TEACHER,
        comment: str | None = None,
    ) -> OntologyClass | OntologyRelation:
        """Reviewer edits fields (e.g. renames a proposed class) before merging."""
        updated_extension = extension.model_copy(update=modified_fields)
        decision = ReviewDecision(
            target_id=extension.id,
            target_type="OntologyExtension",
            action=ReviewAction.MODIFY,
            reviewer_role=reviewer_role,
            reviewer_id=reviewer_id,
            comment=comment,
            modified_payload=modified_fields,
        )
        self._repository.record_review_decision(decision)
        self._forget_pending(extension)
        merged = self._merge_service.merge(updated_extension, active_ontology)
        logger.info("ontology_extension_modified_and_merged", extension_id=str(extension.id), reviewer=reviewer_id)
        return merged

    def merge_with(
        self,
        extension: OntologyExtension,
        active_ontology: ActiveOntology,
        *,
        target_class_name: str,
        reviewer_id: str,
        reviewer_role: ReviewerRole = ReviewerRole.TEACHER,
        comment: str | None = None,
    ) -> OntologyClass:
        """Reviewer decides the proposal duplicates an existing class under a different name —
        fold it in as an alias of `target_class_name` instead of creating a new class."""
        target = active_ontology.get_class(target_class_name)
        if target is None:
            raise OntologyError(f"Cannot merge extension {extension.id}: target class '{target_class_name}' not found")

        alias_extension = extension.model_copy(
            update={
                "kind": extension.kind.__class__("alias"),
                "proposed_parent_class_id": target.id,
            }
        )
        decision = ReviewDecision(
            target_id=extension.id,
            target_type="OntologyExtension",
            action=ReviewAction.MERGE,
            reviewer_role=reviewer_role,
            reviewer_id=reviewer_id,
            comment=comment,
            modified_payload={"merged_into": target_class_name},
        )
        self._repository.record_review_decision(decision)
        self._forget_pending(extension)
        merged = self._merge_service.merge(alias_extension, active_ontology)
        logger.info(
            "ontology_extension_merged_as_alias",
            extension_id=str(extension.id),
            target_class=target_class_name,
            reviewer=reviewer_id,
        )
        return merged  # type: ignore[return-value]

    def auto_review(
        self, extension: OntologyExtension, validation_result: ValidationResult, active_ontology: ActiveOntology
    ) -> tuple[bool, OntologyClass | OntologyRelation | None]:
        """
        Non-interactive review pass run by the pipeline itself (ReviewerRole.SYSTEM) for
        offline/batch runs with no human reviewer available yet. Only extensions that pass
        validation cleanly (VALID, not just WARNING) and clear the configured confidence
        threshold are auto-approved; everything else is left pending for a human reviewer.
        This never bypasses validation or the review/decision audit trail — it simply
        supplies the "reviewer".
        """
        threshold = get_settings().thresholds.ontology_extension_confidence_min
        if validation_result.status == ValidationStatus.VALID and extension.confidence.value >= threshold:
            merged = self.approve(
                extension,
                active_ontology,
                reviewer_id="pipeline-auto-reviewer",
                reviewer_role=ReviewerRole.SYSTEM,
                comment="Auto-approved: validation VALID and confidence above threshold",
            )
            return True, merged

        logger.info(
            "ontology_extension_left_pending",
            extension_id=str(extension.id),
            validation_status=validation_result.status.value,
            confidence=extension.confidence.value,
        )
        return False, None

    def list_pending(self, subject=None) -> list[OntologyExtension]:
        return self._repository.list_pending_extensions(subject)

    def _forget_pending(self, extension: OntologyExtension) -> None:
        remover = getattr(self._repository, "remove_pending_extension", None)
        if callable(remover):
            remover(extension.id)
