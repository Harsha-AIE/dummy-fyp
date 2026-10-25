from __future__ import annotations

from edukg.core.logging_setup import get_logger
from edukg.core.models import (
    OntologyExtension,
    OntologyExtensionKind,
    PipelineStage,
    ValidationIssue,
    ValidationResult,
    ValidationStatus,
)
from edukg.core.config import get_settings
from edukg.modules.ontology.active_ontology import ActiveOntology

logger = get_logger(__name__)


class OntologyValidationService:
    """
    Validates a single OntologyExtension proposal against the currently active
    ontology and against the other proposals in the same batch. Never mutates
    anything — returns a ValidationResult the review layer decides what to do with.
    """

    def __init__(self, confidence_threshold: float | None = None) -> None:
        self._confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else get_settings().thresholds.ontology_extension_confidence_min
        )

    def validate_batch(
        self, extensions: list[OntologyExtension], active_ontology: ActiveOntology
    ) -> list[ValidationResult]:
        seen_names: set[str] = set()
        results: list[ValidationResult] = []
        for extension in extensions:
            result = self.validate(extension, active_ontology, batch_seen_names=seen_names)
            seen_names.add(extension.proposed_name.strip().lower())
            results.append(result)
        return results

    def validate(
        self,
        extension: OntologyExtension,
        active_ontology: ActiveOntology,
        *,
        batch_seen_names: set[str] | None = None,
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        batch_seen_names = batch_seen_names or set()
        name_key = extension.proposed_name.strip().lower()

        issues.extend(self._check_duplicates(extension, active_ontology, name_key, batch_seen_names))
        issues.extend(self._check_structure(extension, active_ontology))
        issues.extend(self._check_confidence(extension))
        issues.extend(self._check_aliases(extension, active_ontology))

        blocking = any(i.severity == ValidationStatus.INVALID for i in issues)
        warning = any(i.severity == ValidationStatus.WARNING for i in issues)
        status = ValidationStatus.INVALID if blocking else (ValidationStatus.WARNING if warning else ValidationStatus.VALID)

        result = ValidationResult(
            stage=PipelineStage.ONTOLOGY_VALIDATION,
            target_id=extension.id,
            target_type="OntologyExtension",
            status=status,
            issues=issues,
        )
        logger.info(
            "ontology_extension_validated",
            extension_id=str(extension.id),
            proposed_name=extension.proposed_name,
            status=status.value,
            issue_count=len(issues),
        )
        return result

    # ---------- individual checks ----------

    def _check_duplicates(
        self, extension: OntologyExtension, active_ontology: ActiveOntology, name_key: str, batch_seen_names: set[str]
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if extension.kind == OntologyExtensionKind.CLASS and active_ontology.has_class(extension.proposed_name):
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_CLASS",
                    message=f"Class '{extension.proposed_name}' already exists in the active ontology",
                    severity=ValidationStatus.INVALID,
                    field_path="proposed_name",
                )
            )
        if extension.kind == OntologyExtensionKind.RELATION and active_ontology.has_relation(extension.proposed_name):
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_PREDICATE",
                    message=f"Relation '{extension.proposed_name}' already exists in the active ontology",
                    severity=ValidationStatus.INVALID,
                    field_path="proposed_name",
                )
            )
        if name_key in batch_seen_names:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_IN_BATCH",
                    message=f"'{extension.proposed_name}' was already proposed earlier in this batch",
                    severity=ValidationStatus.INVALID,
                    field_path="proposed_name",
                )
            )
        return issues

    def _check_structure(self, extension: OntologyExtension, active_ontology: ActiveOntology) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if extension.kind == OntologyExtensionKind.RELATION:
            if extension.proposed_domain_class_id is None:
                issues.append(
                    ValidationIssue(
                        code="MISSING_DOMAIN",
                        message="Relation proposal has no resolvable domain class",
                        severity=ValidationStatus.INVALID,
                        field_path="proposed_domain_class_id",
                    )
                )
            if extension.proposed_range_class_id is None:
                issues.append(
                    ValidationIssue(
                        code="MISSING_RANGE",
                        message="Relation proposal has no resolvable range class",
                        severity=ValidationStatus.INVALID,
                        field_path="proposed_range_class_id",
                    )
                )
            if (
                extension.proposed_domain_class_id is not None
                and extension.proposed_domain_class_id == extension.proposed_range_class_id
            ):
                issues.append(
                    ValidationIssue(
                        code="DOMAIN_RANGE_SAME_CLASS",
                        message="Relation domain and range resolve to the same class — verify this is intentional",
                        severity=ValidationStatus.WARNING,
                    )
                )

        if extension.kind == OntologyExtensionKind.ALIAS and extension.proposed_parent_class_id is None:
            issues.append(
                ValidationIssue(
                    code="ALIAS_MISSING_PARENT",
                    message="Alias proposal does not resolve to an existing parent class",
                    severity=ValidationStatus.INVALID,
                    field_path="proposed_parent_class_id",
                )
            )

        if not extension.justification.strip():
            issues.append(
                ValidationIssue(
                    code="MISSING_JUSTIFICATION",
                    message="Proposal has no justification",
                    severity=ValidationStatus.INVALID,
                    field_path="justification",
                )
            )
        return issues

    def _check_confidence(self, extension: OntologyExtension) -> list[ValidationIssue]:
        if extension.confidence.value < self._confidence_threshold:
            return [
                ValidationIssue(
                    code="LOW_CONFIDENCE",
                    message=(
                        f"Confidence {extension.confidence.value:.2f} is below threshold "
                        f"{self._confidence_threshold:.2f}"
                    ),
                    severity=ValidationStatus.WARNING,
                    field_path="confidence",
                )
            ]
        return []

    def _check_aliases(self, extension: OntologyExtension, active_ontology: ActiveOntology) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for alias in extension.proposed_aliases:
            if active_ontology.has_class(alias) or active_ontology.has_relation(alias):
                issues.append(
                    ValidationIssue(
                        code="ALIAS_COLLIDES_WITH_EXISTING",
                        message=f"Proposed alias '{alias}' collides with an existing ontology name",
                        severity=ValidationStatus.WARNING,
                        field_path="proposed_aliases",
                    )
                )
        return issues
