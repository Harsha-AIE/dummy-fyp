from __future__ import annotations

from uuid import UUID

from edukg.core.config import get_settings
from edukg.core.logging_setup import get_logger
from edukg.core.models import Entity, PipelineStage, Triple, ValidationIssue, ValidationResult, ValidationStatus
from edukg.modules.ontology.active_ontology import ActiveOntology
from edukg.modules.triple_validation.shacl_validator import validate_triple as shacl_validate_triple

logger = get_logger(__name__)


class TripleValidationService:
    """
    Stage 10: domain/range constraint checking. Checks each triple's
    subject/object entity ontology class against the resolved ontology
    relation's domain_class_id/range_class_id directly in Python, reusing
    the exact ValidationResult/ValidationIssue pattern already proven in
    ontology_validator.py — then layers real pyshacl-backed SHACL validation
    on top (see shacl_validator.validate_triple) as an additional signal,
    satisfying the pipeline spec's literal "validates triples using
    pyshacl" requirement without discarding this check's own judgment call
    below, which pyshacl would hit the identical limitation on.

    Domain/range mismatches are WARNING, not INVALID (for both this check
    and the layered pyshacl one): entities are typed via a coarse
    EntityType->OntologyClass mapping during linking (see
    entity_linking/linking_service.py), so most entities resolve to the core
    "Concept" class rather than a precise domain class like "Model" or
    "Dataset" — treating that as a hard failure would reject nearly every
    domain-specific relation. A missing/unmatched predicate is INVALID since
    it should never happen (relation extraction only ever emits predicates
    that already resolved against the active ontology).
    """

    def validate_batch(
        self, triples: list[Triple], entities_by_id: dict[UUID, Entity], active_ontology: ActiveOntology
    ) -> list[ValidationResult]:
        return [self.validate(triple, entities_by_id, active_ontology) for triple in triples]

    def validate(
        self, triple: Triple, entities_by_id: dict[UUID, Entity], active_ontology: ActiveOntology
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        subject = entities_by_id.get(triple.subject_entity_id)
        obj = entities_by_id.get(triple.object_entity_id)
        relation = active_ontology.get_relation(triple.predicate)

        if relation is None:
            issues.append(
                ValidationIssue(
                    code="UNKNOWN_PREDICATE",
                    message=f"Predicate '{triple.predicate}' is not a known ontology relation",
                    severity=ValidationStatus.INVALID,
                    field_path="predicate",
                )
            )
        else:
            issues.extend(self._check_endpoint(subject, relation.domain_class_id, "subject", "DOMAIN"))
            issues.extend(self._check_endpoint(obj, relation.range_class_id, "object", "RANGE"))
            # Real pyshacl validation, layered on top rather than replacing
            # the checks above — see shacl_validator.validate_triple's
            # docstring. A pyshacl failure here (e.g. the package genuinely
            # isn't installed in some environment) must not break triple
            # validation itself; it only means this one extra signal is
            # unavailable for this triple.
            try:
                issues.extend(shacl_validate_triple(triple, subject, obj, active_ontology))
            except Exception as exc:  # noqa: BLE001
                logger.error("shacl_validation_failed", triple_id=str(triple.id), error=str(exc))

        min_confidence = get_settings().thresholds.triple_confidence_min
        if triple.confidence.value < min_confidence:
            issues.append(
                ValidationIssue(
                    code="LOW_CONFIDENCE",
                    message=f"Confidence {triple.confidence.value:.2f} is below threshold {min_confidence:.2f}",
                    severity=ValidationStatus.WARNING,
                    field_path="confidence",
                )
            )

        blocking = any(i.severity == ValidationStatus.INVALID for i in issues)
        warning = any(i.severity == ValidationStatus.WARNING for i in issues)
        status = ValidationStatus.INVALID if blocking else (ValidationStatus.WARNING if warning else ValidationStatus.VALID)

        return ValidationResult(
            stage=PipelineStage.TRIPLE_VALIDATION, target_id=triple.id, target_type="Triple", status=status, issues=issues
        )

    def _check_endpoint(
        self, entity: Entity | None, expected_class_id: UUID, role: str, code_prefix: str
    ) -> list[ValidationIssue]:
        if entity is None:
            return [
                ValidationIssue(
                    code=f"{code_prefix}_ENTITY_MISSING",
                    message=f"Triple's {role} entity could not be resolved",
                    severity=ValidationStatus.WARNING,
                    field_path=f"{role}_entity_id",
                )
            ]
        if entity.ontology_class_id is None:
            return [
                ValidationIssue(
                    code=f"{code_prefix}_CLASS_UNRESOLVED",
                    message=f"Triple's {role} entity has no resolved ontology class",
                    severity=ValidationStatus.WARNING,
                    field_path=f"{role}_entity_id",
                )
            ]
        if entity.ontology_class_id != expected_class_id:
            return [
                ValidationIssue(
                    code=f"{code_prefix}_MISMATCH",
                    message=f"Triple's {role} entity class does not match the relation's expected class",
                    severity=ValidationStatus.WARNING,
                    field_path=f"{role}_entity_id",
                )
            ]
        return []
