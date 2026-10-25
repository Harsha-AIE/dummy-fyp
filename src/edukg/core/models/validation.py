from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from edukg.core.models.common import Identifiable, Timestamped
from edukg.core.models.enums import PipelineStage, ReviewAction, ReviewerRole, ValidationStatus


class ValidationIssue(BaseModel):
    code: str = Field(min_length=1, description="Machine-readable issue code, e.g. 'DUPLICATE_CLASS'")
    message: str = Field(min_length=1)
    severity: ValidationStatus = Field(default=ValidationStatus.WARNING)
    field_path: str | None = None


class ValidationResult(Identifiable):
    """Result of validating any artifact (ontology extension, triple, etc.) at a given stage."""

    stage: PipelineStage
    target_id: UUID
    target_type: str = Field(min_length=1, description="e.g. 'OntologyExtension', 'Triple'")
    status: ValidationStatus
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status in (ValidationStatus.VALID, ValidationStatus.WARNING)

    @property
    def has_blocking_issues(self) -> bool:
        return any(i.severity == ValidationStatus.INVALID for i in self.issues)


class ReviewDecision(Identifiable, Timestamped):
    """A human decision on a proposed ontology extension (or other reviewable artifact)."""

    target_id: UUID
    target_type: str = Field(min_length=1)
    action: ReviewAction
    reviewer_role: ReviewerRole
    reviewer_id: str = Field(min_length=1)
    comment: str | None = None
    modified_payload: dict | None = Field(
        default=None, description="Populated only when action == MODIFY; replaces proposed fields"
    )
