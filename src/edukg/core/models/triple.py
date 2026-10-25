from __future__ import annotations

from uuid import UUID

from pydantic import Field

from edukg.core.models.common import ConfidenceScore, Identifiable, Provenance
from edukg.core.models.enums import ValidationStatus


class Triple(Identifiable):
    """A strongly-typed subject-predicate-object fact, fully traceable to its source."""

    subject_entity_id: UUID
    predicate: str = Field(min_length=1)
    object_entity_id: UUID
    confidence: ConfidenceScore
    provenance: Provenance
    source_document_id: UUID
    page: int | None = Field(default=None, ge=0)
    chunk_id: UUID | None = None
    extractor_version: str = Field(min_length=1)
    relation_id: UUID
    validation_status: ValidationStatus = Field(default=ValidationStatus.PENDING)

    def natural_key(self) -> tuple[UUID, str, UUID]:
        """Identity used for duplicate detection during fusion (excludes confidence/provenance)."""
        return self.subject_entity_id, self.predicate, self.object_entity_id
