from __future__ import annotations

from uuid import UUID

from pydantic import Field, model_validator

from edukg.core.models.common import ConfidenceScore, Identifiable, Provenance
from edukg.core.models.enums import RelationDirection


class Relation(Identifiable):
    """A typed, ontology-constrained candidate relation between two Entities, pre-triple."""

    subject_entity_id: UUID
    predicate: str = Field(min_length=1, description="Must match an OntologyRelation.name")
    object_entity_id: UUID
    direction: RelationDirection = Field(default=RelationDirection.DIRECTED)
    confidence: ConfidenceScore
    provenance: Provenance
    ontology_relation_id: UUID

    @model_validator(mode="after")
    def _no_self_relation_unless_reflexive(self) -> "Relation":
        if self.subject_entity_id == self.object_entity_id:
            raise ValueError("subject_entity_id and object_entity_id must differ")
        return self
