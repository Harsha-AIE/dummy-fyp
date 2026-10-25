from __future__ import annotations

from edukg.core.models import Relation, Triple
from edukg.core.models.enums import ValidationStatus


class TripleGenerator:
    """Stage 9: projects each candidate Relation into a strongly-typed
    Subject-Predicate-Object Triple, carrying provenance through unchanged.
    Validation status starts PENDING — stage 10 (Triple Validation) is the
    only place that changes it."""

    def generate(self, relations: list[Relation]) -> list[Triple]:
        return [
            Triple(
                subject_entity_id=relation.subject_entity_id,
                predicate=relation.predicate,
                object_entity_id=relation.object_entity_id,
                confidence=relation.confidence,
                provenance=relation.provenance,
                source_document_id=relation.provenance.source_document_id,
                page=None,
                chunk_id=relation.provenance.chunk_id,
                extractor_version=relation.provenance.extractor_version,
                relation_id=relation.id,
                validation_status=ValidationStatus.PENDING,
            )
            for relation in relations
        ]
