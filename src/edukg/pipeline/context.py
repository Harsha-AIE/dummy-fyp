from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from edukg.core.models import (
    Chunk,
    Document,
    Entity,
    Mention,
    OntologyClass,
    OntologyExtension,
    OntologyRelation,
    PipelineStage,
    Relation,
    Triple,
    ValidationResult,
)


@dataclass(slots=True)
class PipelineContext:
    """
    Mutable, in-memory carrier of everything produced as a Document flows through
    the pipeline. Each stage reads what it needs and appends its own output —
    it never mutates another stage's section.
    """

    document: Document
    chunks: list[Chunk] = field(default_factory=list)

    ontology_classes: list[OntologyClass] = field(default_factory=list)
    ontology_relations: list[OntologyRelation] = field(default_factory=list)
    proposed_extensions: list[OntologyExtension] = field(default_factory=list)
    accepted_extensions: list[OntologyExtension] = field(default_factory=list)

    mentions: list[Mention] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)

    validation_results: list[ValidationResult] = field(default_factory=list)

    completed_stages: list[PipelineStage] = field(default_factory=list)
    stage_errors: dict[str, str] = field(default_factory=dict)

    def mark_complete(self, stage: PipelineStage) -> None:
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)

    def entity_by_id(self, entity_id: UUID) -> Entity | None:
        return next((e for e in self.entities if e.id == entity_id), None)

    def is_complete_through(self, stage: PipelineStage) -> bool:
        return stage in self.completed_stages
