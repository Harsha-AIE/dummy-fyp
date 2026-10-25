from __future__ import annotations

from uuid import UUID

from pydantic import Field

from edukg.core.models.common import Identifiable, Provenance, Timestamped
from edukg.core.models.enums import EntityType


class Entity(Identifiable, Timestamped):
    """A canonical, resolved entity/concept — the linking target for all Mentions."""

    canonical_name: str = Field(min_length=1)
    entity_type: EntityType
    ontology_class_id: UUID | None = Field(
        default=None, description="Linked OntologyClass, if the entity type is ontology-governed"
    )
    aliases: list[str] = Field(default_factory=list)
    embedding: list[float] | None = Field(default=None, description="Dense vector for FAISS similarity search")
    mention_ids: list[UUID] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)
    description: str | None = None

    def add_alias(self, alias: str) -> None:
        normalized = alias.strip()
        if not normalized:
            return
        if normalized.lower() == self.canonical_name.lower():
            return
        if normalized.lower() in {a.lower() for a in self.aliases}:
            return
        self.aliases.append(normalized)

    def merge_provenance(self, other: Provenance) -> None:
        self.provenance.append(other)
