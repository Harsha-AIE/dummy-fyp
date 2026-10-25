from __future__ import annotations

from dataclasses import dataclass, field

from edukg.core.models import OntologyClass, OntologyRelation, Subject


@dataclass(slots=True)
class ActiveOntology:
    """
    Read-optimized, in-memory view of the ontology available to a given subject:
    core classes/relations (always included) merged with that subject's domain
    classes/relations. This is what the rest of the pipeline (entity extraction,
    relation extraction, triple generation) actually queries.
    """

    subject: Subject
    version: str
    classes_by_name: dict[str, OntologyClass] = field(default_factory=dict)
    relations_by_name: dict[str, OntologyRelation] = field(default_factory=dict)

    @property
    def classes(self) -> list[OntologyClass]:
        return list(self.classes_by_name.values())

    @property
    def relations(self) -> list[OntologyRelation]:
        return list(self.relations_by_name.values())

    def has_class(self, name: str) -> bool:
        return name.strip().lower() in {k.lower() for k in self.classes_by_name}

    def has_relation(self, name: str) -> bool:
        return name.strip().lower() in {k.lower() for k in self.relations_by_name}

    def get_class(self, name: str) -> OntologyClass | None:
        target = name.strip().lower()
        for key, value in self.classes_by_name.items():
            if key.lower() == target:
                return value
        return None

    def get_relation(self, name: str) -> OntologyRelation | None:
        target = name.strip().lower()
        for key, value in self.relations_by_name.items():
            if key.lower() == target:
                return value
        return None

    def summary(self, max_items: int = 40) -> str:
        """Compact text summary used when prompting the LLM for ontology extensions."""
        class_names = ", ".join(sorted(self.classes_by_name)[:max_items])
        relation_names = ", ".join(sorted(self.relations_by_name)[:max_items])
        return f"Existing classes: [{class_names}]\nExisting relations: [{relation_names}]"
