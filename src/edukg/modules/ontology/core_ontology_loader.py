from __future__ import annotations

from edukg.core.exceptions import OntologyError
from edukg.core.interfaces import OntologyRepository
from edukg.core.logging_setup import get_logger
from edukg.core.models import OntologyClass, OntologyRelation

logger = get_logger(__name__)


class CoreOntologyLoader:
    """Loads the Core Ontology (classes/relations shared across every subject) exactly once
    per process and serves cached results afterward. The core ontology must contain at
    least one class — an empty core ontology indicates a missing/misconfigured seed file."""

    def __init__(self, repository: OntologyRepository) -> None:
        self._repository = repository
        self._classes: list[OntologyClass] | None = None
        self._relations: list[OntologyRelation] | None = None

    def load(self, *, force_reload: bool = False) -> tuple[list[OntologyClass], list[OntologyRelation]]:
        if self._classes is not None and self._relations is not None and not force_reload:
            return self._classes, self._relations

        classes, relations = self._repository.load_core_ontology()
        if not classes:
            raise OntologyError("Core ontology is empty or missing — seed data/ontologies/core.json")

        self._classes = classes
        self._relations = relations
        logger.info("core_ontology_ready", class_count=len(classes), relation_count=len(relations))
        return classes, relations
