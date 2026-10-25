from __future__ import annotations

from edukg.core.interfaces import OntologyRepository
from edukg.core.logging_setup import get_logger
from edukg.core.models import OntologyClass, OntologyRelation, Subject

logger = get_logger(__name__)


class DomainOntologyLoader:
    """Loads the domain-specific ontology for a given Subject, cached per subject for
    the lifetime of the process. A subject with no seed file yet is not an error —
    it simply starts with zero domain classes/relations and grows entirely through
    the ontology-extension pipeline."""

    def __init__(self, repository: OntologyRepository) -> None:
        self._repository = repository
        self._cache: dict[Subject, tuple[list[OntologyClass], list[OntologyRelation]]] = {}

    def load(self, subject: Subject, *, force_reload: bool = False) -> tuple[list[OntologyClass], list[OntologyRelation]]:
        if subject in self._cache and not force_reload:
            return self._cache[subject]

        classes, relations = self._repository.load_domain_ontology(subject)
        if not classes and not relations:
            logger.warning("domain_ontology_empty", subject=subject.value)

        self._cache[subject] = (classes, relations)
        return classes, relations

    def invalidate(self, subject: Subject) -> None:
        self._cache.pop(subject, None)
