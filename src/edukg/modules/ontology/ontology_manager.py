from __future__ import annotations

from uuid import UUID

from edukg.core.interfaces import CacheRepository, OntologyRepository
from edukg.core.logging_setup import get_logger
from edukg.core.models import OntologyClass, OntologyRelation, Subject
from edukg.modules.ontology.active_ontology import ActiveOntology
from edukg.modules.ontology.core_ontology_loader import CoreOntologyLoader
from edukg.modules.ontology.domain_ontology_loader import DomainOntologyLoader

logger = get_logger(__name__)

_VERSION_CACHE_PREFIX = "ontology_version"


class OntologyManager:
    """
    Owns the full lifecycle of the active ontology for the pipeline:
    load core, load domain, merge into an ActiveOntology, and expose a single
    clean read/write API. Nothing outside this module talks to
    OntologyRepository, CoreOntologyLoader, or DomainOntologyLoader directly.
    """

    def __init__(
        self,
        repository: OntologyRepository,
        cache: CacheRepository,
        core_loader: CoreOntologyLoader | None = None,
        domain_loader: DomainOntologyLoader | None = None,
    ) -> None:
        self._repository = repository
        self._cache = cache
        self._core_loader = core_loader or CoreOntologyLoader(repository)
        self._domain_loader = domain_loader or DomainOntologyLoader(repository)

    # ---------- versioning (backed by CacheRepository — no new repository methods needed) ----------

    def _version_key(self, subject: Subject) -> str:
        return f"{_VERSION_CACHE_PREFIX}:{subject.value}"

    def current_version(self, subject: Subject) -> str:
        cache = self._cache.get(self._version_key(subject))
        revision = int(cache.decode("utf-8")) if cache else 0
        return f"1.{revision}.0"

    def _bump_version(self, subject: Subject) -> str:
        raw = self._cache.get(self._version_key(subject))
        revision = int(raw.decode("utf-8")) if raw else 0
        revision += 1
        self._cache.set(self._version_key(subject), str(revision).encode("utf-8"))
        return f"1.{revision}.0"

    # ---------- read API ----------

    def get_active_ontology(self, subject: Subject, *, force_reload: bool = False) -> ActiveOntology:
        core_classes, core_relations = self._core_loader.load(force_reload=force_reload)
        domain_classes, domain_relations = self._domain_loader.load(subject, force_reload=force_reload)

        classes_by_name, id_remap = self._merge_classes(core_classes, domain_classes)
        relations_by_name = self._merge_relations(core_relations, domain_relations, id_remap)

        active = ActiveOntology(
            subject=subject,
            version=self.current_version(subject),
            classes_by_name=classes_by_name,
            relations_by_name=relations_by_name,
        )
        logger.info(
            "active_ontology_built",
            subject=subject.value,
            class_count=len(active.classes_by_name),
            relation_count=len(active.relations_by_name),
            version=active.version,
        )
        return active

    def _merge_classes(
        self, core_classes: list[OntologyClass], domain_classes: list[OntologyClass]
    ) -> tuple[dict[str, OntologyClass], dict[UUID, UUID]]:
        """Merges core + domain classes keyed case-insensitively by name, so
        a domain ontology that happens to reuse a core class's name (e.g.
        "Concept") doesn't silently evict it via plain dict overwrite — the
        previous `classes_by_name.update(...)` pattern did exactly that,
        orphaning any existing parent_class_id/domain_class_id/range_class_id
        reference to the evicted core class's UUID with no error, no log,
        and no way to detect it short of a KG audit. Found via a cross-check
        against a teammate's independently-built codebase, which had already
        hit and fixed this. The core class always wins on a name collision
        (it's the shared, foundational definition); the domain duplicate's
        name/aliases are merged in via a non-mutating `.model_copy()` (NOT an
        in-place mutation — core_classes/domain_classes come from
        CoreOntologyLoader/DomainOntologyLoader's own cache, which
        `force_reload` exists specifically to invalidate; mutating a cached
        object here would leak merged aliases across every later call, for
        every subject, not just this one). Returns the merged map plus an
        id_remap so _merge_relations can rewrite any relation that pointed at
        a dropped duplicate's id.
        """
        by_key: dict[str, OntologyClass] = {c.name.lower(): c for c in core_classes if c.is_core}
        id_remap: dict[UUID, UUID] = {}
        for domain_class in domain_classes:
            key = domain_class.name.lower()
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = domain_class
                continue

            merged_aliases = list(existing.aliases)
            for alias in (domain_class.name, *domain_class.aliases):
                if alias.lower() != existing.name.lower() and alias.lower() not in {a.lower() for a in merged_aliases}:
                    merged_aliases.append(alias)
            by_key[key] = existing.model_copy(update={"aliases": merged_aliases})
            id_remap[domain_class.id] = existing.id
            logger.warning(
                "ontology_class_name_collision_resolved",
                name=domain_class.name,
                subject=domain_class.subject.value,
                kept_class_id=str(existing.id),
                dropped_class_id=str(domain_class.id),
            )
        return {c.name: c for c in by_key.values()}, id_remap

    def _merge_relations(
        self,
        core_relations: list[OntologyRelation],
        domain_relations: list[OntologyRelation],
        id_remap: dict[UUID, UUID],
    ) -> dict[str, OntologyRelation]:
        """Same dedup reasoning as _merge_classes for relation names — core
        wins on a name collision, the domain duplicate is dropped (relations
        aren't referenced by id from elsewhere the way classes are, so there's
        nothing to alias-merge or id-remap for the dropped one itself).
        Separately, every *kept* relation's domain_class_id/range_class_id is
        rewritten through id_remap, so a relation that referenced a class id
        dropped by _merge_classes still resolves to the class that was
        actually kept, instead of silently pointing at a UUID that no longer
        appears anywhere in the merged ActiveOntology."""

        def _remapped(relation: OntologyRelation) -> OntologyRelation:
            new_domain = id_remap.get(relation.domain_class_id, relation.domain_class_id)
            new_range = id_remap.get(relation.range_class_id, relation.range_class_id)
            if new_domain == relation.domain_class_id and new_range == relation.range_class_id:
                return relation
            return relation.model_copy(update={"domain_class_id": new_domain, "range_class_id": new_range})

        by_key: dict[str, OntologyRelation] = {r.name.lower(): _remapped(r) for r in core_relations}
        for domain_relation in domain_relations:
            key = domain_relation.name.lower()
            if key in by_key:
                logger.warning(
                    "ontology_relation_name_collision_resolved",
                    name=domain_relation.name,
                    subject=domain_relation.subject.value,
                    kept_relation_id=str(by_key[key].id),
                    dropped_relation_id=str(domain_relation.id),
                )
                continue
            by_key[key] = _remapped(domain_relation)
        return {r.name: r for r in by_key.values()}

    def find_class(self, name: str, subject: Subject) -> OntologyClass | None:
        return self._repository.find_class_by_name(name, subject)

    def find_relation(self, name: str, subject: Subject) -> OntologyRelation | None:
        return self._repository.find_relation_by_name(name, subject)

    # ---------- write API — the ONLY path by which classes/relations are added ----------

    def add_class(self, ontology_class: OntologyClass) -> OntologyClass:
        saved = self._repository.add_class(ontology_class)
        self._domain_loader.invalidate(ontology_class.subject)
        self._bump_version(ontology_class.subject)
        return saved

    def add_relation(self, relation: OntologyRelation) -> OntologyRelation:
        saved = self._repository.add_relation(relation)
        self._domain_loader.invalidate(relation.subject)
        self._bump_version(relation.subject)
        return saved
