from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from edukg.core.config import get_settings
from edukg.core.exceptions import CacheError
from edukg.core.interfaces import OntologyRepository
from edukg.core.logging_setup import get_logger
from edukg.core.models import (
    OntologyClass,
    OntologyExtension,
    OntologyRelation,
    ReviewDecision,
    Subject,
)

logger = get_logger(__name__)


class FilesystemOntologyRepository(OntologyRepository):
    """
    Persists the ontology as JSON under data/ontologies:

      core.json                    -> {"classes": [...], "relations": [...]}
      domains/<subject>.json       -> {"classes": [...], "relations": [...]}
      extensions/pending/<id>.json -> OntologyExtension awaiting review
      reviews/<id>.json            -> ReviewDecision, filename = decision.id

    In-memory caches are populated lazily and kept in sync on every write so
    get_class/get_relation/find_*_by_name are O(1) after first load.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or get_settings().paths.ontology_dir
        self._domains_dir = self._base_dir / "domains"
        self._extensions_dir = self._base_dir / "extensions" / "pending"
        self._reviews_dir = self._base_dir / "reviews"
        for d in (self._base_dir, self._domains_dir, self._extensions_dir, self._reviews_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._classes_by_id: dict[UUID, OntologyClass] = {}
        self._relations_by_id: dict[UUID, OntologyRelation] = {}
        self._core_loaded = False
        self._domains_loaded: set[Subject] = set()

    # ---------- internal file helpers ----------

    def _core_path(self) -> Path:
        return self._base_dir / "core.json"

    def _domain_path(self, subject: Subject) -> Path:
        return self._domains_dir / f"{subject.value}.json"

    def _read_bundle(self, path: Path) -> tuple[list[OntologyClass], list[OntologyRelation]]:
        if not path.exists():
            return [], []
        raw = json.loads(path.read_text(encoding="utf-8"))
        classes = [OntologyClass.model_validate(c) for c in raw.get("classes", [])]
        relations = [OntologyRelation.model_validate(r) for r in raw.get("relations", [])]
        return classes, relations

    def _write_bundle(self, path: Path, classes: list[OntologyClass], relations: list[OntologyRelation]) -> None:
        payload = {
            "classes": [c.model_dump(mode="json") for c in classes],
            "relations": [r.model_dump(mode="json") for r in relations],
        }
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            raise CacheError(f"Failed writing ontology bundle to {path}") from exc

    def _index(self, classes: list[OntologyClass], relations: list[OntologyRelation]) -> None:
        for c in classes:
            self._classes_by_id[c.id] = c
        for r in relations:
            self._relations_by_id[r.id] = r

    # ---------- OntologyRepository interface ----------

    def load_core_ontology(self) -> tuple[list[OntologyClass], list[OntologyRelation]]:
        classes, relations = self._read_bundle(self._core_path())
        self._index(classes, relations)
        self._core_loaded = True
        logger.info("core_ontology_loaded", class_count=len(classes), relation_count=len(relations))
        return classes, relations

    def load_domain_ontology(self, subject: Subject) -> tuple[list[OntologyClass], list[OntologyRelation]]:
        classes, relations = self._read_bundle(self._domain_path(subject))
        self._index(classes, relations)
        self._domains_loaded.add(subject)
        logger.info(
            "domain_ontology_loaded", subject=subject.value, class_count=len(classes), relation_count=len(relations)
        )
        return classes, relations

    def get_class(self, class_id: UUID) -> OntologyClass | None:
        return self._classes_by_id.get(class_id)

    def get_relation(self, relation_id: UUID) -> OntologyRelation | None:
        return self._relations_by_id.get(relation_id)

    def find_class_by_name(self, name: str, subject: Subject) -> OntologyClass | None:
        target = name.strip().lower()
        for c in self._classes_by_id.values():
            if c.name.strip().lower() == target and (c.is_core or c.subject == subject):
                return c
            if target in {a.strip().lower() for a in c.aliases} and (c.is_core or c.subject == subject):
                return c
        return None

    def find_relation_by_name(self, name: str, subject: Subject) -> OntologyRelation | None:
        target = name.strip().lower()
        for r in self._relations_by_id.values():
            if r.name.strip().lower() == target and r.subject == subject:
                return r
        return None

    def add_class(self, ontology_class: OntologyClass) -> OntologyClass:
        path = self._core_path() if ontology_class.is_core else self._domain_path(ontology_class.subject)
        classes, relations = self._read_bundle(path)
        classes = [c for c in classes if c.id != ontology_class.id]
        classes.append(ontology_class)
        self._write_bundle(path, classes, relations)
        self._index([ontology_class], [])
        logger.info("ontology_class_added", name=ontology_class.name, subject=ontology_class.subject.value)
        return ontology_class

    def add_relation(self, relation: OntologyRelation) -> OntologyRelation:
        # OntologyRelation (Module 1, frozen) has no is_core flag the way OntologyClass does.
        # Convention (matches seed data): subject == Subject.UNKNOWN means "core relation".
        path = self._core_path() if relation.subject == Subject.UNKNOWN else self._domain_path(relation.subject)
        classes, relations = self._read_bundle(path)
        relations = [r for r in relations if r.id != relation.id]
        relations.append(relation)
        self._write_bundle(path, classes, relations)
        self._index([], [relation])
        logger.info("ontology_relation_added", name=relation.name, subject=relation.subject.value)
        return relation

    def save_extension_proposal(self, extension: OntologyExtension) -> OntologyExtension:
        path = self._extensions_dir / f"{extension.id}.json"
        try:
            path.write_text(json.dumps(extension.model_dump(mode="json"), indent=2), encoding="utf-8")
        except OSError as exc:
            raise CacheError(f"Failed saving ontology extension {extension.id}") from exc
        logger.info("ontology_extension_saved", extension_id=str(extension.id), kind=extension.kind.value)
        return extension

    def list_pending_extensions(self, subject: Subject | None = None) -> list[OntologyExtension]:
        results: list[OntologyExtension] = []
        for path in sorted(self._extensions_dir.glob("*.json")):
            extension = OntologyExtension.model_validate(json.loads(path.read_text(encoding="utf-8")))
            if subject is None or extension.subject == subject:
                results.append(extension)
        return results

    def remove_pending_extension(self, extension_id: UUID) -> None:
        path = self._extensions_dir / f"{extension_id}.json"
        path.unlink(missing_ok=True)

    def record_review_decision(self, decision: ReviewDecision) -> ReviewDecision:
        path = self._reviews_dir / f"{decision.id}.json"
        try:
            path.write_text(json.dumps(decision.model_dump(mode="json"), indent=2), encoding="utf-8")
        except OSError as exc:
            raise CacheError(f"Failed recording review decision {decision.id}") from exc
        logger.info(
            "review_decision_recorded",
            decision_id=str(decision.id),
            target_id=str(decision.target_id),
            action=decision.action.value,
        )
        return decision
