from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from uuid import UUID

from edukg.core.models import (
    Chunk,
    Document,
    Entity,
    OntologyClass,
    OntologyExtension,
    OntologyRelation,
    ReviewDecision,
    Subject,
    Triple,
)


class DocumentRepository(ABC):
    @abstractmethod
    def save(self, document: Document) -> Document: ...

    @abstractmethod
    def get(self, document_id: UUID) -> Document | None: ...

    @abstractmethod
    def get_by_hash(self, file_hash: str) -> Document | None: ...

    @abstractmethod
    def save_chunks(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def get_chunks(self, document_id: UUID) -> list[Chunk]: ...

    @abstractmethod
    def list_all_documents(self) -> list[Document]: ...


class OntologyRepository(ABC):
    """Read/write access to the active ontology store. Never called directly by business logic
    outside the OntologyManager service — all ontology mutation flows through it."""

    @abstractmethod
    def load_core_ontology(self) -> tuple[list[OntologyClass], list[OntologyRelation]]: ...

    @abstractmethod
    def load_domain_ontology(self, subject: Subject) -> tuple[list[OntologyClass], list[OntologyRelation]]: ...

    @abstractmethod
    def get_class(self, class_id: UUID) -> OntologyClass | None: ...

    @abstractmethod
    def get_relation(self, relation_id: UUID) -> OntologyRelation | None: ...

    @abstractmethod
    def find_class_by_name(self, name: str, subject: Subject) -> OntologyClass | None: ...

    @abstractmethod
    def find_relation_by_name(self, name: str, subject: Subject) -> OntologyRelation | None: ...

    @abstractmethod
    def add_class(self, ontology_class: OntologyClass) -> OntologyClass: ...

    @abstractmethod
    def add_relation(self, relation: OntologyRelation) -> OntologyRelation: ...

    @abstractmethod
    def save_extension_proposal(self, extension: OntologyExtension) -> OntologyExtension: ...

    @abstractmethod
    def list_pending_extensions(self, subject: Subject | None = None) -> list[OntologyExtension]: ...

    @abstractmethod
    def record_review_decision(self, decision: ReviewDecision) -> ReviewDecision: ...


class EntityRepository(ABC):
    @abstractmethod
    def save(self, entity: Entity) -> Entity: ...

    @abstractmethod
    def get(self, entity_id: UUID) -> Entity | None: ...

    @abstractmethod
    def find_by_canonical_name(self, name: str) -> Entity | None: ...

    @abstractmethod
    def list_all(self) -> list[Entity]: ...


class GraphRepository(ABC):
    """Sole point of contact with Neo4j. Business logic depends on this interface only."""

    @abstractmethod
    def upsert_entity(self, entity: Entity) -> None: ...

    @abstractmethod
    def upsert_triple(self, triple: Triple, subject: Entity, obj: Entity) -> None: ...

    @abstractmethod
    def batch_upsert_triples(self, triples: list[Triple], entities_by_id: dict[UUID, Entity]) -> None: ...

    @abstractmethod
    def ensure_constraints(self) -> None: ...

    @abstractmethod
    def shortest_path_length(self, from_entity_id: UUID, to_entity_id: UUID, max_hops: int) -> int | None:
        """Hop count of the shortest path between two entities, bounded by
        max_hops. None if no path exists within that bound (not "infinite
        distance" — callers decide how to treat "no known path")."""
        ...

    @abstractmethod
    def close(self) -> None: ...


class VectorIndexRepository(ABC):
    """FAISS-backed similarity search used by entity linking."""

    @abstractmethod
    def add(self, entity_id: UUID, embedding: list[float]) -> None: ...

    @abstractmethod
    def search(self, embedding: list[float], top_k: int) -> list[tuple[UUID, float]]: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...


class CacheRepository(ABC):
    @abstractmethod
    def get(self, key: str) -> bytes | None: ...

    @abstractmethod
    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...
