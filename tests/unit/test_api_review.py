"""Router functions are plain Python functions under FastAPI's Depends()
wiring, so they're called directly here with a lightweight fake resources
object — the same "call the handler, skip the framework" philosophy already
used to test pipeline stage handlers via .run() directly. No FastAPI
TestClient/real Azure/real Neo4j needed for this logic."""

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException

from edukg.api.routers import review
from edukg.core.models import (
    ConfidenceScore,
    Entity,
    EntityType,
    OntologyClass,
    Provenance,
    Subject,
    Triple,
)
from edukg.core.models.enums import ValidationStatus
from edukg.infrastructure.filesystem.entity_repository import FilesystemEntityRepository
from edukg.infrastructure.filesystem.ontology_repository import FilesystemOntologyRepository
from edukg.modules.fusion.pending_triple_store import PendingTripleStore
from edukg.modules.ontology.ontology_manager import OntologyManager
from edukg.modules.ontology.ontology_merge_service import OntologyMergeService
from edukg.modules.ontology.ontology_review_service import OntologyReviewService


class _FakeGraphRepo:
    def __init__(self) -> None:
        self.upserted_entities: list[Entity] = []
        self.upserted_triples: list[Triple] = []

    def upsert_entity(self, entity: Entity) -> None:
        self.upserted_entities.append(entity)

    def upsert_triple(self, triple: Triple, subject: Entity, obj: Entity) -> None:
        self.upserted_triples.append(triple)


@dataclass
class _FakeResources:
    entity_repo: FilesystemEntityRepository
    pending_triple_store: PendingTripleStore
    ontology_repo: FilesystemOntologyRepository
    ontology_manager: OntologyManager
    ontology_review_service: OntologyReviewService
    graph_repo: _FakeGraphRepo


def _make_resources(tmp_path: Path) -> _FakeResources:
    entity_repo = FilesystemEntityRepository(base_dir=tmp_path / "entities")
    ontology_repo = FilesystemOntologyRepository(base_dir=tmp_path / "ontologies")
    concept = OntologyClass(name="Concept", subject=Subject.UNKNOWN, is_core=True)
    ontology_repo.add_class(concept)
    from edukg.infrastructure.cache.filesystem_cache import FilesystemCacheRepository

    cache_repo = FilesystemCacheRepository(base_dir=tmp_path / "cache")
    ontology_manager = OntologyManager(ontology_repo, cache_repo)
    merge_service = OntologyMergeService(ontology_manager)
    ontology_review_service = OntologyReviewService(ontology_repo, merge_service)

    return _FakeResources(
        entity_repo=entity_repo,
        pending_triple_store=PendingTripleStore(base_dir=tmp_path / "pending_triples"),
        ontology_repo=ontology_repo,
        ontology_manager=ontology_manager,
        ontology_review_service=ontology_review_service,
        graph_repo=_FakeGraphRepo(),
    )


def _make_triple(subject_id, object_id, *, confidence: float = 0.4) -> Triple:
    return Triple(
        subject_entity_id=subject_id,
        predicate="is_related_to",
        object_entity_id=object_id,
        confidence=ConfidenceScore(value=confidence, method="test"),
        provenance=Provenance(source_document_id=uuid4(), extractor_name="test", extractor_version="1.0.0"),
        source_document_id=uuid4(),
        extractor_version="1.0.0",
        relation_id=uuid4(),
        validation_status=ValidationStatus.VALID,
    )


def test_approve_triple_writes_to_graph_and_removes_from_pending(tmp_path: Path) -> None:
    resources = _make_resources(tmp_path)
    subject = Entity(canonical_name="Subject", entity_type=EntityType.CONCEPT)
    obj = Entity(canonical_name="Object", entity_type=EntityType.CONCEPT)
    resources.entity_repo.save(subject)
    resources.entity_repo.save(obj)
    triple = _make_triple(subject.id, obj.id)
    resources.pending_triple_store.save(triple)

    result = review.approve_triple(triple.id, resources)

    assert result.status == "approved"
    assert resources.pending_triple_store.get(triple.id) is None
    assert len(resources.graph_repo.upserted_entities) == 2
    assert resources.graph_repo.upserted_triples[0].id == triple.id


def test_approve_triple_404_when_not_pending(tmp_path: Path) -> None:
    resources = _make_resources(tmp_path)
    with pytest.raises(HTTPException) as exc_info:
        review.approve_triple(uuid4(), resources)
    assert exc_info.value.status_code == 404


def test_approve_triple_409_when_entity_missing(tmp_path: Path) -> None:
    resources = _make_resources(tmp_path)
    triple = _make_triple(uuid4(), uuid4())  # entities never saved
    resources.pending_triple_store.save(triple)

    with pytest.raises(HTTPException) as exc_info:
        review.approve_triple(triple.id, resources)
    assert exc_info.value.status_code == 409
    assert resources.pending_triple_store.get(triple.id) is not None  # left untouched for retry


def test_reject_triple_removes_without_touching_graph(tmp_path: Path) -> None:
    resources = _make_resources(tmp_path)
    triple = _make_triple(uuid4(), uuid4())
    resources.pending_triple_store.save(triple)

    result = review.reject_triple(triple.id, resources)

    assert result.status == "rejected"
    assert resources.pending_triple_store.get(triple.id) is None
    assert resources.graph_repo.upserted_entities == []


def test_bulk_approve_only_affects_triples_at_or_above_threshold(tmp_path: Path) -> None:
    resources = _make_resources(tmp_path)
    entities = [Entity(canonical_name=f"E{i}", entity_type=EntityType.CONCEPT) for i in range(6)]
    for e in entities:
        resources.entity_repo.save(e)
    low = _make_triple(entities[0].id, entities[1].id, confidence=0.3)
    mid = _make_triple(entities[2].id, entities[3].id, confidence=0.7)
    high = _make_triple(entities[4].id, entities[5].id, confidence=0.9)
    for t in (low, mid, high):
        resources.pending_triple_store.save(t)

    result = review.bulk_approve_triples(review.BulkApproveRequest(min_confidence=0.7), resources)

    assert result.count == 2
    assert resources.pending_triple_store.get(low.id) is not None  # left pending, below threshold
    assert resources.pending_triple_store.get(mid.id) is None
    assert resources.pending_triple_store.get(high.id) is None
    assert {t.id for t in resources.graph_repo.upserted_triples} == {mid.id, high.id}


def test_bulk_approve_skips_triples_with_missing_entities_without_aborting_the_batch(tmp_path: Path) -> None:
    resources = _make_resources(tmp_path)
    good_subject = Entity(canonical_name="Good", entity_type=EntityType.CONCEPT)
    resources.entity_repo.save(good_subject)
    good = _make_triple(good_subject.id, good_subject.id, confidence=0.8)
    stale = _make_triple(uuid4(), uuid4(), confidence=0.8)  # entities never saved
    resources.pending_triple_store.save(good)
    resources.pending_triple_store.save(stale)

    result = review.bulk_approve_triples(review.BulkApproveRequest(min_confidence=0.5), resources)

    assert result.count == 1  # only the resolvable one counted
    assert resources.pending_triple_store.get(good.id) is None
    assert resources.pending_triple_store.get(stale.id) is not None  # left pending for retry, not silently dropped


def test_bulk_reject_only_affects_triples_at_or_below_threshold(tmp_path: Path) -> None:
    resources = _make_resources(tmp_path)
    low = _make_triple(uuid4(), uuid4(), confidence=0.3)
    mid = _make_triple(uuid4(), uuid4(), confidence=0.5)
    high = _make_triple(uuid4(), uuid4(), confidence=0.9)
    for t in (low, mid, high):
        resources.pending_triple_store.save(t)

    result = review.bulk_reject_triples(review.BulkRejectRequest(max_confidence=0.5), resources)

    assert result.count == 2
    assert resources.pending_triple_store.get(low.id) is None
    assert resources.pending_triple_store.get(mid.id) is None
    assert resources.pending_triple_store.get(high.id) is not None  # left pending, above threshold
    assert resources.graph_repo.upserted_triples == []  # reject never touches the graph


def test_list_pending_triples_resolves_entity_names(tmp_path: Path) -> None:
    resources = _make_resources(tmp_path)
    subject = Entity(canonical_name="Gradient Descent", entity_type=EntityType.CONCEPT)
    obj = Entity(canonical_name="Loss Function", entity_type=EntityType.CONCEPT)
    resources.entity_repo.save(subject)
    resources.entity_repo.save(obj)
    resources.pending_triple_store.save(_make_triple(subject.id, obj.id))

    result = review.list_pending_triples(resources=resources)

    assert len(result.items) == 1
    assert result.total == 1
    assert result.items[0].subject_name == "Gradient Descent"
    assert result.items[0].object_name == "Loss Function"


def test_list_pending_triples_handles_missing_entity_gracefully(tmp_path: Path) -> None:
    resources = _make_resources(tmp_path)
    resources.pending_triple_store.save(_make_triple(uuid4(), uuid4()))

    result = review.list_pending_triples(resources=resources)

    assert result.items[0].subject_name == "(unknown entity)"


def test_list_pending_triples_paginates_and_sorts_by_confidence_descending(tmp_path: Path) -> None:
    resources = _make_resources(tmp_path)
    for confidence in (0.35, 0.9, 0.6):
        resources.pending_triple_store.save(_make_triple(uuid4(), uuid4(), confidence=confidence))

    page1 = review.list_pending_triples(page=1, page_size=2, resources=resources)
    page2 = review.list_pending_triples(page=2, page_size=2, resources=resources)

    assert page1.total == 3
    assert [t.confidence for t in page1.items] == [0.9, 0.6]
    assert [t.confidence for t in page2.items] == [0.35]


def test_list_pending_triples_rejects_invalid_page_params(tmp_path: Path) -> None:
    resources = _make_resources(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        review.list_pending_triples(page=0, resources=resources)
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        review.list_pending_triples(page_size=9999, resources=resources)
    assert exc_info.value.status_code == 400
