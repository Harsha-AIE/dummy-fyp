"""Integration test against a real Neo4j instance (stage 12). Skipped
automatically if the configured instance (EDUKG_NEO4J_* in .env) isn't
reachable — this suite is not part of the no-external-services unit suite."""

from uuid import uuid4

import pytest

from edukg.core.config import get_settings
from edukg.core.models import ConfidenceScore, Entity, Provenance, Triple
from edukg.core.models.enums import EntityType, ValidationStatus
from edukg.infrastructure.neo4j.neo4j_graph_repository import Neo4jGraphRepository
from edukg.modules.storage.graph_storage_service import GraphStorageService


def _delete_test_entities(*entity_ids) -> None:
    """Standalone cleanup connection, deliberately not reusing the
    Neo4jGraphRepository under test, so teardown doesn't depend on the very
    code the test exists to exercise."""
    from neo4j import GraphDatabase

    settings = get_settings().neo4j
    driver = GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password))
    try:
        with driver.session(database=settings.database) as session:
            session.run(
                "MATCH (e:Entity) WHERE e.id IN $ids DETACH DELETE e", ids=[str(i) for i in entity_ids]
            )
    finally:
        driver.close()


@pytest.fixture
def graph_repo():
    repo = Neo4jGraphRepository()
    try:
        repo.ensure_constraints()
    except Exception as exc:  # noqa: BLE001 - any connectivity failure skips this suite
        pytest.skip(f"Neo4j not reachable, skipping integration test: {exc}")
    yield repo
    repo.close()


def test_store_upserts_entities_and_triples(graph_repo) -> None:
    subject = Entity(canonical_name=f"TestSubject-{uuid4()}", entity_type=EntityType.CONCEPT)
    obj = Entity(canonical_name=f"TestObject-{uuid4()}", entity_type=EntityType.CONCEPT)
    triple = Triple(
        subject_entity_id=subject.id,
        predicate="is_related_to",
        object_entity_id=obj.id,
        confidence=ConfidenceScore(value=0.9, method="test"),
        provenance=Provenance(source_document_id=uuid4(), extractor_name="test", extractor_version="1.0.0"),
        source_document_id=uuid4(),
        extractor_version="1.0.0",
        relation_id=uuid4(),
        validation_status=ValidationStatus.VALID,
    )

    service = GraphStorageService(graph_repo)
    try:
        service.store([subject, obj], [triple])

        # Re-running with the same ids must not create duplicate relationships —
        # MERGE should increment evidence_count instead.
        service.store([subject, obj], [triple])

        assert graph_repo.shortest_path_length(subject.id, obj.id, max_hops=4) == 1
        assert graph_repo.shortest_path_length(subject.id, subject.id, max_hops=4) == 0
        assert graph_repo.shortest_path_length(subject.id, uuid4(), max_hops=4) is None

        # Regression test: shortest_path_lengths_bulk once crashed the whole
        # request (Neo4j's shortestPath() forbids identical start/end nodes)
        # whenever a from_id also appeared in to_ids — exactly the common
        # "query concept == chunk concept" case the re-ranker hits constantly.
        missing = uuid4()
        bulk = graph_repo.shortest_path_lengths_bulk([subject.id], [obj.id, subject.id, missing], max_hops=4)
        assert bulk[(subject.id, obj.id)] == 1
        assert bulk[(subject.id, subject.id)] == 0
        assert (subject.id, missing) not in bulk
    finally:
        # This test writes to whatever real Neo4j instance is configured (not
        # a throwaway per-test DB) — it must not leave orphan nodes behind on
        # every run, so clean up unconditionally even if an assertion fails.
        _delete_test_entities(subject.id, obj.id)


def _triple(subject_id, predicate, object_id) -> Triple:
    return Triple(
        subject_entity_id=subject_id,
        predicate=predicate,
        object_entity_id=object_id,
        confidence=ConfidenceScore(value=0.9, method="test"),
        provenance=Provenance(source_document_id=uuid4(), extractor_name="test", extractor_version="1.0.0"),
        source_document_id=uuid4(),
        extractor_version="1.0.0",
        relation_id=uuid4(),
        validation_status=ValidationStatus.VALID,
    )


def test_materialize_transitive_closure_infers_a_to_c_from_a_chain(graph_repo) -> None:
    a = Entity(canonical_name=f"A-{uuid4()}", entity_type=EntityType.CONCEPT)
    b = Entity(canonical_name=f"B-{uuid4()}", entity_type=EntityType.CONCEPT)
    c = Entity(canonical_name=f"C-{uuid4()}", entity_type=EntityType.CONCEPT)
    predicate = "is_prerequisite_of"

    service = GraphStorageService(graph_repo)
    try:
        # A -> B -> C, chained — no direct A -> C edge exists yet.
        service.store([a, b, c], [_triple(a.id, predicate, b.id), _triple(b.id, predicate, c.id)])

        inferred_count = graph_repo.materialize_transitive_closure(predicate, max_hops=6)

        assert inferred_count == 1
        assert graph_repo.shortest_path_length(a.id, c.id, max_hops=1) == 1  # a direct edge now exists

        # Re-running must not duplicate the inferred edge or the original
        # chain — the "NOT (a)-[:RELATES_TO {predicate}]->(b)" guard in the
        # Cypher should make this idempotent.
        second_count = graph_repo.materialize_transitive_closure(predicate, max_hops=6)
        assert second_count == 0
    finally:
        _delete_test_entities(a.id, b.id, c.id)


def test_get_prerequisites_bulk_returns_direct_prerequisites_only(graph_repo) -> None:
    a = Entity(canonical_name=f"A-{uuid4()}", entity_type=EntityType.CONCEPT)
    b = Entity(canonical_name=f"B-{uuid4()}", entity_type=EntityType.CONCEPT)
    c = Entity(canonical_name=f"C-{uuid4()}", entity_type=EntityType.CONCEPT)
    unrelated = Entity(canonical_name=f"U-{uuid4()}", entity_type=EntityType.CONCEPT)
    predicate = "is_prerequisite_of"

    service = GraphStorageService(graph_repo)
    try:
        # A -> B -> C chained (A is prerequisite of B, B is prerequisite of C).
        service.store(
            [a, b, c, unrelated], [_triple(a.id, predicate, b.id), _triple(b.id, predicate, c.id)]
        )

        result = graph_repo.get_prerequisites_bulk([b.id, c.id, unrelated.id], predicate=predicate)

        assert result[b.id] == [a.id]  # B's direct prerequisite is A
        assert result[c.id] == [b.id]  # C's direct prerequisite is B, NOT A (no *1..N here)
        assert result[unrelated.id] == []
    finally:
        _delete_test_entities(a.id, b.id, c.id, unrelated.id)


def test_get_subgraph_returns_only_edges_between_given_concepts(graph_repo) -> None:
    a = Entity(canonical_name=f"A-{uuid4()}", entity_type=EntityType.CONCEPT)
    b = Entity(canonical_name=f"B-{uuid4()}", entity_type=EntityType.CONCEPT)
    outside = Entity(canonical_name=f"Outside-{uuid4()}", entity_type=EntityType.CONCEPT)
    predicate = "is_related_to"

    service = GraphStorageService(graph_repo)
    try:
        # A -> B (both in scope) and B -> outside (outside is NOT in the
        # requested concept_ids) — the subgraph must include the first edge
        # and exclude the second, even though B is an endpoint of both.
        service.store([a, b, outside], [_triple(a.id, predicate, b.id), _triple(b.id, predicate, outside.id)])

        nodes, edges = graph_repo.get_subgraph([a.id, b.id])

        node_ids = {n["id"] for n in nodes}
        assert node_ids == {str(a.id), str(b.id)}
        assert len(edges) == 1
        assert edges[0]["source"] == str(a.id)
        assert edges[0]["target"] == str(b.id)
    finally:
        _delete_test_entities(a.id, b.id, outside.id)


def test_get_subgraph_returns_empty_for_no_concept_ids(graph_repo) -> None:
    assert graph_repo.get_subgraph([]) == ([], [])
