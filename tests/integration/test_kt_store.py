"""Integration test against the real Neon Postgres instance (DATABASE_URL in
.env). Skipped automatically if unreachable — same convention as
test_postgres_document_store.py / test_graph_storage.py."""

from uuid import uuid4

import pytest

from edukg.core.config import get_settings
from edukg.core.models.enums import KTEventSource
from edukg.core.models.knowledge_tracing import MasteryEstimate
from edukg.infrastructure.postgres.kt_store import KnowledgeTracingStore


@pytest.fixture
def store():
    s = KnowledgeTracingStore()
    try:
        s.ensure_schema()
    except Exception as exc:  # noqa: BLE001 - any connectivity failure skips this suite
        pytest.skip(f"Postgres not reachable, skipping integration test: {exc}")
    return s


def test_record_and_list_events_round_trip(store: KnowledgeTracingStore) -> None:
    student_id = f"test-student-{uuid4()}"
    concept_id = uuid4()

    try:
        store.record_event(student_id, concept_id, KTEventSource.DIALOGUE, correct=False, weight=1.0)
        store.record_event(student_id, concept_id, KTEventSource.DIALOGUE, correct=True, weight=0.3)

        events = store.list_events(student_id, concept_id)
        assert len(events) == 2
        assert [e.correct for e in events] == [False, True]  # chronological order
    finally:
        _cleanup(student_id)


def test_upsert_mastery_is_keyed_by_student_and_concept(store: KnowledgeTracingStore) -> None:
    student_id = f"test-student-{uuid4()}"
    concept_id = uuid4()

    try:
        estimate = MasteryEstimate(
            student_id=student_id,
            concept_id=concept_id,
            bkt_mastery=0.4,
            dkt_mastery=None,
            ensemble_mastery=0.4,
            explanation="initial",
        )
        store.upsert_mastery(estimate)

        updated = MasteryEstimate(
            student_id=student_id,
            concept_id=concept_id,
            bkt_mastery=0.7,
            dkt_mastery=0.6,
            ensemble_mastery=0.65,
            explanation="updated",
        )
        store.upsert_mastery(updated)

        results = store.get_mastery(student_id)
        assert len(results) == 1  # upsert, not a second row
        assert results[0].bkt_mastery == 0.7
        assert results[0].explanation == "updated"
    finally:
        _cleanup(student_id)


def _cleanup(student_id: str) -> None:
    # Standalone connection, deliberately not reusing the store under test,
    # so teardown doesn't depend on the very code the test exists to exercise.
    import psycopg2

    conn = psycopg2.connect(get_settings().postgres.dsn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute("DELETE FROM kt_events WHERE student_id = %s", (student_id,))
            cur.execute("DELETE FROM mastery_estimates WHERE student_id = %s", (student_id,))
    finally:
        conn.close()
