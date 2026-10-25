"""Integration test against the real Neon Postgres instance (DATABASE_URL in
.env). Skipped automatically if unreachable — not part of the
no-external-services unit suite, same convention as test_graph_storage.py."""

from uuid import uuid4

import pytest

from edukg.core.config import get_settings
from edukg.infrastructure.postgres.postgres_document_store import PostgresDocumentStore


@pytest.fixture
def store():
    s = PostgresDocumentStore()
    try:
        s.ensure_schema()
    except Exception as exc:  # noqa: BLE001 - any connectivity failure skips this suite
        pytest.skip(f"Postgres not reachable, skipping integration test: {exc}")
    return s


def test_save_and_get_round_trip(store: PostgresDocumentStore) -> None:
    filename = f"test-{uuid4()}.pdf"
    content = b"%PDF-1.4 integration test content"

    try:
        store.save(filename, content, "application/pdf")
        assert store.get(filename) == content

        # Re-saving the same filename must upsert, not duplicate.
        store.save(filename, b"updated content", "application/pdf")
        assert store.get(filename) == b"updated content"

        names = {d["filename"] for d in store.list_all()}
        assert filename in names
    finally:
        _cleanup(store, filename)


def test_get_missing_document_returns_none(store: PostgresDocumentStore) -> None:
    assert store.get(f"nonexistent-{uuid4()}.pdf") is None


def _cleanup(store: PostgresDocumentStore, filename: str) -> None:
    # Standalone connection, deliberately not reusing the store under test,
    # so teardown doesn't depend on the very code the test exists to exercise.
    import psycopg2

    conn = psycopg2.connect(get_settings().postgres.dsn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute("DELETE FROM uploaded_documents WHERE filename = %s", (filename,))
    finally:
        conn.close()
