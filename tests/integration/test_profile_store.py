"""Integration test against the real Neon Postgres instance (DATABASE_URL in
.env). Skipped automatically if unreachable — same convention as
test_kt_store.py / test_postgres_document_store.py / test_graph_storage.py."""

from uuid import uuid4

import pytest

from edukg.core.config import get_settings
from edukg.core.models.enums import ProfileRole
from edukg.infrastructure.postgres.profile_store import ProfileStore


@pytest.fixture
def store():
    s = ProfileStore()
    try:
        s.ensure_schema()
    except Exception as exc:  # noqa: BLE001 - any connectivity failure skips this suite
        pytest.skip(f"Postgres not reachable, skipping integration test: {exc}")
    return s


def test_upsert_profile_is_idempotent_not_a_duplicate_row(store: ProfileStore) -> None:
    student_id = f"test-student-{uuid4()}"

    try:
        store.upsert_profile(student_id, "Harsha", ProfileRole.STUDENT)
        updated = store.upsert_profile(student_id, "Harsha S.", ProfileRole.TEACHER)

        assert updated.display_name == "Harsha S."
        assert updated.role == ProfileRole.TEACHER

        teachers = store.list_by_role(ProfileRole.TEACHER)
        assert any(p.student_id == student_id for p in teachers)
        students = store.list_by_role(ProfileRole.STUDENT)
        assert not any(p.student_id == student_id for p in students)  # role changed, not duplicated
    finally:
        _cleanup(student_id)


def test_list_by_role_only_returns_the_matching_role(store: ProfileStore) -> None:
    student_id = f"test-student-{uuid4()}"
    teacher_id = f"test-teacher-{uuid4()}"

    try:
        store.upsert_profile(student_id, "A Student", ProfileRole.STUDENT)
        store.upsert_profile(teacher_id, "A Teacher", ProfileRole.TEACHER)

        students = store.list_by_role(ProfileRole.STUDENT)
        assert any(p.student_id == student_id for p in students)
        assert not any(p.student_id == teacher_id for p in students)
    finally:
        _cleanup(student_id)
        _cleanup(teacher_id)


def _cleanup(student_id: str) -> None:
    # Standalone connection, deliberately not reusing the store under test,
    # so teardown doesn't depend on the very code the test exists to exercise.
    import psycopg2

    conn = psycopg2.connect(get_settings().postgres.dsn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute("DELETE FROM profiles WHERE student_id = %s", (student_id,))
    finally:
        conn.close()
