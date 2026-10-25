from __future__ import annotations

from edukg.core.config import PostgresSettings, get_settings
from edukg.core.exceptions import ExternalServiceError
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import ProfileRole
from edukg.core.models.profile import Profile

logger = get_logger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS profiles (
    student_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_UPSERT_PROFILE_SQL = """
INSERT INTO profiles (student_id, display_name, role, created_at, updated_at)
VALUES (%s, %s, %s, now(), now())
ON CONFLICT (student_id) DO UPDATE
SET display_name = EXCLUDED.display_name,
    role = EXCLUDED.role,
    updated_at = now()
RETURNING student_id, display_name, role, created_at, updated_at
"""

_PROFILE_COLUMNS = "student_id, display_name, role, created_at, updated_at"


class ProfileStore:
    """Durable store for the Teacher/Student role picker's identity —
    which browser (keyed by the same opaque student_id Knowledge Tracing
    already uses) picked which role and display name. Same conventions as
    KnowledgeTracingStore deliberately reused: one connection per call (no
    pooling), ensure_schema() run once at startup, ExternalServiceError on
    any failure. Classroom-sized table — no extra index on `role` beyond
    the primary key."""

    def __init__(self, settings: PostgresSettings | None = None) -> None:
        self._settings = settings or get_settings().postgres

    def _connect(self):
        try:
            import psycopg2
        except ImportError as exc:
            raise ExternalServiceError("The 'psycopg2' package is not installed") from exc
        if not self._settings.dsn:
            raise ExternalServiceError("DATABASE_URL is not configured")
        try:
            conn = psycopg2.connect(self._settings.dsn, connect_timeout=self._settings.connect_timeout_seconds)
            conn.set_client_encoding("UTF8")
            return conn
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed connecting to Postgres: {exc}") from exc

    def ensure_schema(self) -> None:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed ensuring profile schema: {exc}") from exc
        finally:
            conn.close()
        logger.info("profile_schema_ensured")

    def upsert_profile(self, student_id: str, display_name: str, role: ProfileRole) -> Profile:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(_UPSERT_PROFILE_SQL, (student_id, display_name, role.value))
                row = cur.fetchone()
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed upserting profile for '{student_id}': {exc}") from exc
        finally:
            conn.close()
        logger.info("profile_upserted", student_id=student_id, role=role.value)
        return _row_to_profile(row)

    def list_by_role(self, role: ProfileRole) -> list[Profile]:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    f"SELECT {_PROFILE_COLUMNS} FROM profiles WHERE role = %s ORDER BY display_name ASC",
                    (role.value,),
                )
                return [_row_to_profile(row) for row in cur.fetchall()]
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed listing profiles for role '{role.value}': {exc}") from exc
        finally:
            conn.close()


def _row_to_profile(row: tuple) -> Profile:
    student_id, display_name, role, created_at, updated_at = row
    return Profile(
        student_id=student_id,
        display_name=display_name,
        role=ProfileRole(role),
        created_at=created_at,
        updated_at=updated_at,
    )
