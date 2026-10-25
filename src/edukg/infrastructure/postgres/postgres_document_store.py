from __future__ import annotations

from typing import Any

from edukg.core.config import PostgresSettings, get_settings
from edukg.core.exceptions import ExternalServiceError
from edukg.core.logging_setup import get_logger

logger = get_logger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS uploaded_documents (
    id SERIAL PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    content_type TEXT,
    content BYTEA NOT NULL,
    byte_size INTEGER NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

_UPSERT_SQL = """
INSERT INTO uploaded_documents (filename, content_type, content, byte_size)
VALUES (%s, %s, %s, %s)
ON CONFLICT (filename) DO UPDATE
SET content_type = EXCLUDED.content_type,
    content = EXCLUDED.content,
    byte_size = EXCLUDED.byte_size,
    uploaded_at = now()
"""


class PostgresDocumentStore:
    """
    Durable persistence-of-record for uploaded source documents, backed by
    Neon-hosted Postgres (see PostgresSettings). The pipeline still reads
    from the local data/raw copy to actually process a document —
    PDFProcessor/DOCXProcessor/PaddleOCR all expect a real filesystem path,
    and rewriting that stack to be DB-native is out of scope here — so
    Postgres is not in the hot processing path. It exists so an uploaded
    file survives a filesystem wipe and isn't tied to a single server's
    disk, per the original architecture's "PostgreSQL (Ops DB)" role.

    A new connection is opened per call rather than pooled — simplest thing
    that's safe under FastAPI's threaded execution of sync routes, and
    adequate at this upload volume; a real deployment would add pooling.
    """

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
            return psycopg2.connect(self._settings.dsn, connect_timeout=self._settings.connect_timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed connecting to Postgres: {exc}") from exc

    def ensure_schema(self) -> None:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed ensuring Postgres schema: {exc}") from exc
        finally:
            conn.close()
        logger.info("postgres_schema_ensured")

    def save(self, filename: str, content: bytes, content_type: str | None) -> None:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(_UPSERT_SQL, (filename, content_type, content, len(content)))
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed saving '{filename}' to Postgres: {exc}") from exc
        finally:
            conn.close()
        logger.info("postgres_document_saved", filename=filename, byte_size=len(content))

    def get(self, filename: str) -> bytes | None:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute("SELECT content FROM uploaded_documents WHERE filename = %s", (filename,))
                row = cur.fetchone()
                return bytes(row[0]) if row is not None else None
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed reading '{filename}' from Postgres: {exc}") from exc
        finally:
            conn.close()

    def list_all(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT filename, content_type, byte_size, uploaded_at FROM uploaded_documents ORDER BY uploaded_at DESC"
                )
                return [
                    {"filename": r[0], "content_type": r[1], "byte_size": r[2], "uploaded_at": r[3]}
                    for r in cur.fetchall()
                ]
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed listing documents from Postgres: {exc}") from exc
        finally:
            conn.close()
