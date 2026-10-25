from __future__ import annotations

from uuid import UUID

from edukg.core.config import PostgresSettings, get_settings
from edukg.core.exceptions import ExternalServiceError
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import KTEventSource
from edukg.core.models.knowledge_tracing import KTEvent, MasteryEstimate

logger = get_logger(__name__)

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS kt_events (
    id UUID PRIMARY KEY,
    student_id TEXT NOT NULL,
    concept_id UUID NOT NULL,
    source TEXT NOT NULL,
    correct BOOLEAN NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_kt_events_student_concept
    ON kt_events (student_id, concept_id, occurred_at);

CREATE TABLE IF NOT EXISTS mastery_estimates (
    student_id TEXT NOT NULL,
    concept_id UUID NOT NULL,
    bkt_mastery REAL NOT NULL,
    dkt_mastery REAL,
    ensemble_mastery REAL NOT NULL,
    explanation TEXT NOT NULL,
    top_contributing_events JSONB NOT NULL DEFAULT '[]',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (student_id, concept_id)
);
"""

_INSERT_EVENT_SQL = """
INSERT INTO kt_events (id, student_id, concept_id, source, correct, weight, occurred_at)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

_UPSERT_MASTERY_SQL = """
INSERT INTO mastery_estimates
    (student_id, concept_id, bkt_mastery, dkt_mastery, ensemble_mastery, explanation, top_contributing_events, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (student_id, concept_id) DO UPDATE
SET bkt_mastery = EXCLUDED.bkt_mastery,
    dkt_mastery = EXCLUDED.dkt_mastery,
    ensemble_mastery = EXCLUDED.ensemble_mastery,
    explanation = EXCLUDED.explanation,
    top_contributing_events = EXCLUDED.top_contributing_events,
    updated_at = now()
"""

_EVENT_COLUMNS = "id, student_id, concept_id, source, correct, weight, occurred_at"
_MASTERY_COLUMNS = (
    "student_id, concept_id, bkt_mastery, dkt_mastery, ensemble_mastery, explanation, top_contributing_events, updated_at"
)


class KnowledgeTracingStore:
    """Durable store for Knowledge Tracing's inputs (`kt_events`) and outputs
    (`mastery_estimates`), backed by the same Neon Postgres instance as
    PostgresDocumentStore — same conventions deliberately reused: one
    connection per call (no pooling), ensure_schema() run once at startup,
    ExternalServiceError on any failure.

    Only `source=dialogue` events exist today — `source=worksheet` is
    reserved for the (unbuilt) worksheet-grading feature, per the original
    architecture's data model, and is never written by any current caller.
    """

    def __init__(self, settings: PostgresSettings | None = None) -> None:
        self._settings = settings or get_settings().postgres

    def _connect(self):
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError as exc:
            raise ExternalServiceError("The 'psycopg2' package is not installed") from exc
        if not self._settings.dsn:
            raise ExternalServiceError("DATABASE_URL is not configured")
        try:
            conn = psycopg2.connect(self._settings.dsn, connect_timeout=self._settings.connect_timeout_seconds)
            # Windows' default client encoding is locale-dependent (often
            # cp1252, not UTF-8) — without this, non-ASCII TEXT (explanation
            # strings containing an em dash, etc.) gets mangled on the way
            # in. PostgresDocumentStore never hit this because it only
            # stores BYTEA, not TEXT.
            conn.set_client_encoding("UTF8")
            psycopg2.extras.register_uuid(conn_or_curs=conn)
            return conn
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed connecting to Postgres: {exc}") from exc

    def ensure_schema(self) -> None:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(_CREATE_TABLES_SQL)
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed ensuring KT schema: {exc}") from exc
        finally:
            conn.close()
        logger.info("kt_schema_ensured")

    def record_event(
        self, student_id: str, concept_id: UUID, source: KTEventSource, correct: bool, weight: float = 1.0
    ) -> KTEvent:
        event = KTEvent(student_id=student_id, concept_id=concept_id, source=source, correct=correct, weight=weight)
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    _INSERT_EVENT_SQL,
                    (event.id, student_id, concept_id, source.value, correct, weight, event.created_at),
                )
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed recording KT event: {exc}") from exc
        finally:
            conn.close()
        logger.info(
            "kt_event_recorded", student_id=student_id, concept_id=str(concept_id), source=source.value, correct=correct
        )
        return event

    def list_events(self, student_id: str, concept_id: UUID | None = None) -> list[KTEvent]:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                if concept_id is None:
                    cur.execute(
                        f"SELECT {_EVENT_COLUMNS} FROM kt_events WHERE student_id = %s ORDER BY occurred_at ASC",
                        (student_id,),
                    )
                else:
                    cur.execute(
                        f"SELECT {_EVENT_COLUMNS} FROM kt_events WHERE student_id = %s AND concept_id = %s "
                        "ORDER BY occurred_at ASC",
                        (student_id, concept_id),
                    )
                return [_row_to_event(row) for row in cur.fetchall()]
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed listing KT events for '{student_id}': {exc}") from exc
        finally:
            conn.close()

    def list_all_events(self) -> list[KTEvent]:
        """Every event across every student — used only by scripts/train_dkt.py."""
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(f"SELECT {_EVENT_COLUMNS} FROM kt_events ORDER BY occurred_at ASC")
                return [_row_to_event(row) for row in cur.fetchall()]
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed listing all KT events: {exc}") from exc
        finally:
            conn.close()

    def upsert_mastery(self, estimate: MasteryEstimate) -> None:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                import psycopg2.extras

                cur.execute(
                    _UPSERT_MASTERY_SQL,
                    (
                        estimate.student_id,
                        estimate.concept_id,
                        estimate.bkt_mastery,
                        estimate.dkt_mastery,
                        estimate.ensemble_mastery,
                        estimate.explanation,
                        psycopg2.extras.Json([str(eid) for eid in estimate.top_contributing_event_ids]),
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed upserting mastery estimate: {exc}") from exc
        finally:
            conn.close()
        logger.info(
            "kt_mastery_upserted",
            student_id=estimate.student_id,
            concept_id=str(estimate.concept_id),
            ensemble_mastery=estimate.ensemble_mastery,
        )

    def get_mastery(self, student_id: str) -> list[MasteryEstimate]:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    f"SELECT {_MASTERY_COLUMNS} FROM mastery_estimates WHERE student_id = %s ORDER BY updated_at DESC",
                    (student_id,),
                )
                return [_row_to_mastery(row) for row in cur.fetchall()]
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed reading mastery for '{student_id}': {exc}") from exc
        finally:
            conn.close()

    def list_all_mastery(self) -> list[MasteryEstimate]:
        """Every mastery estimate across every student — used by
        scripts/metrics_report.py for aggregate reporting."""
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(f"SELECT {_MASTERY_COLUMNS} FROM mastery_estimates ORDER BY updated_at DESC")
                return [_row_to_mastery(row) for row in cur.fetchall()]
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed listing all mastery estimates: {exc}") from exc
        finally:
            conn.close()


def _row_to_event(row: tuple) -> KTEvent:
    event_id, student_id, concept_id, source, correct, weight, occurred_at = row
    return KTEvent(
        id=event_id,
        student_id=student_id,
        concept_id=concept_id,
        source=KTEventSource(source),
        correct=correct,
        weight=weight,
        created_at=occurred_at,
        updated_at=occurred_at,
    )


def _row_to_mastery(row: tuple) -> MasteryEstimate:
    student_id, concept_id, bkt_mastery, dkt_mastery, ensemble_mastery, explanation, top_events, updated_at = row
    return MasteryEstimate(
        student_id=student_id,
        concept_id=concept_id,
        bkt_mastery=bkt_mastery,
        dkt_mastery=dkt_mastery,
        ensemble_mastery=ensemble_mastery,
        explanation=explanation,
        top_contributing_event_ids=[UUID(eid) for eid in (top_events or [])],
        created_at=updated_at,
        updated_at=updated_at,
    )
