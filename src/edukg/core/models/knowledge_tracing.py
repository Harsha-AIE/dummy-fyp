from __future__ import annotations

from uuid import UUID

from pydantic import Field

from edukg.core.models.common import Identifiable, Timestamped
from edukg.core.models.enums import KTEventSource


class KTEvent(Identifiable, Timestamped):
    """A single correctness observation for one student/concept pair —
    matches the `kt_events` table in the project's data architecture notes.
    Currently only `source=dialogue` events are produced, derived from the
    Pedagogical Guide's classified intent (see
    `agents/knowledge_tracing/knowledge_tracing_agent.py` for the exact
    heuristic) — `source=worksheet` is reserved for the (unbuilt) grading
    feature and never emitted today."""

    student_id: str = Field(min_length=1)
    concept_id: UUID
    source: KTEventSource
    correct: bool
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class MasteryEstimate(Timestamped):
    """Knowledge Tracing's output for one student/concept pair — the fused
    BKT+DKT mastery vector entry, matching the `mastery_estimates` table.
    Keyed by (student_id, concept_id), not a standalone UUID identity, since
    it's an upserted current-state row, not an event log."""

    student_id: str = Field(min_length=1)
    concept_id: UUID
    bkt_mastery: float = Field(ge=0.0, le=1.0)
    dkt_mastery: float | None = Field(default=None, ge=0.0, le=1.0)
    ensemble_mastery: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)
    top_contributing_event_ids: list[UUID] = Field(default_factory=list)
