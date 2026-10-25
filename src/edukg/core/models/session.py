from __future__ import annotations

from uuid import UUID

from pydantic import Field

from edukg.core.models.common import Identifiable, Timestamped
from edukg.core.models.enums import IntentCategory


class DialogueTurn(Identifiable, Timestamped):
    """A single student message processed by the Pedagogical Guide Agent —
    matches the `dialogue_turns` table in the project's data architecture
    notes. `student_id`/`session_id` stay optional: no TutorSession/
    orchestrator layer exists yet to supply them."""

    student_id: str | None = Field(default=None)
    session_id: UUID | None = Field(default=None)
    message: str = Field(min_length=1)
    intent: IntentCategory
    confidence: float = Field(ge=0.0, le=1.0)
    escalated_to_llm: bool = Field(default=False)
    detected_concepts: list[UUID] = Field(default_factory=list)
