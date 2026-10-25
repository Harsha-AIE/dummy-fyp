from __future__ import annotations

from pydantic import Field

from edukg.core.models.common import Timestamped
from edukg.core.models.enums import ProfileRole


class Profile(Timestamped):
    """A browser identity's self-selected account type and display name —
    matches the `profiles` table. Keyed by student_id (the same opaque id
    Knowledge Tracing already uses), not a standalone UUID identity, since
    it's an upserted current-state row, not an event log — same reasoning
    MasteryEstimate documents for itself."""

    student_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    role: ProfileRole
