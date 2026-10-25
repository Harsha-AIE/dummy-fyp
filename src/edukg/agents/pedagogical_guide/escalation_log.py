from __future__ import annotations

import json
from pathlib import Path

from edukg.core.config import get_settings
from edukg.core.exceptions import CacheError
from edukg.core.logging_setup import get_logger
from edukg.core.models import DialogueTurn

logger = get_logger(__name__)


class EscalationLogStore:
    """Filesystem trail of every tier-1 -> tier-2 escalation, mirroring the
    pending-review filesystem convention used elsewhere (e.g.
    modules/fusion/pending_triple_store.py). No trainer consumes this yet —
    it exists so "escalation rate over time" is a real, computable metric
    once one does, rather than an unimplementable claim."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or get_settings().paths.data_dir / "agent_logs" / "escalations"
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, turn: DialogueTurn) -> None:
        path = self._base_dir / f"{turn.id}.json"
        try:
            path.write_text(json.dumps(turn.model_dump(mode="json"), indent=2), encoding="utf-8")
        except OSError as exc:
            raise CacheError(f"Failed saving escalation log {turn.id}") from exc
        logger.info("escalation_logged", turn_id=str(turn.id), intent=turn.intent.value)

    def list_all(self) -> list[DialogueTurn]:
        return [
            DialogueTurn.model_validate(json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(self._base_dir.glob("*.json"))
        ]
