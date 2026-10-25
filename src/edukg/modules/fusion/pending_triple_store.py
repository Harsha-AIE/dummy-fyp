from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from edukg.core.config import get_settings
from edukg.core.exceptions import CacheError
from edukg.core.logging_setup import get_logger
from edukg.core.models import Triple

logger = get_logger(__name__)


class PendingTripleStore:
    """Filesystem persistence for triples that Knowledge Fusion routed to
    human review (validation violations or below-threshold confidence) —
    mirrors the extensions/pending/<id>.json convention already used by
    FilesystemOntologyRepository for ontology extensions, one JSON file per
    pending triple under data/processed/pending_triples."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or get_settings().paths.processed_dir / "pending_triples"
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, triple: Triple) -> None:
        path = self._base_dir / f"{triple.id}.json"
        try:
            path.write_text(json.dumps(triple.model_dump(mode="json"), indent=2), encoding="utf-8")
        except OSError as exc:
            raise CacheError(f"Failed saving pending triple {triple.id}") from exc
        logger.info("pending_triple_saved", triple_id=str(triple.id))

    def save_all(self, triples: list[Triple]) -> None:
        for triple in triples:
            self.save(triple)

    def list_pending(self) -> list[Triple]:
        return [Triple.model_validate(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(self._base_dir.glob("*.json"))]

    def get(self, triple_id: UUID) -> Triple | None:
        path = self._base_dir / f"{triple_id}.json"
        if not path.exists():
            return None
        return Triple.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def remove(self, triple_id: UUID) -> None:
        path = self._base_dir / f"{triple_id}.json"
        path.unlink(missing_ok=True)
        logger.info("pending_triple_removed", triple_id=str(triple_id))
