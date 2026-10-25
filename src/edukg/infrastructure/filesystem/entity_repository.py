from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import UUID

from edukg.core.config import get_settings
from edukg.core.exceptions import CacheError
from edukg.core.interfaces import EntityRepository
from edukg.core.logging_setup import get_logger
from edukg.core.models import Entity

logger = get_logger(__name__)

# Entity names come from extracted text (KeyBERT/YAKE keyphrases can span a
# paragraph break and include a literal newline), so unlike the ontology's
# fixed, hand-authored names, these must be defensively sanitized for every
# character Windows rejects in a filename, not just the few punctuation marks
# a "safe path" helper elsewhere assumes are the only risk.
_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]+')


class FilesystemEntityRepository(EntityRepository):
    """Persists canonical Entity records as JSON under data/processed/entities.
    One file per entity: <entity_id>.json. A name index (<normalized_name>.id)
    mirrors FilesystemDocumentRepository's hash-index pattern, enabling fast
    find_by_canonical_name lookups for entity linking's exact-match check."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or get_settings().paths.processed_dir / "entities"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._name_index_dir = self._base_dir / "_name_index"
        self._name_index_dir.mkdir(parents=True, exist_ok=True)

    def _entity_path(self, entity_id: UUID) -> Path:
        return self._base_dir / f"{entity_id}.json"

    def _name_key(self, name: str) -> str:
        normalized = " ".join(name.split()).lower()  # collapses embedded newlines/tabs too
        return _UNSAFE_FILENAME_CHARS.sub("_", normalized) or "_"

    def save(self, entity: Entity) -> Entity:
        try:
            self._entity_path(entity.id).write_text(
                json.dumps(entity.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            (self._name_index_dir / f"{self._name_key(entity.canonical_name)}.id").write_text(
                str(entity.id), encoding="utf-8"
            )
            for alias in entity.aliases:
                (self._name_index_dir / f"{self._name_key(alias)}.id").write_text(str(entity.id), encoding="utf-8")
        except OSError as exc:
            raise CacheError(f"Failed saving entity {entity.id}") from exc
        logger.info("entity_saved", entity_id=str(entity.id), canonical_name=entity.canonical_name)
        return entity

    def get(self, entity_id: UUID) -> Entity | None:
        path = self._entity_path(entity_id)
        if not path.exists():
            return None
        return Entity.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def find_by_canonical_name(self, name: str) -> Entity | None:
        index_path = self._name_index_dir / f"{self._name_key(name)}.id"
        if not index_path.exists():
            return None
        entity_id = UUID(index_path.read_text(encoding="utf-8").strip())
        return self.get(entity_id)

    def list_all(self) -> list[Entity]:
        entities: list[Entity] = []
        for path in sorted(self._base_dir.glob("*.json")):
            entities.append(Entity.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        return entities
