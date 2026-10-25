from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from edukg.core.config import get_settings
from edukg.core.exceptions import CacheError
from edukg.core.models import (
    Chunk,
    Document,
    Entity,
    Mention,
    OntologyClass,
    OntologyExtension,
    OntologyRelation,
    Relation,
    Triple,
    ValidationResult,
)
from edukg.core.logging_setup import get_logger
from edukg.pipeline.context import PipelineContext

logger = get_logger(__name__)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class CheckpointStore:
    """Writes/reads a PipelineContext snapshot to the local filesystem cache directory.
    Enables safe interruption of long-running document processing and, per the
    resource-exhaustion requirement, gives the caller an explicit artifact to
    download at any point in a run rather than losing partial progress."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or get_settings().paths.cache_dir / "checkpoints"
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, document_id: UUID) -> Path:
        return self._base_dir / f"{document_id}.checkpoint.json"

    def save(self, context: PipelineContext) -> Path:
        path = self._path_for(context.document.id)
        payload = {
            "document": context.document.model_dump(mode="json"),
            "chunks": [c.model_dump(mode="json") for c in context.chunks],
            "ontology_classes": [c.model_dump(mode="json") for c in context.ontology_classes],
            "ontology_relations": [r.model_dump(mode="json") for r in context.ontology_relations],
            "proposed_extensions": [e.model_dump(mode="json") for e in context.proposed_extensions],
            "accepted_extensions": [e.model_dump(mode="json") for e in context.accepted_extensions],
            "mentions": [m.model_dump(mode="json") for m in context.mentions],
            "entities": [e.model_dump(mode="json") for e in context.entities],
            "relations": [r.model_dump(mode="json") for r in context.relations],
            "triples": [t.model_dump(mode="json") for t in context.triples],
            "validation_results": [v.model_dump(mode="json") for v in context.validation_results],
            "completed_stages": [s.value for s in context.completed_stages],
            "stage_errors": context.stage_errors,
        }
        try:
            path.write_text(json.dumps(payload, default=_json_default, indent=2), encoding="utf-8")
        except OSError as exc:
            raise CacheError(f"Failed writing checkpoint for document {context.document.id}") from exc
        logger.info("checkpoint_saved", document_id=str(context.document.id), path=str(path))
        return path

    def load(self, document_id: UUID) -> PipelineContext | None:
        path = self._path_for(document_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        from edukg.core.models.enums import PipelineStage

        context = PipelineContext(document=Document.model_validate(raw["document"]))
        context.chunks = [Chunk.model_validate(c) for c in raw["chunks"]]
        context.ontology_classes = [OntologyClass.model_validate(c) for c in raw["ontology_classes"]]
        context.ontology_relations = [OntologyRelation.model_validate(r) for r in raw["ontology_relations"]]
        context.proposed_extensions = [OntologyExtension.model_validate(e) for e in raw["proposed_extensions"]]
        context.accepted_extensions = [OntologyExtension.model_validate(e) for e in raw["accepted_extensions"]]
        context.mentions = [Mention.model_validate(m) for m in raw["mentions"]]
        context.entities = [Entity.model_validate(e) for e in raw["entities"]]
        context.relations = [Relation.model_validate(r) for r in raw["relations"]]
        context.triples = [Triple.model_validate(t) for t in raw["triples"]]
        context.validation_results = [ValidationResult.model_validate(v) for v in raw["validation_results"]]
        context.completed_stages = [PipelineStage(s) for s in raw["completed_stages"]]
        context.stage_errors = raw["stage_errors"]
        logger.info("checkpoint_loaded", document_id=str(document_id), path=str(path))
        return context

    def export_path(self, document_id: UUID) -> Path | None:
        path = self._path_for(document_id)
        return path if path.exists() else None
