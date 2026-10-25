from __future__ import annotations

from edukg.core.interfaces import GraphRepository
from edukg.core.logging_setup import get_logger
from edukg.core.models import Entity, Triple

logger = get_logger(__name__)


class GraphStorageService:
    """Stage 12: persists accepted entities and triples into the graph via
    GraphRepository (Neo4j in production). Entities are upserted before
    triples since MERGE-ing a relationship requires both endpoint nodes to
    already exist."""

    def __init__(self, graph_repository: GraphRepository) -> None:
        self._graph = graph_repository

    def store(self, entities: list[Entity], triples: list[Triple]) -> None:
        self._graph.ensure_constraints()
        for entity in entities:
            self._graph.upsert_entity(entity)

        entities_by_id = {entity.id: entity for entity in entities}
        self._graph.batch_upsert_triples(triples, entities_by_id)

        logger.info("graph_storage_done", entity_count=len(entities), triple_count=len(triples))
