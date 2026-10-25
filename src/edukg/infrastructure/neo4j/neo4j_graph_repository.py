from __future__ import annotations

from uuid import UUID

from edukg.core.config import Neo4jSettings, get_settings
from edukg.core.exceptions import ExternalServiceError, GraphStorageError
from edukg.core.interfaces import GraphRepository
from edukg.core.logging_setup import get_logger
from edukg.core.models import Entity, Triple

logger = get_logger(__name__)

_UPSERT_ENTITY_CYPHER = """
MERGE (e:Entity {id: $id})
SET e.canonical_name = $canonical_name,
    e.entity_type = $entity_type,
    e.aliases = $aliases,
    e.description = $description
"""

# Relationship type is fixed to RELATES_TO with the ontology predicate stored as
# a property (rather than using the predicate as the Cypher relationship type)
# so predicates discovered at runtime never require dynamic type interpolation
# — keeps every Cypher statement fully parameterized, no string-built queries.
_UPSERT_TRIPLE_CYPHER = """
MATCH (s:Entity {id: $subject_id})
MATCH (o:Entity {id: $object_id})
MERGE (s)-[r:RELATES_TO {predicate: $predicate}]->(o)
ON CREATE SET r.evidence_count = 1, r.confidence = $confidence, r.triple_id = $triple_id
ON MATCH SET r.evidence_count = r.evidence_count + 1,
             r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END
"""

_CONSTRAINT_CYPHER = "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"

# max_hops is interpolated directly (not as a bound parameter) because Cypher
# variable-length path bounds must be literal integers, not query parameters;
# it is always an int passed from our own code, never user-supplied text, so
# this is not a Cypher-injection risk the way string interpolation would be.
_SHORTEST_PATH_CYPHER_TEMPLATE = """
MATCH (a:Entity {{id: $from_id}}), (b:Entity {{id: $to_id}})
MATCH p = shortestPath((a)-[*..{max_hops}]-(b))
RETURN length(p) AS hops
"""

# Computes every (from_id, to_id) shortest-path distance in ONE round trip
# instead of one round trip per pair — see shortest_path_lengths_bulk's
# docstring for why this exists: the re-ranker was calling
# shortest_path_length() once per (query_concept, chunk_concept) pair per
# candidate chunk, which is O(candidate_pool * concepts^2) sequential
# network round trips to Neo4j for a single RAG query (measured: tens of
# thousands of queries logged, several seconds of added latency per
# request). Same-id pairs (distance 0) are filtered out beforehand in
# Python, not relied on here, matching shortest_path_length's own explicit
# early-return for that case.
_BULK_SHORTEST_PATH_CYPHER_TEMPLATE = """
UNWIND $from_ids AS from_id
MATCH (a:Entity {{id: from_id}})
MATCH (b:Entity) WHERE b.id IN $to_ids AND b.id <> from_id
MATCH p = shortestPath((a)-[*..{max_hops}]-(b))
RETURN from_id, b.id AS to_id, length(p) AS hops
"""

# Finds every (a, b) pair connected by a chain of `predicate`-typed edges up
# to max_hops long, where no *direct* edge for that predicate already exists
# between them — these are the pairs transitive closure should add. `ALL(rel
# IN r WHERE ...)` checks every relationship in the matched path shares the
# same predicate (a plain *1..N pattern alone would match mixed-predicate
# paths too, which isn't what "transitive over this one relation" means).
_FIND_TRANSITIVE_PAIRS_CYPHER_TEMPLATE = """
MATCH p = (a:Entity)-[r:RELATES_TO*1..{max_hops}]->(b:Entity)
WHERE a.id <> b.id
  AND ALL(rel IN r WHERE rel.predicate = $predicate)
  AND NOT (a)-[:RELATES_TO {{predicate: $predicate}}]->(b)
RETURN DISTINCT a.id AS from_id, b.id AS to_id
"""

_MERGE_INFERRED_RELATION_CYPHER = """
MATCH (a:Entity {id: $from_id})
MATCH (b:Entity {id: $to_id})
MERGE (a)-[r:RELATES_TO {predicate: $predicate}]->(b)
ON CREATE SET r.evidence_count = 0, r.confidence = $confidence, r.inferred = true
"""

# For each of $concept_ids, finds every concept with a direct
# `predicate`-typed edge INTO it — i.e. what must be understood before it.
# Direct edges only (no *1..N here, unlike the transitive-pairs query above)
# — TopicRecommender walks one prerequisite hop at a time itself, since it
# needs to know about each intermediate concept's own mastery, not just
# whether a path exists.
_PREREQUISITES_BULK_CYPHER = """
UNWIND $concept_ids AS cid
MATCH (prereq:Entity)-[r:RELATES_TO {predicate: $predicate}]->(target:Entity {id: cid})
RETURN cid AS concept_id, prereq.id AS prereq_id
"""

# Mind Map (Tier 4): edges *among* a given set of concepts only — unlike
# export_graph's full scan, this deliberately excludes any edge reaching
# outside $concept_ids, since the mind map is scoped to one document's own
# top concepts, not the whole corpus's graph.
_SUBGRAPH_NODES_CYPHER = """
MATCH (e:Entity) WHERE e.id IN $concept_ids
RETURN e
"""

_SUBGRAPH_EDGES_CYPHER = """
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE a.id IN $concept_ids AND b.id IN $concept_ids
RETURN a.id AS a_id, b.id AS b_id, r
"""


class Neo4jGraphRepository(GraphRepository):
    """
    Sole point of contact with Neo4j (stage 12, Graph Storage). Entities are
    merged by id; triples are merged as a RELATES_TO relationship keyed by
    (subject, predicate, object) so repeated evidence increments a counter
    rather than creating duplicate edges — mirrors the additive-confidence /
    evidence-count description in the pipeline spec's Phase 5.
    """

    def __init__(self, settings: Neo4jSettings | None = None) -> None:
        self._settings = settings or get_settings().neo4j
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise ExternalServiceError("The 'neo4j' package is not installed") from exc
            self._driver = GraphDatabase.driver(
                self._settings.uri,
                auth=(self._settings.user, self._settings.password),
                max_connection_pool_size=self._settings.max_connection_pool_size,
            )
        return self._driver

    def ensure_constraints(self) -> None:
        driver = self._get_driver()
        try:
            with driver.session(database=self._settings.database) as session:
                session.run(_CONSTRAINT_CYPHER)
        except Exception as exc:  # noqa: BLE001
            raise GraphStorageError(f"Failed ensuring Neo4j constraints: {exc}") from exc
        logger.info("neo4j_constraints_ensured")

    def upsert_entity(self, entity: Entity) -> None:
        driver = self._get_driver()
        try:
            with driver.session(database=self._settings.database) as session:
                session.run(
                    _UPSERT_ENTITY_CYPHER,
                    id=str(entity.id),
                    canonical_name=entity.canonical_name,
                    entity_type=entity.entity_type.value,
                    aliases=entity.aliases,
                    description=entity.description,
                )
        except Exception as exc:  # noqa: BLE001
            raise GraphStorageError(f"Failed upserting entity {entity.id}: {exc}") from exc

    def upsert_triple(self, triple: Triple, subject: Entity, obj: Entity) -> None:
        driver = self._get_driver()
        try:
            with driver.session(database=self._settings.database) as session:
                session.run(
                    _UPSERT_TRIPLE_CYPHER,
                    subject_id=str(subject.id),
                    object_id=str(obj.id),
                    predicate=triple.predicate,
                    confidence=triple.confidence.value,
                    triple_id=str(triple.id),
                )
        except Exception as exc:  # noqa: BLE001
            raise GraphStorageError(f"Failed upserting triple {triple.id}: {exc}") from exc

    def batch_upsert_triples(self, triples: list[Triple], entities_by_id: dict[UUID, Entity]) -> None:
        driver = self._get_driver()
        try:
            with driver.session(database=self._settings.database) as session:
                for triple in triples:
                    subject = entities_by_id.get(triple.subject_entity_id)
                    obj = entities_by_id.get(triple.object_entity_id)
                    if subject is None or obj is None:
                        logger.warning("batch_upsert_skipped_missing_entity", triple_id=str(triple.id))
                        continue
                    session.run(
                        _UPSERT_TRIPLE_CYPHER,
                        subject_id=str(subject.id),
                        object_id=str(obj.id),
                        predicate=triple.predicate,
                        confidence=triple.confidence.value,
                        triple_id=str(triple.id),
                    )
        except Exception as exc:  # noqa: BLE001
            raise GraphStorageError(f"Failed batch upserting {len(triples)} triples: {exc}") from exc
        logger.info("neo4j_batch_upsert_done", triple_count=len(triples))

    def export_graph(self) -> tuple[list[dict], list[dict]]:
        """Full node/edge scan for the API layer's graph visualizer — not part
        of the GraphRepository ABC (it's a UI-serving concern, not a pipeline-
        stage one), so it lives only on this concrete class. No pagination at
        current scale; add it here first if the graph grows large."""
        driver = self._get_driver()
        try:
            with driver.session(database=self._settings.database) as session:
                nodes = [
                    {
                        "id": record["e"]["id"],
                        "name": record["e"]["canonical_name"],
                        "type": record["e"]["entity_type"],
                    }
                    for record in session.run("MATCH (e:Entity) RETURN e")
                ]
                edges = [
                    {
                        "source": record["a_id"],
                        "target": record["b_id"],
                        "predicate": record["r"]["predicate"],
                        "confidence": record["r"]["confidence"],
                        "evidence_count": record["r"]["evidence_count"],
                        # Edges from materialize_transitive_closure() set this
                        # true; directly-extracted edges never have the
                        # property at all, so .get(..., False) is the correct
                        # default rather than a hardcoded False — lets the
                        # visualizer distinguish "the source text said this"
                        # from "the graph inferred this transitively".
                        "inferred": record["r"].get("inferred", False),
                    }
                    for record in session.run(
                        "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) RETURN a.id AS a_id, b.id AS b_id, r"
                    )
                ]
        except Exception as exc:  # noqa: BLE001
            raise GraphStorageError(f"Failed exporting graph: {exc}") from exc
        return nodes, edges

    def shortest_path_length(self, from_entity_id: UUID, to_entity_id: UUID, max_hops: int) -> int | None:
        if from_entity_id == to_entity_id:
            return 0
        if not isinstance(max_hops, int) or max_hops < 1:
            raise GraphStorageError(f"max_hops must be a positive int, got {max_hops!r}")

        driver = self._get_driver()
        try:
            with driver.session(database=self._settings.database) as session:
                record = session.run(
                    _SHORTEST_PATH_CYPHER_TEMPLATE.format(max_hops=max_hops),
                    from_id=str(from_entity_id),
                    to_id=str(to_entity_id),
                ).single()
        except Exception as exc:  # noqa: BLE001
            raise GraphStorageError(
                f"Failed computing shortest path {from_entity_id} -> {to_entity_id}: {exc}"
            ) from exc
        return record["hops"] if record is not None else None

    def shortest_path_lengths_bulk(
        self, from_ids: list[UUID], to_ids: list[UUID], max_hops: int
    ) -> dict[tuple[UUID, UUID], int]:
        """Bulk form of shortest_path_length — every (from_id, to_id) pair's
        hop distance in one Neo4j round trip. Not part of the GraphRepository
        ABC (same reasoning as export_graph(): this is a re-ranking
        performance concern for the API/agent layer, not a pipeline-stage
        contract). Same-id pairs are resolved to 0 directly in Python and
        excluded from the query."""
        if not isinstance(max_hops, int) or max_hops < 1:
            raise GraphStorageError(f"max_hops must be a positive int, got {max_hops!r}")

        result: dict[tuple[UUID, UUID], int] = {}
        for shared_id in set(from_ids) & set(to_ids):
            result[(shared_id, shared_id)] = 0

        if not from_ids or not to_ids:
            return result

        driver = self._get_driver()
        try:
            with driver.session(database=self._settings.database) as session:
                records = session.run(
                    _BULK_SHORTEST_PATH_CYPHER_TEMPLATE.format(max_hops=max_hops),
                    from_ids=[str(i) for i in from_ids],
                    to_ids=[str(i) for i in to_ids],
                )
                for record in records:
                    key = (UUID(record["from_id"]), UUID(record["to_id"]))
                    hops = record["hops"]
                    if key not in result or hops < result[key]:
                        result[key] = hops
        except Exception as exc:  # noqa: BLE001
            raise GraphStorageError(f"Failed computing bulk shortest paths: {exc}") from exc
        return result

    def materialize_transitive_closure(self, predicate: str, max_hops: int = 6) -> int:
        """
        For a relation the active ontology marks `is_transitive=True` (e.g.
        is_prerequisite_of: A->B, B->C implies A->C), finds every pair
        connected by a chain of that predicate up to max_hops long with no
        existing direct edge, and MERGEs one in — distinguished from a
        directly-extracted edge via `inferred: true` so provenance (what the
        source text actually said vs. what the graph derived) is never
        conflated; `evidence_count: 0` for the same reason, so it's never
        counted as independently-corroborated evidence if more documents are
        ingested later. Called once per transitive relation after a pipeline
        run's Graph Storage stage completes (see pipeline_runner.py) — this
        is Stage 12's "transitive reasoning" the pipeline spec calls for
        (Cypher-based, multi-hop prerequisite chaining), previously never
        implemented. Returns the number of edges newly inferred.
        """
        if not isinstance(max_hops, int) or max_hops < 1:
            raise GraphStorageError(f"max_hops must be a positive int, got {max_hops!r}")

        driver = self._get_driver()
        try:
            with driver.session(database=self._settings.database) as session:
                pairs = list(
                    session.run(_FIND_TRANSITIVE_PAIRS_CYPHER_TEMPLATE.format(max_hops=max_hops), predicate=predicate)
                )
                for pair in pairs:
                    session.run(
                        _MERGE_INFERRED_RELATION_CYPHER,
                        from_id=pair["from_id"],
                        to_id=pair["to_id"],
                        predicate=predicate,
                        confidence=1.0,
                    )
        except Exception as exc:  # noqa: BLE001
            raise GraphStorageError(f"Failed materializing transitive closure for '{predicate}': {exc}") from exc
        logger.info("transitive_closure_materialized", predicate=predicate, inferred_edge_count=len(pairs))
        return len(pairs)

    def get_prerequisites_bulk(
        self, concept_ids: list[UUID], predicate: str = "is_prerequisite_of"
    ) -> dict[UUID, list[UUID]]:
        """For each given concept, the ids of concepts with a direct
        predicate-typed edge into it (what must be understood before it) —
        one bulk round trip, same UNWIND pattern as
        shortest_path_lengths_bulk. Not part of the GraphRepository ABC for
        the same reason that method isn't: an agent-layer read concern
        (TopicRecommender), not a pipeline-stage contract. A concept with no
        prerequisites simply doesn't appear as a key in the result."""
        result: dict[UUID, list[UUID]] = {cid: [] for cid in concept_ids}
        if not concept_ids:
            return result

        driver = self._get_driver()
        try:
            with driver.session(database=self._settings.database) as session:
                records = session.run(
                    _PREREQUISITES_BULK_CYPHER,
                    concept_ids=[str(i) for i in concept_ids],
                    predicate=predicate,
                )
                for record in records:
                    result[UUID(record["concept_id"])].append(UUID(record["prereq_id"]))
        except Exception as exc:  # noqa: BLE001
            raise GraphStorageError(f"Failed fetching bulk prerequisites: {exc}") from exc
        return result

    def get_subgraph(self, concept_ids: list[UUID]) -> tuple[list[dict], list[dict]]:
        """Mind Map (Tier 4): the same node/edge dict shape export_graph()
        returns, scoped to only the given concepts and the edges directly
        between them — not part of the GraphRepository ABC for the same
        reason export_graph() isn't (a UI-serving concern, not a pipeline-
        stage one)."""
        if not concept_ids:
            return [], []

        driver = self._get_driver()
        try:
            with driver.session(database=self._settings.database) as session:
                ids = [str(i) for i in concept_ids]
                nodes = [
                    {"id": record["e"]["id"], "name": record["e"]["canonical_name"], "type": record["e"]["entity_type"]}
                    for record in session.run(_SUBGRAPH_NODES_CYPHER, concept_ids=ids)
                ]
                edges = [
                    {
                        "source": record["a_id"],
                        "target": record["b_id"],
                        "predicate": record["r"]["predicate"],
                        "confidence": record["r"]["confidence"],
                        "evidence_count": record["r"]["evidence_count"],
                        "inferred": record["r"].get("inferred", False),
                    }
                    for record in session.run(_SUBGRAPH_EDGES_CYPHER, concept_ids=ids)
                ]
        except Exception as exc:  # noqa: BLE001
            raise GraphStorageError(f"Failed fetching subgraph: {exc}") from exc
        return nodes, edges

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
