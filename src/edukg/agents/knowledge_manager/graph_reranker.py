from __future__ import annotations

from uuid import UUID

from edukg.core.logging_setup import get_logger
from edukg.infrastructure.neo4j.neo4j_graph_repository import Neo4jGraphRepository

logger = get_logger(__name__)

_DEFAULT_MAX_HOPS = 4


class GraphDistanceReranker:
    """
    KG-grounded re-ranking term from the pipeline spec's re-ranking formula
    `score = alpha*similarity + beta*(1/graph_distance) + gamma*mastery_gap_weight`.
    No Knowledge Tracing agent exists yet (see MODULE_3_5_NOTES.md-style scope
    notes for this pass), so the mastery_gap_weight term is dropped — this
    class supplies only the graph-distance term, and stays swappable: once a
    mastery vector exists, a caller can add a third scoring term without
    touching this class.
    """

    def __init__(self, graph_repository: Neo4jGraphRepository, max_hops: int = _DEFAULT_MAX_HOPS) -> None:
        # Typed as the concrete Neo4jGraphRepository, not the abstract
        # GraphRepository — this class calls shortest_path_lengths_bulk(),
        # which is deliberately not part of that ABC (see its docstring:
        # same reasoning as export_graph(), an API/agent-layer performance
        # concern, not a pipeline-stage contract). The abstract type
        # annotation was a lie about what's actually required; this says so.
        self._graph = graph_repository
        self._max_hops = max_hops

    def score(self, query_concepts: list[UUID], chunk_concepts: list[UUID]) -> float:
        """1/(1 + shortest_hop_distance) over every (query_concept, chunk_concept)
        pair, taking the closest pair found. 0.0 if either side has no
        concepts, or no pair is within max_hops of each other."""
        if not query_concepts or not chunk_concepts:
            return 0.0

        best_distance: int | None = None
        for query_id in query_concepts:
            for chunk_id in chunk_concepts:
                distance = self._graph.shortest_path_length(query_id, chunk_id, self._max_hops)
                if distance is not None and (best_distance is None or distance < best_distance):
                    best_distance = distance
                    if best_distance == 0:
                        return 1.0  # can't do better than identical concepts

        if best_distance is None:
            return 0.0
        return 1.0 / (1 + best_distance)

    def score_batch(self, query_concepts: list[UUID], candidates_chunk_concepts: list[list[UUID]]) -> list[float]:
        """Same formula as score(), for many candidate chunks against the
        same query_concepts, but computed from ONE bulk graph lookup instead
        of one Neo4j round trip per (query_concept, chunk_concept) pair per
        candidate — score() calling shortest_path_length() from inside
        KnowledgeManagerAgent.retrieve()'s per-candidate loop was issuing
        dozens to hundreds of sequential Neo4j queries for a single RAG
        request (found via a real, measured multi-second latency
        regression). Returns scores in the same order as
        candidates_chunk_concepts."""
        if not query_concepts:
            return [0.0] * len(candidates_chunk_concepts)

        all_chunk_concepts = {cid for concepts in candidates_chunk_concepts for cid in concepts}
        if not all_chunk_concepts:
            return [0.0] * len(candidates_chunk_concepts)

        distances = self._graph.shortest_path_lengths_bulk(query_concepts, list(all_chunk_concepts), self._max_hops)

        scores: list[float] = []
        for chunk_concepts in candidates_chunk_concepts:
            if not chunk_concepts:
                scores.append(0.0)
                continue
            best_distance: int | None = None
            for query_id in query_concepts:
                for chunk_id in chunk_concepts:
                    distance = distances.get((query_id, chunk_id))
                    if distance is not None and (best_distance is None or distance < best_distance):
                        best_distance = distance
            scores.append(0.0 if best_distance is None else 1.0 / (1 + best_distance))
        return scores
