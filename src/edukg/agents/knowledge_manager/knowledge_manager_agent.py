from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from edukg.agents.knowledge_manager.content_indexer import ContentIndex
from edukg.agents.knowledge_manager.graph_reranker import GraphDistanceReranker
from edukg.agents.knowledge_manager.rrf_fusion import reciprocal_rank_fusion
from edukg.agents.shared.concept_tagger import ConceptTagger
from edukg.core.interfaces import EmbeddingProvider
from edukg.core.logging_setup import get_logger
from edukg.core.models import Chunk

logger = get_logger(__name__)

_DEFAULT_CANDIDATE_POOL = 20
# A concept referenced by more than this fraction of all indexed chunks is
# too common across the corpus to be a useful graph-distance re-ranking
# signal — found via a real, live symptom: generic non-domain-specific
# words extracted as "concepts" (e.g. "selected", "first" before the
# extraction-time filters, a bare "1") happened to appear in both a query
# and an entirely unrelated document, giving that unrelated chunk a false
# perfect graph_score=1.0 purely because they share the same overly-common
# concept — not because the chunk was actually relevant. No stopword list
# can fully prevent generic words from ever becoming entities (many are
# genuine domain terms in one document's context and filler in another), so
# this discounts the SYMPTOM directly: a concept's re-ranking signal is
# only trusted when it's distinctive enough to actually mean something.
# _MIN_GENERIC_CHUNK_COUNT keeps tiny corpora from over-triggering (a
# concept in 2 of 5 chunks isn't suspicious the way it would be in 200).
_GENERIC_CONCEPT_MAX_CHUNK_FRACTION = 0.15
_MIN_GENERIC_CHUNK_COUNT = 3

MasteryProvider = Callable[[str], dict[UUID, float]]


def _mastery_gap_score(chunk_concepts: list[UUID], mastery_by_concept: dict[UUID, float]) -> float:
    """Average `1 - mastery` over the chunk's concepts that the student has
    a mastery reading for — higher means the chunk covers more of the
    student's weak spots, so it should rank higher. 0.0 if there's no
    overlap (never penalizes a chunk for concepts we simply have no
    mastery data on)."""
    gaps = [1.0 - mastery_by_concept[cid] for cid in chunk_concepts if cid in mastery_by_concept]
    return sum(gaps) / len(gaps) if gaps else 0.0


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    fused_score: float
    graph_score: float
    final_score: float


class KnowledgeManagerAgent:
    """
    Stage-7-adjacent RAG agent (Phase 3 of the multi-agent architecture):
    hybrid dense+lexical retrieval, fused via Reciprocal Rank Fusion, then
    re-ranked by KG graph-distance from the query's concepts to each
    candidate chunk's concepts, plus an optional mastery-gap term. Implements
    `score = alpha*similarity + beta*(1/graph_distance) + gamma*mastery_gap_weight`
    from the pipeline spec in full — the gamma term is inert (gamma=0.0,
    mastery_provider=None) unless a Knowledge Tracing agent is wired in, so
    behavior is unchanged for any caller that doesn't opt in. RRF's raw
    scores and the reranker's 0-1 graph score are on very different numeric
    scales, so the fused score is min-max normalized against the current
    candidate pool before combining — otherwise the graph term would
    dominate regardless of `alpha`/`beta`.
    """

    def __init__(
        self,
        content_index: ContentIndex,
        embedding_provider: EmbeddingProvider,
        concept_tagger: ConceptTagger,
        reranker: GraphDistanceReranker,
        *,
        candidate_pool: int = _DEFAULT_CANDIDATE_POOL,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.0,
        mastery_provider: MasteryProvider | None = None,
    ) -> None:
        self._index = content_index
        self._embeddings = embedding_provider
        self._concept_tagger = concept_tagger
        self._reranker = reranker
        self._candidate_pool = candidate_pool
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._mastery_provider = mastery_provider

    def retrieve(self, query: str, top_k: int = 5, *, student_id: str | None = None) -> list[RetrievedChunk]:
        if not query.strip() or not self._index.chunks_by_id:
            return []

        query_vector = self._embeddings.embed([query])[0]
        dense_ranking = [chunk_id for chunk_id, _ in self._index.dense_index.search(query_vector, top_k=self._candidate_pool)]
        lexical_ranking = [chunk_id for chunk_id, _ in self._index.lexical_index.search(query, top_k=self._candidate_pool)]

        fused = reciprocal_rank_fusion([dense_ranking, lexical_ranking])
        fused = [(cid, score) for cid, score in fused if cid in self._index.chunks_by_id][: self._candidate_pool]
        if not fused:
            return []

        max_fused_score = max(score for _, score in fused) or 1.0
        query_concepts = [entity.id for entity in self._concept_tagger.tag(query)]
        query_concepts = self._drop_generic_concepts(query_concepts)

        mastery_by_concept: dict[UUID, float] = {}
        if student_id is not None and self._mastery_provider is not None:
            # Best-effort, same reasoning as the KT piggyback in
            # api/routers/agents.py's _classify(): the mastery-gap term is
            # an optional re-ranking boost (gamma defaults to 0.0/inert
            # without it), not a requirement for retrieval to work. Found
            # live: a Postgres hiccup here used to fail the entire /ask
            # request with an unhandled 502 even though dense+lexical
            # search, fusion, and graph re-ranking above had all already
            # succeeded — losing real, usable results over an optional
            # enhancement that simply degrades to "no mastery boost" instead.
            try:
                mastery_by_concept = self._mastery_provider(student_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("mastery_provider_failed", student_id=student_id, error=str(exc))

        candidate_chunk_concepts = [self._index.chunk_concepts.get(chunk_id, []) for chunk_id, _ in fused]
        graph_scores = self._reranker.score_batch(query_concepts, candidate_chunk_concepts)

        results: list[RetrievedChunk] = []
        for (chunk_id, fused_score), chunk_concepts, graph_score in zip(
            fused, candidate_chunk_concepts, graph_scores, strict=True
        ):
            normalized_fused = fused_score / max_fused_score
            final_score = self._alpha * normalized_fused + self._beta * graph_score
            if mastery_by_concept:
                final_score += self._gamma * _mastery_gap_score(chunk_concepts, mastery_by_concept)
            results.append(
                RetrievedChunk(
                    chunk=self._index.chunks_by_id[chunk_id],
                    fused_score=normalized_fused,
                    graph_score=graph_score,
                    final_score=final_score,
                )
            )

        results.sort(key=lambda r: r.final_score, reverse=True)
        logger.info("knowledge_manager_retrieve_done", query_concept_count=len(query_concepts), candidate_count=len(fused))
        return results[:top_k]

    def _drop_generic_concepts(self, concepts: list[UUID]) -> list[UUID]:
        """Excludes concepts referenced by an unusually large fraction of the
        corpus's chunks from graph-distance re-ranking — see
        _GENERIC_CONCEPT_MAX_CHUNK_FRACTION's docstring for why."""
        total_chunks = len(self._index.chunks_by_id)
        if total_chunks == 0 or not concepts:
            return concepts

        chunk_counts: dict[UUID, int] = {}
        for chunk_concept_ids in self._index.chunk_concepts.values():
            for concept_id in set(chunk_concept_ids):
                chunk_counts[concept_id] = chunk_counts.get(concept_id, 0) + 1

        threshold = max(_MIN_GENERIC_CHUNK_COUNT, total_chunks * _GENERIC_CONCEPT_MAX_CHUNK_FRACTION)
        return [c for c in concepts if chunk_counts.get(c, 0) <= threshold]
