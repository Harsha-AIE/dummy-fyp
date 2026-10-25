from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from edukg.agents.knowledge_tracing.knowledge_tracing_agent import KnowledgeTracingAgent
from edukg.core.interfaces import EntityRepository
from edukg.core.logging_setup import get_logger
from edukg.infrastructure.neo4j.neo4j_graph_repository import Neo4jGraphRepository

logger = get_logger(__name__)

_WEAK_THRESHOLD = 0.5


@dataclass(slots=True)
class TopicRecommendation:
    concept_id: UUID
    concept_name: str
    mastery: float
    reason: str  # "foundational" | "ready"
    unblocks: list[str] = field(default_factory=list)


class TopicRecommender:
    """
    Connects two pieces that already exist separately — Knowledge Tracing's
    mastery vector and the KG's prerequisite structure (including the
    transitive closure materialize_transitive_closure() produces) — into the
    one thing a student actually wants: what to study next, and why. Without
    this, a student sees a bare mastery score per concept with no guidance
    on learning order.

    For each weak concept, walks one prerequisite hop back: if a direct
    prerequisite is itself weak or never engaged with, that's the real gap —
    recommend it instead of the concept it blocks, since studying the
    dependent concept first wouldn't stick without the foundation. If every
    prerequisite is already solid (or there are none), the concept is simply
    recommended directly — the foundation is there, it just hasn't been
    learned yet.
    """

    def __init__(
        self, knowledge_tracing_agent: KnowledgeTracingAgent, graph_repo: Neo4jGraphRepository, entity_repo: EntityRepository
    ) -> None:
        self._kt = knowledge_tracing_agent
        self._graph = graph_repo
        self._entities = entity_repo

    def recommend(self, student_id: str, limit: int = 5) -> list[TopicRecommendation]:
        estimates = self._kt.get_mastery_vector(student_id)
        mastery_by_concept: dict[UUID, float] = {e.concept_id: e.ensemble_mastery for e in estimates}
        weak = [e for e in estimates if e.ensemble_mastery < _WEAK_THRESHOLD]
        if not weak:
            return []

        prereqs_by_concept = self._graph.get_prerequisites_bulk([e.concept_id for e in weak])

        # Blocking prerequisite id -> names of the weak concepts it blocks
        # (several weak concepts can share the same unresolved foundation —
        # merge them into one recommendation rather than listing it twice).
        blockers: dict[UUID, list[str]] = {}
        ready_candidates: list = []
        for estimate in weak:
            concept_name = self._concept_name(estimate.concept_id)
            prereq_ids = prereqs_by_concept.get(estimate.concept_id, [])
            unresolved = [pid for pid in prereq_ids if mastery_by_concept.get(pid, 0.0) < _WEAK_THRESHOLD]
            if unresolved:
                weakest_prereq_id = min(unresolved, key=lambda pid: mastery_by_concept.get(pid, 0.0))
                blockers.setdefault(weakest_prereq_id, []).append(concept_name)
            else:
                ready_candidates.append(estimate)

        # A concept that's independently weak AND already surfaced as a
        # blocker for something else (its own prerequisites happen to be
        # fine, but it's still someone else's foundational gap) must not
        # also get a second, redundant "ready" recommendation for itself.
        ready = [e for e in ready_candidates if e.concept_id not in blockers]

        recommendations: list[TopicRecommendation] = []
        for concept_id, unblocks in blockers.items():
            recommendations.append(
                TopicRecommendation(
                    concept_id=concept_id,
                    concept_name=self._concept_name(concept_id),
                    mastery=mastery_by_concept.get(concept_id, 0.0),
                    reason="foundational",
                    unblocks=unblocks,
                )
            )
        for estimate in ready:
            recommendations.append(
                TopicRecommendation(
                    concept_id=estimate.concept_id,
                    concept_name=self._concept_name(estimate.concept_id),
                    mastery=estimate.ensemble_mastery,
                    reason="ready",
                    unblocks=[],
                )
            )

        # Highest-leverage first: concepts unblocking the most other weak
        # concepts, then lowest mastery within the same leverage.
        recommendations.sort(key=lambda r: (-len(r.unblocks), r.mastery))
        logger.info("topic_recommendations_built", student_id=student_id, count=len(recommendations))
        return recommendations[:limit]

    def _concept_name(self, concept_id: UUID) -> str:
        entity = self._entities.get(concept_id)
        return entity.canonical_name if entity is not None else "this concept"
