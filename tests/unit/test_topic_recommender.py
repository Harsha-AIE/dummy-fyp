from dataclasses import dataclass
from uuid import UUID, uuid4

from edukg.agents.knowledge_tracing.topic_recommender import TopicRecommender


@dataclass
class _FakeEstimate:
    concept_id: UUID
    ensemble_mastery: float


class _FakeKnowledgeTracingAgent:
    def __init__(self, estimates: list[_FakeEstimate]) -> None:
        self._estimates = estimates

    def get_mastery_vector(self, student_id: str) -> list[_FakeEstimate]:
        return self._estimates


class _FakeGraphRepo:
    def __init__(self, prereqs: dict[UUID, list[UUID]]) -> None:
        self._prereqs = prereqs

    def get_prerequisites_bulk(self, concept_ids: list[UUID], predicate: str = "is_prerequisite_of") -> dict[UUID, list[UUID]]:
        return {cid: self._prereqs.get(cid, []) for cid in concept_ids}


class _FakeEntity:
    def __init__(self, name: str) -> None:
        self.canonical_name = name


class _FakeEntityRepo:
    def __init__(self, names: dict[UUID, str]) -> None:
        self._names = names

    def get(self, entity_id: UUID):
        name = self._names.get(entity_id)
        return _FakeEntity(name) if name else None


def test_recommends_weak_concept_directly_when_no_unresolved_prerequisite() -> None:
    concept = uuid4()
    kt = _FakeKnowledgeTracingAgent([_FakeEstimate(concept, 0.2)])
    graph = _FakeGraphRepo({})  # no prerequisites at all
    entities = _FakeEntityRepo({concept: "Backpropagation"})
    recommender = TopicRecommender(kt, graph, entities)

    recs = recommender.recommend("student-1")

    assert len(recs) == 1
    assert recs[0].concept_id == concept
    assert recs[0].concept_name == "Backpropagation"
    assert recs[0].reason == "ready"
    assert recs[0].unblocks == []


def test_recommends_the_weak_prerequisite_instead_of_the_concept_it_blocks() -> None:
    advanced, foundation = uuid4(), uuid4()
    kt = _FakeKnowledgeTracingAgent([_FakeEstimate(advanced, 0.2), _FakeEstimate(foundation, 0.1)])
    graph = _FakeGraphRepo({advanced: [foundation]})
    entities = _FakeEntityRepo({advanced: "Backpropagation", foundation: "Chain Rule"})
    recommender = TopicRecommender(kt, graph, entities)

    recs = recommender.recommend("student-1")

    assert len(recs) == 1
    assert recs[0].concept_id == foundation
    assert recs[0].concept_name == "Chain Rule"
    assert recs[0].reason == "foundational"
    assert recs[0].unblocks == ["Backpropagation"]


def test_recommends_concept_directly_when_its_prerequisite_is_already_mastered() -> None:
    advanced, mastered_foundation = uuid4(), uuid4()
    kt = _FakeKnowledgeTracingAgent([_FakeEstimate(advanced, 0.2), _FakeEstimate(mastered_foundation, 0.9)])
    graph = _FakeGraphRepo({advanced: [mastered_foundation]})
    entities = _FakeEntityRepo({advanced: "Backpropagation", mastered_foundation: "Chain Rule"})
    recommender = TopicRecommender(kt, graph, entities)

    recs = recommender.recommend("student-1")

    assert len(recs) == 1
    assert recs[0].concept_id == advanced
    assert recs[0].reason == "ready"


def test_merges_a_shared_blocking_prerequisite_and_ranks_it_first() -> None:
    # Regression-style test: one weak foundational concept blocks two
    # different weak advanced concepts -- it should appear once, with both
    # names merged into unblocks, and sort ahead of a single-leverage "ready"
    # recommendation.
    foundation, advanced_a, advanced_b, unrelated = uuid4(), uuid4(), uuid4(), uuid4()
    kt = _FakeKnowledgeTracingAgent(
        [
            _FakeEstimate(foundation, 0.1),
            _FakeEstimate(advanced_a, 0.2),
            _FakeEstimate(advanced_b, 0.3),
            _FakeEstimate(unrelated, 0.4),
        ]
    )
    graph = _FakeGraphRepo({advanced_a: [foundation], advanced_b: [foundation]})
    entities = _FakeEntityRepo(
        {foundation: "Chain Rule", advanced_a: "Backpropagation", advanced_b: "Gradient Descent", unrelated: "Overfitting"}
    )
    recommender = TopicRecommender(kt, graph, entities)

    recs = recommender.recommend("student-1")

    foundation_rec = next(r for r in recs if r.concept_id == foundation)
    assert sorted(foundation_rec.unblocks) == ["Backpropagation", "Gradient Descent"]
    assert recs[0].concept_id == foundation  # highest leverage (unblocks 2) sorts first


def test_no_weak_concepts_returns_empty_list() -> None:
    kt = _FakeKnowledgeTracingAgent([_FakeEstimate(uuid4(), 0.9)])
    recommender = TopicRecommender(kt, _FakeGraphRepo({}), _FakeEntityRepo({}))

    assert recommender.recommend("student-1") == []


def test_respects_limit() -> None:
    estimates = [_FakeEstimate(uuid4(), 0.1) for _ in range(10)]
    kt = _FakeKnowledgeTracingAgent(estimates)
    names = {e.concept_id: f"Concept {i}" for i, e in enumerate(estimates)}
    recommender = TopicRecommender(kt, _FakeGraphRepo({}), _FakeEntityRepo(names))

    recs = recommender.recommend("student-1", limit=3)

    assert len(recs) == 3
