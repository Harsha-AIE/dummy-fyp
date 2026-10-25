"""Router functions called directly, same convention as test_api_review.py /
test_api_pipeline.py — no FastAPI TestClient, a lightweight fake resources
object standing in for AppResources."""

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from edukg.api.routers import knowledge_tracing
from edukg.core.models.enums import KTEventSource
from edukg.core.models.knowledge_tracing import KTEvent, MasteryEstimate


class _FakeEntity:
    def __init__(self, entity_id: UUID, canonical_name: str) -> None:
        self.id = entity_id
        self.canonical_name = canonical_name


class _FakeEntityRepo:
    def __init__(self, entities: dict[UUID, _FakeEntity]) -> None:
        self._entities = entities

    def get(self, entity_id: UUID):
        return self._entities.get(entity_id)


class _FakeKnowledgeTracingAgent:
    def __init__(self, estimates: list[MasteryEstimate]) -> None:
        self._estimates = estimates

    def get_mastery_vector(self, student_id: str) -> list[MasteryEstimate]:
        return [e for e in self._estimates if e.student_id == student_id]


class _FakeKTStore:
    def __init__(self, events: list[KTEvent]) -> None:
        self._events = events

    def list_events(self, student_id: str, concept_id=None):
        return [e for e in self._events if e.student_id == student_id]


@dataclass
class _FakeRecommendation:
    concept_id: UUID
    concept_name: str
    mastery: float
    reason: str
    unblocks: list[str]


class _FakeTopicRecommender:
    def __init__(self, recommendations: list[_FakeRecommendation]) -> None:
        self._recommendations = recommendations

    def recommend(self, student_id: str, limit: int = 5):
        return self._recommendations[:limit]


@dataclass
class _FakeResources:
    entity_repo: _FakeEntityRepo = field(default_factory=lambda: _FakeEntityRepo({}))
    knowledge_tracing_agent: _FakeKnowledgeTracingAgent = field(
        default_factory=lambda: _FakeKnowledgeTracingAgent([])
    )
    kt_store: _FakeKTStore = field(default_factory=lambda: _FakeKTStore([]))
    topic_recommender: _FakeTopicRecommender = field(default_factory=lambda: _FakeTopicRecommender([]))


def test_mastery_endpoint_resolves_concept_names() -> None:
    concept = uuid4()
    estimate = MasteryEstimate(
        student_id="student-1",
        concept_id=concept,
        bkt_mastery=0.6,
        dkt_mastery=None,
        ensemble_mastery=0.6,
        explanation="some explanation",
    )
    resources = _FakeResources(
        entity_repo=_FakeEntityRepo({concept: _FakeEntity(concept, "Photosynthesis")}),
        knowledge_tracing_agent=_FakeKnowledgeTracingAgent([estimate]),
    )

    response = knowledge_tracing.mastery("student-1", resources)

    assert response.student_id == "student-1"
    assert len(response.estimates) == 1
    assert response.estimates[0].concept_name == "Photosynthesis"
    assert response.estimates[0].ensemble_mastery == 0.6


def test_mastery_endpoint_only_returns_that_students_estimates() -> None:
    concept = uuid4()
    estimate_a = MasteryEstimate(
        student_id="student-1", concept_id=concept, bkt_mastery=0.5, ensemble_mastery=0.5, explanation="x"
    )
    estimate_b = MasteryEstimate(
        student_id="student-2", concept_id=concept, bkt_mastery=0.9, ensemble_mastery=0.9, explanation="y"
    )
    resources = _FakeResources(knowledge_tracing_agent=_FakeKnowledgeTracingAgent([estimate_a, estimate_b]))

    response = knowledge_tracing.mastery("student-1", resources)

    assert len(response.estimates) == 1
    assert response.estimates[0].ensemble_mastery == 0.5


def test_mastery_endpoint_missing_entity_falls_back_to_generic_name() -> None:
    concept = uuid4()
    estimate = MasteryEstimate(
        student_id="student-1", concept_id=concept, bkt_mastery=0.4, ensemble_mastery=0.4, explanation="x"
    )
    resources = _FakeResources(knowledge_tracing_agent=_FakeKnowledgeTracingAgent([estimate]))

    response = knowledge_tracing.mastery("student-1", resources)

    assert response.estimates[0].concept_name == "this concept"


def test_recommendations_endpoint_maps_topic_recommendations() -> None:
    concept = uuid4()
    rec = _FakeRecommendation(
        concept_id=concept, concept_name="Chain Rule", mastery=0.1, reason="foundational", unblocks=["Backpropagation"]
    )
    resources = _FakeResources(topic_recommender=_FakeTopicRecommender([rec]))

    response = knowledge_tracing.recommendations("student-1", resources)

    assert response.student_id == "student-1"
    assert len(response.recommendations) == 1
    assert response.recommendations[0].concept_name == "Chain Rule"
    assert response.recommendations[0].reason == "foundational"
    assert response.recommendations[0].unblocks == ["Backpropagation"]


def test_recommendations_endpoint_empty_when_no_recommendations() -> None:
    response = knowledge_tracing.recommendations("student-1", _FakeResources())

    assert response.recommendations == []


def test_events_endpoint_maps_kt_events() -> None:
    concept = uuid4()
    event = KTEvent(student_id="student-1", concept_id=concept, source=KTEventSource.DIALOGUE, correct=False, weight=1.0)
    resources = _FakeResources(kt_store=_FakeKTStore([event]))

    result = knowledge_tracing.events("student-1", resources)

    assert len(result) == 1
    assert result[0].id == event.id
    assert result[0].correct is False
    assert result[0].source == "dialogue"
