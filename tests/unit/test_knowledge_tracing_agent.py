"""KnowledgeTracingAgent against an in-memory fake store — verifies the
intent->correctness heuristic table and the mastery upsert flow, without
touching Postgres or a real DKT model (DKTInferenceService points at an
empty tmp dir, so it's deterministically unavailable — BKT-only path)."""

from uuid import UUID, uuid4

from edukg.agents.knowledge_tracing.dkt_service import DKTInferenceService
from edukg.agents.knowledge_tracing.knowledge_tracing_agent import KnowledgeTracingAgent
from edukg.core.models.enums import IntentCategory
from edukg.core.models.knowledge_tracing import KTEvent, MasteryEstimate
from edukg.core.models.session import DialogueTurn


class _FakeEntity:
    def __init__(self, entity_id: UUID, canonical_name: str) -> None:
        self.id = entity_id
        self.canonical_name = canonical_name


class _FakeEntityRepo:
    def __init__(self, entities: dict[UUID, _FakeEntity]) -> None:
        self._entities = entities

    def get(self, entity_id: UUID):
        return self._entities.get(entity_id)


class _FakeKTStore:
    def __init__(self) -> None:
        self.events: list[KTEvent] = []
        self.mastery: dict[tuple[str, UUID], MasteryEstimate] = {}

    def record_event(self, student_id, concept_id, source, correct, weight=1.0):
        event = KTEvent(student_id=student_id, concept_id=concept_id, source=source, correct=correct, weight=weight)
        self.events.append(event)
        return event

    def list_events(self, student_id, concept_id=None):
        return [
            e for e in self.events if e.student_id == student_id and (concept_id is None or e.concept_id == concept_id)
        ]

    def upsert_mastery(self, estimate: MasteryEstimate) -> None:
        self.mastery[(estimate.student_id, estimate.concept_id)] = estimate

    def get_mastery(self, student_id):
        return [m for (sid, _), m in self.mastery.items() if sid == student_id]


def _agent(tmp_path, entities: dict[UUID, _FakeEntity] | None = None) -> tuple[KnowledgeTracingAgent, _FakeKTStore]:
    store = _FakeKTStore()
    entity_repo = _FakeEntityRepo(entities or {})
    dkt = DKTInferenceService(model_dir=tmp_path / "no-model-here")
    return KnowledgeTracingAgent(store, entity_repo, dkt), store


def _turn(intent: IntentCategory, concepts: list[UUID], student_id: str | None = "student-1") -> DialogueTurn:
    return DialogueTurn(student_id=student_id, message="msg", intent=intent, confidence=0.9, detected_concepts=concepts)


def test_misconception_records_negative_full_weight_event(tmp_path) -> None:
    concept = uuid4()
    agent, store = _agent(tmp_path, {concept: _FakeEntity(concept, "Photosynthesis")})

    agent.record_dialogue_turn(_turn(IntentCategory.MISCONCEPTION, [concept]))

    assert len(store.events) == 1
    assert store.events[0].correct is False
    assert store.events[0].weight == 1.0


def test_question_records_weak_positive_event(tmp_path) -> None:
    concept = uuid4()
    agent, store = _agent(tmp_path, {concept: _FakeEntity(concept, "Photosynthesis")})

    agent.record_dialogue_turn(_turn(IntentCategory.QUESTION, [concept]))

    assert store.events[0].correct is True
    assert store.events[0].weight == 0.3


def test_example_request_records_weaker_positive_event(tmp_path) -> None:
    concept = uuid4()
    agent, store = _agent(tmp_path, {concept: _FakeEntity(concept, "Photosynthesis")})

    agent.record_dialogue_turn(_turn(IntentCategory.EXAMPLE_REQUEST, [concept]))

    assert store.events[0].correct is True
    assert store.events[0].weight == 0.2


def test_off_topic_produces_no_event(tmp_path) -> None:
    concept = uuid4()
    agent, store = _agent(tmp_path, {concept: _FakeEntity(concept, "Photosynthesis")})

    result = agent.record_dialogue_turn(_turn(IntentCategory.OFF_TOPIC, [concept]))

    assert result == []
    assert store.events == []


def test_no_student_id_produces_no_event(tmp_path) -> None:
    concept = uuid4()
    agent, store = _agent(tmp_path, {concept: _FakeEntity(concept, "Photosynthesis")})

    result = agent.record_dialogue_turn(_turn(IntentCategory.MISCONCEPTION, [concept], student_id=None))

    assert result == []
    assert store.events == []


def test_no_detected_concepts_produces_no_event(tmp_path) -> None:
    agent, store = _agent(tmp_path)

    result = agent.record_dialogue_turn(_turn(IntentCategory.MISCONCEPTION, []))

    assert result == []
    assert store.events == []


def test_record_dialogue_turn_upserts_mastery_estimate(tmp_path) -> None:
    concept = uuid4()
    agent, store = _agent(tmp_path, {concept: _FakeEntity(concept, "Photosynthesis")})

    estimates = agent.record_dialogue_turn(_turn(IntentCategory.MISCONCEPTION, [concept]))

    assert len(estimates) == 1
    assert estimates[0].concept_id == concept
    assert estimates[0].dkt_mastery is None  # no DKT model loaded in this test
    assert estimates[0].ensemble_mastery == estimates[0].bkt_mastery
    assert ("student-1", concept) in store.mastery


def test_record_quiz_result_records_full_weight_correct_event(tmp_path) -> None:
    concept = uuid4()
    agent, store = _agent(tmp_path, {concept: _FakeEntity(concept, "Photosynthesis")})

    estimate = agent.record_quiz_result("student-1", concept, correct=True)

    assert len(store.events) == 1
    assert store.events[0].source.value == "quiz"
    assert store.events[0].correct is True
    assert store.events[0].weight == 1.0
    assert estimate is not None
    assert estimate.concept_id == concept


def test_record_quiz_result_records_incorrect_answer(tmp_path) -> None:
    concept = uuid4()
    agent, store = _agent(tmp_path, {concept: _FakeEntity(concept, "Photosynthesis")})

    agent.record_quiz_result("student-1", concept, correct=False)

    assert store.events[0].correct is False
    assert store.events[0].weight == 1.0


def test_record_quiz_result_ignores_weight_override_default(tmp_path) -> None:
    concept = uuid4()
    agent, store = _agent(tmp_path, {concept: _FakeEntity(concept, "Photosynthesis")})

    agent.record_quiz_result("student-1", concept, correct=True, weight=0.5)

    assert store.events[0].weight == 0.5


def test_get_mastery_vector_reads_from_store(tmp_path) -> None:
    concept = uuid4()
    agent, store = _agent(tmp_path, {concept: _FakeEntity(concept, "Photosynthesis")})
    agent.record_dialogue_turn(_turn(IntentCategory.QUESTION, [concept]))

    vector = agent.get_mastery_vector("student-1")

    assert len(vector) == 1
    assert vector[0].concept_id == concept
