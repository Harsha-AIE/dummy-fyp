"""Router functions called directly, same convention as test_api_review.py /
test_api_pipeline.py — no FastAPI TestClient, lightweight fakes standing in
for AppResources. Covers /ask running classification and retrieval
concurrently (see routers/agents.py) — both results must come back correct
and independent of each other's timing/failure."""

import time
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from edukg.api.routers import agents
from edukg.core.models.enums import IntentCategory
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


class _FakeKnowledgeTracingAgent:
    def record_dialogue_turn(self, turn):
        return []


class _FakePedagogicalGuideAgent:
    """Simulates the tier-2 LLM escalation's real latency, so a test can
    assert retrieval isn't blocked behind it."""

    def __init__(self, delay_seconds: float = 0.0, concepts: list[UUID] | None = None) -> None:
        self.delay_seconds = delay_seconds
        self.concepts = concepts or []

    def classify(self, message: str, *, student_id: str | None = None) -> DialogueTurn:
        time.sleep(self.delay_seconds)
        return DialogueTurn(
            student_id=student_id,
            message=message,
            intent=IntentCategory.QUESTION,
            confidence=0.9,
            escalated_to_llm=self.delay_seconds > 0,
            detected_concepts=self.concepts,
        )


@dataclass
class _RetrievedChunkStub:
    chunk: object
    fused_score: float
    graph_score: float
    final_score: float


class _FakeKnowledgeManagerAgent:
    def __init__(self, results: list | None = None) -> None:
        self.results = results or []
        self.call_time: float | None = None

    def retrieve(self, query: str, top_k: int, *, student_id: str | None = None):
        self.call_time = time.monotonic()
        return self.results


@dataclass
class _TutorResponseStub:
    reply: str = "a guiding reply"
    guardrail_passed: bool = True
    guardrail_note: str | None = None


class _FakeTutorResponseAgent:
    """Records the exact arguments ask() passed through, so a test can
    assert the orchestrator actually received the classification/retrieval
    results it's supposed to merge — not just that it was called."""

    def __init__(self, response: _TutorResponseStub | None = None) -> None:
        self.response = response or _TutorResponseStub()
        self.last_call: dict | None = None

    def generate(self, **kwargs):
        self.last_call = kwargs
        return self.response


@dataclass
class _AskPayload:
    message: str
    top_k: int
    student_id: str | None


@dataclass
class _FakeResources:
    entity_repo: _FakeEntityRepo = field(default_factory=lambda: _FakeEntityRepo({}))
    knowledge_tracing_agent: _FakeKnowledgeTracingAgent = field(default_factory=_FakeKnowledgeTracingAgent)
    pedagogical_guide_agent: _FakePedagogicalGuideAgent = field(default_factory=_FakePedagogicalGuideAgent)
    knowledge_manager_agent: _FakeKnowledgeManagerAgent = field(default_factory=_FakeKnowledgeManagerAgent)
    tutor_response_agent: _FakeTutorResponseAgent = field(default_factory=_FakeTutorResponseAgent)


def test_ask_combines_classification_and_retrieval() -> None:
    concept = uuid4()
    resources = _FakeResources(
        entity_repo=_FakeEntityRepo({concept: _FakeEntity(concept, "Photosynthesis")}),
        pedagogical_guide_agent=_FakePedagogicalGuideAgent(concepts=[concept]),
    )
    payload = _AskPayload("what is photosynthesis", 5, None)

    response = agents.ask(payload, resources)

    assert response.classification.intent == "question"
    assert [c.canonical_name for c in response.classification.detected_concepts] == ["Photosynthesis"]
    assert response.retrieval.results == []
    assert response.tutor_response.reply == "a guiding reply"
    assert response.tutor_response.guardrail_passed is True


def test_ask_passes_classification_and_retrieval_results_into_the_orchestrator() -> None:
    """The Tutor Response Orchestrator is the final synthesis layer — it
    must actually receive what the other two agents produced, not be called
    with empty/default data regardless of their output."""
    concept = uuid4()
    chunk_stub = type(
        "Chunk", (), {"id": uuid4(), "text": "Photosynthesis converts light to energy.", "page_number": 1}
    )()
    chunk = _RetrievedChunkStub(chunk=chunk_stub, fused_score=0.9, graph_score=0.5, final_score=0.8)
    tutor_agent = _FakeTutorResponseAgent()
    resources = _FakeResources(
        entity_repo=_FakeEntityRepo({concept: _FakeEntity(concept, "Photosynthesis")}),
        pedagogical_guide_agent=_FakePedagogicalGuideAgent(concepts=[concept]),
        knowledge_manager_agent=_FakeKnowledgeManagerAgent(results=[chunk]),
        tutor_response_agent=tutor_agent,
    )
    payload = _AskPayload("what is photosynthesis", 5, None)

    agents.ask(payload, resources)

    assert tutor_agent.last_call is not None
    assert tutor_agent.last_call["intent"] == "question"
    assert tutor_agent.last_call["concept_names"] == ["Photosynthesis"]
    assert tutor_agent.last_call["evidence_chunks"] == ["Photosynthesis converts light to energy."]


def test_ask_runs_classification_and_retrieval_concurrently() -> None:
    """Classification and retrieval are independent — retrieval must not sit
    behind classification's (potentially several-second) LLM escalation."""
    slow_classify = _FakePedagogicalGuideAgent(delay_seconds=0.3)
    knowledge_manager = _FakeKnowledgeManagerAgent()
    resources = _FakeResources(pedagogical_guide_agent=slow_classify, knowledge_manager_agent=knowledge_manager)
    payload = _AskPayload("a slow question", 5, None)

    start = time.monotonic()
    agents.ask(payload, resources)
    elapsed = time.monotonic() - start

    assert knowledge_manager.call_time is not None
    # Retrieval (near-instant) must have run well before classification (0.3s) finished,
    # not been queued after it — proves the two ran on separate threads, not sequentially.
    assert knowledge_manager.call_time - start < 0.15
    assert elapsed < 0.3 + 0.15  # bounded by the slower branch, not the sum of both
