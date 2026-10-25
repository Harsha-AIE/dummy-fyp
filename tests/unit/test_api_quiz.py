"""Router functions called directly, same convention as test_api_kt.py /
test_api_pipeline.py — no FastAPI TestClient, a lightweight fake resources
object standing in for AppResources."""

import json
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from edukg.agents.knowledge_manager.content_indexer import ContentIndex
from edukg.api.routers import quiz
from edukg.api.schemas import QuizDiagnosticRequest, QuizQuestionRequest, QuizSubmitRequest
from edukg.core.models import Chunk
from edukg.core.models.enums import ExtractionMethod
from edukg.core.models.knowledge_tracing import MasteryEstimate


class _FakeEntity:
    def __init__(self, entity_id: UUID, canonical_name: str) -> None:
        self.id = entity_id
        self.canonical_name = canonical_name


class _FakeEntityRepo:
    def __init__(self, entities: dict[UUID, _FakeEntity] | None = None) -> None:
        self._entities = entities or {}

    def get(self, entity_id: UUID):
        return self._entities.get(entity_id)


class _FakeLLMProvider:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete_json(self, *, system_prompt: str, user_prompt: str, schema_name: str) -> str:
        return self.response

    def model_name(self) -> str:
        return "fake-model"


class _FakeChunk:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeRetrievedChunk:
    def __init__(self, text: str) -> None:
        self.chunk = _FakeChunk(text)


class _FakeKnowledgeManagerAgent:
    def retrieve(self, query: str, top_k: int = 5, *, student_id: str | None = None):
        return [_FakeRetrievedChunk("some grounding evidence")]


class _FakeKnowledgeTracingAgent:
    def __init__(self, estimate: MasteryEstimate | None) -> None:
        self.estimate = estimate
        self.last_call: dict | None = None

    def record_quiz_result(self, student_id: str, concept_id: UUID, correct: bool, weight: float = 1.0):
        self.last_call = {"student_id": student_id, "concept_id": concept_id, "correct": correct, "weight": weight}
        return self.estimate


def _canned_llm_response(**overrides) -> str:
    payload = {
        "question_text": "What does backpropagation compute?",
        "options": ["The gradient of the loss", "The learning rate", "The batch size", "The activation function"],
        "correct_index": 0,
        "explanation": "Backpropagation computes gradients via the chain rule.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _chunk(document_id: UUID, sequence_index: int) -> Chunk:
    return Chunk(
        document_id=document_id,
        sequence_index=sequence_index,
        text="some chunk text",
        char_start=0,
        char_end=15,
        extraction_method=ExtractionMethod.NATIVE_TEXT,
        token_count=3,
    )


def _empty_content_index() -> ContentIndex:
    return ContentIndex(chunks_by_id={}, dense_index=None, lexical_index=None, chunk_concepts={})


@dataclass
class _FakeResources:
    entity_repo: _FakeEntityRepo = field(default_factory=_FakeEntityRepo)
    llm_provider: _FakeLLMProvider = field(default_factory=lambda: _FakeLLMProvider(_canned_llm_response()))
    knowledge_manager_agent: _FakeKnowledgeManagerAgent = field(default_factory=_FakeKnowledgeManagerAgent)
    knowledge_tracing_agent: _FakeKnowledgeTracingAgent = field(default_factory=lambda: _FakeKnowledgeTracingAgent(None))
    content_index: ContentIndex = field(default_factory=_empty_content_index)


def test_generate_question_returns_question_without_correct_index() -> None:
    concept = uuid4()
    resources = _FakeResources(entity_repo=_FakeEntityRepo({concept: _FakeEntity(concept, "Backpropagation")}))

    result = quiz.generate_question(QuizQuestionRequest(concept_id=concept, student_id="student-1"), resources)

    assert result.concept_name == "Backpropagation"
    assert len(result.options) == 4
    assert "correct_index" not in result.model_dump()


def test_generate_question_404s_for_unknown_concept() -> None:
    with pytest.raises(HTTPException) as exc_info:
        quiz.generate_question(QuizQuestionRequest(concept_id=uuid4(), student_id="student-1"), _FakeResources())
    assert exc_info.value.status_code == 404


def test_submit_answer_grades_correct_and_updates_mastery() -> None:
    concept = uuid4()
    estimate = MasteryEstimate(
        student_id="student-1", concept_id=concept, bkt_mastery=0.7, ensemble_mastery=0.7, explanation="x"
    )
    resources = _FakeResources(
        entity_repo=_FakeEntityRepo({concept: _FakeEntity(concept, "Backpropagation")}),
        knowledge_tracing_agent=_FakeKnowledgeTracingAgent(estimate),
    )
    question = quiz.generate_question(QuizQuestionRequest(concept_id=concept, student_id="student-1"), resources)

    result = quiz.submit_answer(
        QuizSubmitRequest(question_id=question.question_id, selected_index=0, student_id="student-1"), resources
    )

    assert result.correct is True
    assert result.correct_index == 0
    assert result.updated_mastery is not None
    assert result.updated_mastery.ensemble_mastery == 0.7
    assert resources.knowledge_tracing_agent.last_call == {
        "student_id": "student-1",
        "concept_id": concept,
        "correct": True,
        "weight": 1.0,
    }


def test_submit_answer_grades_incorrect() -> None:
    concept = uuid4()
    resources = _FakeResources(entity_repo=_FakeEntityRepo({concept: _FakeEntity(concept, "Backpropagation")}))
    question = quiz.generate_question(QuizQuestionRequest(concept_id=concept, student_id="student-1"), resources)

    result = quiz.submit_answer(
        QuizSubmitRequest(question_id=question.question_id, selected_index=1, student_id="student-1"), resources
    )

    assert result.correct is False
    assert resources.knowledge_tracing_agent.last_call["correct"] is False


def test_submit_answer_404s_for_unknown_question_id() -> None:
    with pytest.raises(HTTPException) as exc_info:
        quiz.submit_answer(
            QuizSubmitRequest(question_id=uuid4(), selected_index=0, student_id="student-1"), _FakeResources()
        )
    assert exc_info.value.status_code == 404


def test_submit_answer_404s_on_second_submission_of_same_question() -> None:
    concept = uuid4()
    resources = _FakeResources(entity_repo=_FakeEntityRepo({concept: _FakeEntity(concept, "Backpropagation")}))
    question = quiz.generate_question(QuizQuestionRequest(concept_id=concept, student_id="student-1"), resources)

    quiz.submit_answer(
        QuizSubmitRequest(question_id=question.question_id, selected_index=0, student_id="student-1"), resources
    )

    with pytest.raises(HTTPException) as exc_info:
        quiz.submit_answer(
            QuizSubmitRequest(question_id=question.question_id, selected_index=0, student_id="student-1"), resources
        )
    assert exc_info.value.status_code == 404


def test_generate_diagnostic_returns_one_question_per_top_concept() -> None:
    document_id = uuid4()
    chunk = _chunk(document_id, 0)
    concept = uuid4()
    content_index = ContentIndex(
        chunks_by_id={chunk.id: chunk}, dense_index=None, lexical_index=None, chunk_concepts={chunk.id: [concept]}
    )
    resources = _FakeResources(
        entity_repo=_FakeEntityRepo({concept: _FakeEntity(concept, "Backpropagation")}), content_index=content_index
    )

    result = quiz.generate_diagnostic(QuizDiagnosticRequest(document_id=document_id, student_id="student-1"), resources)

    assert result.document_id == document_id
    assert len(result.questions) == 1
    assert result.questions[0].concept_name == "Backpropagation"


def test_generate_diagnostic_returns_empty_for_document_with_no_concepts() -> None:
    result = quiz.generate_diagnostic(
        QuizDiagnosticRequest(document_id=uuid4(), student_id="student-1"), _FakeResources()
    )

    assert result.questions == []
