import json
from uuid import uuid4

import pytest

from edukg.agents.quiz.quiz_agent import QuizAgent
from edukg.core.exceptions import QuizGenerationError


class _FakeLLMProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_call: dict | None = None

    def complete_json(self, *, system_prompt: str, user_prompt: str, schema_name: str) -> str:
        self.last_call = {"system_prompt": system_prompt, "user_prompt": user_prompt, "schema_name": schema_name}
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
    def __init__(self, evidence_texts: list[str]) -> None:
        self.evidence_texts = evidence_texts
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def retrieve(self, query: str, top_k: int = 5, *, student_id: str | None = None):
        self.last_query = query
        self.last_top_k = top_k
        return [_FakeRetrievedChunk(t) for t in self.evidence_texts]


def _canned_response(**overrides) -> str:
    payload = {
        "question_text": "What does backpropagation compute?",
        "options": ["The gradient of the loss", "The learning rate", "The batch size", "The activation function"],
        "correct_index": 0,
        "explanation": "Backpropagation computes gradients via the chain rule.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_generate_question_includes_retrieved_evidence_in_prompt() -> None:
    llm = _FakeLLMProvider(_canned_response())
    km = _FakeKnowledgeManagerAgent(["Backprop uses the chain rule to compute gradients."])
    agent = QuizAgent(llm, km)
    concept_id = uuid4()

    agent.generate_question(concept_id, "Backpropagation")

    assert km.last_query == "Backpropagation"
    assert km.last_top_k == 3
    assert "Backprop uses the chain rule" in llm.last_call["user_prompt"]
    assert "Backpropagation" in llm.last_call["user_prompt"]


def test_generate_question_parses_llm_response_into_quiz_question() -> None:
    llm = _FakeLLMProvider(_canned_response())
    km = _FakeKnowledgeManagerAgent(["some evidence"])
    agent = QuizAgent(llm, km)
    concept_id = uuid4()

    question = agent.generate_question(concept_id, "Backpropagation")

    assert question.concept_id == concept_id
    assert question.concept_name == "Backpropagation"
    assert question.question_text == "What does backpropagation compute?"
    assert len(question.options) == 4
    assert question.correct_index == 0
    assert question.explanation


def test_generate_question_uses_given_evidence_instead_of_retrieving() -> None:
    llm = _FakeLLMProvider(_canned_response())
    km = _FakeKnowledgeManagerAgent(["evidence retrieve() would have found, but should NOT be used"])
    agent = QuizAgent(llm, km)

    agent.generate_question(uuid4(), "Backpropagation", evidence=["document-scoped evidence only"])

    assert km.last_query is None  # retrieve() was never called
    assert "document-scoped evidence only" in llm.last_call["user_prompt"]
    assert "retrieve() would have found" not in llm.last_call["user_prompt"]


def test_generate_question_with_empty_evidence_override_does_not_fall_back_to_retrieve() -> None:
    llm = _FakeLLMProvider(_canned_response())
    km = _FakeKnowledgeManagerAgent(["retrieve() would have found this"])
    agent = QuizAgent(llm, km)

    agent.generate_question(uuid4(), "Backpropagation", evidence=[])

    assert km.last_query is None  # an explicit empty list is not the same as "no override given"
    assert "(no evidence retrieved" in llm.last_call["user_prompt"]


def test_generate_question_handles_no_evidence_gracefully() -> None:
    llm = _FakeLLMProvider(_canned_response())
    km = _FakeKnowledgeManagerAgent([])
    agent = QuizAgent(llm, km)

    agent.generate_question(uuid4(), "Backpropagation")

    assert "(no evidence retrieved" in llm.last_call["user_prompt"]


def test_generate_question_raises_on_malformed_json() -> None:
    llm = _FakeLLMProvider("not json at all")
    km = _FakeKnowledgeManagerAgent(["evidence"])
    agent = QuizAgent(llm, km)

    with pytest.raises(QuizGenerationError):
        agent.generate_question(uuid4(), "Backpropagation")


def test_generate_question_raises_on_missing_fields() -> None:
    llm = _FakeLLMProvider(json.dumps({"question_text": "x"}))
    km = _FakeKnowledgeManagerAgent(["evidence"])
    agent = QuizAgent(llm, km)

    with pytest.raises(QuizGenerationError):
        agent.generate_question(uuid4(), "Backpropagation")


def test_generate_question_raises_on_wrong_option_count() -> None:
    llm = _FakeLLMProvider(_canned_response(options=["only one"]))
    km = _FakeKnowledgeManagerAgent(["evidence"])
    agent = QuizAgent(llm, km)

    with pytest.raises(QuizGenerationError):
        agent.generate_question(uuid4(), "Backpropagation")


def test_generate_question_raises_on_out_of_range_correct_index() -> None:
    llm = _FakeLLMProvider(_canned_response(correct_index=9))
    km = _FakeKnowledgeManagerAgent(["evidence"])
    agent = QuizAgent(llm, km)

    with pytest.raises(QuizGenerationError):
        agent.generate_question(uuid4(), "Backpropagation")
