import json

import pytest

from edukg.agents.tutor_orchestrator.tutor_response_agent import TutorResponseAgent
from edukg.core.exceptions import TutorOrchestratorError


class _FakeLLMProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_call: dict | None = None

    def complete_json(self, *, system_prompt: str, user_prompt: str, schema_name: str) -> str:
        self.last_call = {"system_prompt": system_prompt, "user_prompt": user_prompt, "schema_name": schema_name}
        return self.response

    def model_name(self) -> str:
        return "fake-model"


def _canned_response(**overrides) -> str:
    payload = {"reply": "Let's think about what you already know first.", "guardrail_passed": True, "guardrail_note": None}
    payload.update(overrides)
    return json.dumps(payload)


def test_generate_builds_a_prompt_including_evidence_and_mastery() -> None:
    llm = _FakeLLMProvider(_canned_response())
    agent = TutorResponseAgent(llm)

    agent.generate(
        message="Why do microorganisms produce antibiotics?",
        intent="misconception",
        concept_names=["Antibiotic Resistance"],
        evidence_chunks=["Antibiotics are secondary metabolites that inhibit competitor growth."],
        mastery_summary=[("Antibiotic Resistance", 0.2)],
    )

    assert llm.last_call is not None
    assert "Antibiotics are secondary metabolites" in llm.last_call["user_prompt"]
    assert "Antibiotic Resistance: 20% mastery" in llm.last_call["user_prompt"]
    assert "misconception" in llm.last_call["user_prompt"]


def test_generate_parses_the_llm_response_into_a_tutor_response() -> None:
    llm = _FakeLLMProvider(_canned_response(reply="Here's a hint.", guardrail_passed=False, guardrail_note="gave the answer away"))
    agent = TutorResponseAgent(llm)

    result = agent.generate(
        message="What is DNA?", intent="question", concept_names=[], evidence_chunks=[], mastery_summary=[]
    )

    assert result.reply == "Here's a hint."
    assert result.guardrail_passed is False
    assert result.guardrail_note == "gave the answer away"


def test_generate_handles_empty_evidence_and_mastery_gracefully() -> None:
    llm = _FakeLLMProvider(_canned_response())
    agent = TutorResponseAgent(llm)

    result = agent.generate(message="hi", intent="off_topic", concept_names=[], evidence_chunks=[], mastery_summary=[])

    assert result.reply
    assert "(no relevant evidence found)" in llm.last_call["user_prompt"]
    assert "(no mastery data yet for this student)" in llm.last_call["user_prompt"]


def test_generate_raises_tutor_orchestrator_error_on_malformed_json() -> None:
    llm = _FakeLLMProvider("not json at all")
    agent = TutorResponseAgent(llm)

    with pytest.raises(TutorOrchestratorError):
        agent.generate(message="hi", intent="question", concept_names=[], evidence_chunks=[], mastery_summary=[])


def test_generate_raises_tutor_orchestrator_error_on_missing_fields() -> None:
    llm = _FakeLLMProvider(json.dumps({"reply": "ok"}))  # missing guardrail_passed
    agent = TutorResponseAgent(llm)

    with pytest.raises(TutorOrchestratorError):
        agent.generate(message="hi", intent="question", concept_names=[], evidence_chunks=[], mastery_summary=[])
