from pathlib import Path

from edukg.agents.pedagogical_guide.escalation_log import EscalationLogStore
from edukg.agents.pedagogical_guide.pedagogical_guide_agent import PedagogicalGuideAgent
from edukg.core.models.enums import IntentCategory


class _FakeTier1:
    def __init__(self, intent: IntentCategory, confidence: float) -> None:
        self._intent = intent
        self._confidence = confidence

    def classify(self, text: str):
        return self._intent, self._confidence


class _FakeTier2:
    def __init__(self, intent: IntentCategory, confidence: float) -> None:
        self._intent = intent
        self._confidence = confidence
        self.calls = 0

    def classify(self, text: str):
        self.calls += 1
        return self._intent, self._confidence, "llm justification"


class _FakeConceptTagger:
    def tag(self, text: str):
        return []


def test_high_confidence_tier1_does_not_escalate(tmp_path: Path) -> None:
    tier1 = _FakeTier1(IntentCategory.OFF_TOPIC, 0.9)
    tier2 = _FakeTier2(IntentCategory.QUESTION, 0.5)
    log_store = EscalationLogStore(base_dir=tmp_path)
    agent = PedagogicalGuideAgent(tier1, tier2, _FakeConceptTagger(), log_store, confidence_threshold=0.6)

    turn = agent.classify("hi")

    assert turn.escalated_to_llm is False
    assert turn.intent == IntentCategory.OFF_TOPIC
    assert turn.confidence == 0.9
    assert tier2.calls == 0
    assert log_store.list_all() == []


def test_low_confidence_tier1_escalates_and_logs(tmp_path: Path) -> None:
    tier1 = _FakeTier1(IntentCategory.UNKNOWN, 0.0)
    tier2 = _FakeTier2(IntentCategory.MISCONCEPTION, 0.85)
    log_store = EscalationLogStore(base_dir=tmp_path)
    agent = PedagogicalGuideAgent(tier1, tier2, _FakeConceptTagger(), log_store, confidence_threshold=0.6)

    turn = agent.classify("gibberish message")

    assert turn.escalated_to_llm is True
    assert turn.intent == IntentCategory.MISCONCEPTION
    assert turn.confidence == 0.85
    assert tier2.calls == 1
    logged = log_store.list_all()
    assert len(logged) == 1
    assert logged[0].id == turn.id


def test_boundary_confidence_equal_to_threshold_does_not_escalate(tmp_path: Path) -> None:
    tier1 = _FakeTier1(IntentCategory.QUESTION, 0.6)
    tier2 = _FakeTier2(IntentCategory.MISCONCEPTION, 0.9)
    agent = PedagogicalGuideAgent(tier1, tier2, _FakeConceptTagger(), EscalationLogStore(base_dir=tmp_path), confidence_threshold=0.6)

    turn = agent.classify("message")

    assert turn.escalated_to_llm is False
    assert tier2.calls == 0
