from edukg.agents.pedagogical_guide.keyword_intent_strategy import KeywordIntentStrategy
from edukg.core.models.enums import IntentCategory


def test_classifies_misconception_phrase() -> None:
    intent, confidence = KeywordIntentStrategy().classify("Wait, why doesn't this work? I thought gravity was constant.")
    assert intent == IntentCategory.MISCONCEPTION
    assert confidence > 0.0


def test_classifies_example_request() -> None:
    intent, _confidence = KeywordIntentStrategy().classify("Can you show me an example of this?")
    assert intent == IntentCategory.EXAMPLE_REQUEST


def test_classifies_off_topic_greeting() -> None:
    intent, _confidence = KeywordIntentStrategy().classify("hi there, thanks!")
    assert intent == IntentCategory.OFF_TOPIC


def test_no_signal_returns_unknown_with_zero_confidence() -> None:
    intent, confidence = KeywordIntentStrategy().classify("asdkjfh qweoiuvn")
    assert intent == IntentCategory.UNKNOWN
    assert confidence == 0.0


def test_blank_text_returns_unknown() -> None:
    intent, confidence = KeywordIntentStrategy().classify("   ")
    assert intent == IntentCategory.UNKNOWN
    assert confidence == 0.0
