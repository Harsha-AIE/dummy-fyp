from __future__ import annotations

import json

from edukg.core.exceptions import PedagogicalGuideError
from edukg.core.interfaces import LLMProvider
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import IntentCategory

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an intent classifier for an AI tutoring system. Classify the \
student's message into exactly one of these categories:
- "question": a genuine request for information or explanation.
- "misconception": the student states or implies a belief that is likely incorrect,
  or expresses confusion/contradiction about something they were taught.
- "example_request": the student is asking for a worked example or demonstration.
- "off_topic": greetings, thanks, small talk, or anything not about the course material.

Respond with ONLY a JSON object of this exact shape, no prose, no markdown fences:
{"intent": "question" | "misconception" | "example_request" | "off_topic", \
"confidence": number between 0.0 and 1.0, "justification": string}
"""


class LLMIntentClassifier:
    """
    Tier 2 of the Pedagogical Guide agent's cascade: invoked only when tier 1
    (KeywordIntentStrategy) isn't confident enough. Every call here is an
    escalation and gets logged by PedagogicalGuideAgent via EscalationLogStore
    — this is the "active-learning loop" data source the pipeline spec
    describes (no trainer exists to close that loop yet, but the trail is real).
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    def classify(self, text: str) -> tuple[IntentCategory, float, str]:
        raw_response = self._llm.complete_json(
            system_prompt=_SYSTEM_PROMPT, user_prompt=text, schema_name="intent_classification"
        )
        try:
            payload = json.loads(raw_response)
            intent = IntentCategory(payload["intent"])
            confidence = float(payload["confidence"])
            justification = str(payload.get("justification", ""))
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise PedagogicalGuideError(f"LLM intent response was malformed: {exc}") from exc

        logger.info("llm_intent_classified", intent=intent.value, confidence=confidence)
        return intent, max(0.0, min(1.0, confidence)), justification
