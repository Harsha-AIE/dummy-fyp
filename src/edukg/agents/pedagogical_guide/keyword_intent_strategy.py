from __future__ import annotations

import re

from edukg.agents.pedagogical_guide.intent_keywords import INTENT_KEYWORDS
from edukg.agents.pedagogical_guide.strategy import IntentClassificationStrategy
from edukg.core.models.enums import IntentCategory


class KeywordIntentStrategy(IntentClassificationStrategy):
    """
    Scores each IntentCategory by counting occurrences of its phrase bank in
    the message (case-insensitive, whole-phrase-boundary match), normalizes
    by bank size, and returns the top-scoring intent with a confidence
    derived from how dominant it was over the rest. Fully offline,
    deterministic.

    Phrase boundaries are anchored with \\b at both ends (not just per-word
    like KeywordSubjectDetectionStrategy) — a plain substring search lets a
    short phrase like "why do" false-positive-match inside an unrelated word
    ("doesn't" contains "do" as its first two letters), which previously
    made "why doesn't this work?" score as QUESTION instead of MISCONCEPTION.
    \\b works here because every phrase in the bank starts and ends with a
    word character.
    """

    def __init__(self, keyword_banks: dict[IntentCategory, list[str]] | None = None) -> None:
        self._keyword_banks = keyword_banks or INTENT_KEYWORDS
        self._compiled: dict[IntentCategory, list[re.Pattern]] = {
            intent: [re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE) for phrase in phrases]
            for intent, phrases in self._keyword_banks.items()
        }

    def classify(self, text: str) -> tuple[IntentCategory, float]:
        if not text.strip():
            return IntentCategory.UNKNOWN, 0.0

        raw_scores: dict[IntentCategory, float] = {}
        for intent, patterns in self._compiled.items():
            hits = sum(len(p.findall(text)) for p in patterns)
            raw_scores[intent] = hits / len(patterns) if patterns else 0.0

        total = sum(raw_scores.values())
        if total <= 0.0:
            return IntentCategory.UNKNOWN, 0.0

        best_intent = max(raw_scores, key=raw_scores.get)
        confidence = raw_scores[best_intent] / total
        return best_intent, round(min(confidence, 1.0), 4)
