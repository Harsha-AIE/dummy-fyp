from __future__ import annotations

import re

from edukg.core.models import Subject
from edukg.modules.subject_detection.strategy import SubjectDetectionStrategy
from edukg.modules.subject_detection.subject_keywords import SUBJECT_KEYWORDS


class KeywordSubjectDetectionStrategy(SubjectDetectionStrategy):
    """
    Scores each Subject by counting occurrences of its keyword bank in the document
    text (case-insensitive, phrase-aware), normalizes by bank size, and returns the
    top-scoring subject with a confidence derived from how dominant it was over the
    runner-up. Fully offline, deterministic, and dependency-free.
    """

    def __init__(self, keyword_banks: dict[Subject, list[str]] | None = None) -> None:
        self._keyword_banks = keyword_banks or SUBJECT_KEYWORDS
        self._compiled: dict[Subject, list[re.Pattern]] = {
            subject: [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in keywords]
            for subject, keywords in self._keyword_banks.items()
        }

    def detect(self, text: str) -> tuple[Subject, float]:
        if not text.strip():
            return Subject.UNKNOWN, 0.0

        raw_scores: dict[Subject, float] = {}
        for subject, patterns in self._compiled.items():
            hits = sum(len(p.findall(text)) for p in patterns)
            raw_scores[subject] = hits / len(patterns) if patterns else 0.0

        total = sum(raw_scores.values())
        if total <= 0.0:
            return Subject.UNKNOWN, 0.0

        best_subject = max(raw_scores, key=raw_scores.get)
        confidence = raw_scores[best_subject] / total
        return best_subject, round(min(confidence, 1.0), 4)
