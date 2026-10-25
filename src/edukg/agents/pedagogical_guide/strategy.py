from __future__ import annotations

from abc import ABC, abstractmethod

from edukg.core.models.enums import IntentCategory


class IntentClassificationStrategy(ABC):
    """
    Contract for the Pedagogical Guide agent's tier-1 (fast, local) intent
    classifier. Same swappable-strategy shape as
    modules/subject_detection/strategy.py's SubjectDetectionStrategy —
    keyword-based today, replaceable with a trained classifier later without
    touching PedagogicalGuideAgent.
    """

    @abstractmethod
    def classify(self, text: str) -> tuple[IntentCategory, float]:
        """Return (best_intent, confidence in [0.0, 1.0]). Must return
        (IntentCategory.UNKNOWN, 0.0) rather than guessing when signal is
        insufficient — that low confidence is what triggers tier-2 escalation."""
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__
