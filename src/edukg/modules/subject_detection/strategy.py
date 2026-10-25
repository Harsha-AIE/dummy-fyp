from __future__ import annotations

from abc import ABC, abstractmethod

from edukg.core.models import Subject


class SubjectDetectionStrategy(ABC):
    """
    Contract for any subject-detection algorithm. Implementations are fully
    swappable (keyword-based today; embedding/classifier-based later) without
    touching SubjectDetectionStageHandler or any other pipeline code.
    """

    @abstractmethod
    def detect(self, text: str) -> tuple[Subject, float]:
        """Return (best_subject, confidence in [0.0, 1.0]). Must return
        (Subject.UNKNOWN, 0.0) rather than guessing when signal is insufficient."""
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__
