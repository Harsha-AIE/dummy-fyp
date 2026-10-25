from __future__ import annotations

import threading
from uuid import UUID

from edukg.agents.quiz.quiz_agent import QuizQuestion


class QuizRegistry:
    """In-memory registry of generated quiz questions, keyed by question id
    — same deliberate-scope choice as PipelineRunRegistry (pipeline_runner.py):
    nothing here needs to survive a process restart. Holds the full
    QuizQuestion, correct_index included, server-side only — the API layer
    must never send correct_index to the client before grading, which is
    what makes a quiz answer a meaningful correctness signal rather than a
    trivially-gameable one."""

    def __init__(self) -> None:
        self._questions: dict[UUID, QuizQuestion] = {}
        self._lock = threading.Lock()

    def put(self, question: QuizQuestion) -> None:
        with self._lock:
            self._questions[question.id] = question

    def pop(self, question_id: UUID) -> QuizQuestion | None:
        """Removes and returns the question — a question can only be
        submitted once, preventing a client from re-submitting the same
        question_id after already seeing the correct answer in the first
        submit response."""
        with self._lock:
            return self._questions.pop(question_id, None)
