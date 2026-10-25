from __future__ import annotations

from uuid import UUID

from edukg.agents.knowledge_tracing.bkt import compute_mastery
from edukg.agents.knowledge_tracing.dkt_service import DKTInferenceService
from edukg.core.interfaces import EntityRepository
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import IntentCategory, KTEventSource
from edukg.core.models.knowledge_tracing import KTEvent, MasteryEstimate
from edukg.core.models.session import DialogueTurn
from edukg.infrastructure.postgres.kt_store import KnowledgeTracingStore

logger = get_logger(__name__)

# The correctness-signal heuristic (flagged, not silently made — see
# MODULE_KT_NOTES.md): the Pedagogical Guide classifies *intent*, not answer
# correctness, and there is no quiz/assessment layer yet. MISCONCEPTION is
# the one intent that is itself a checked, negative correctness signal — the
# classifier already determined the student expressed a wrong belief.
# QUESTION/EXAMPLE_REQUEST are weak positive engagement signals, not real
# assessments, so they carry a low `weight` rather than being trusted as
# fully as a misconception. OFF_TOPIC/UNKNOWN produce no event at all: there
# is no reliable concept-level signal in either case. This table is the
# single place to change if/when a real assessment/orchestrator layer
# starts producing a genuine correctness signal.
_INTENT_SIGNAL: dict[IntentCategory, tuple[bool, float]] = {
    IntentCategory.MISCONCEPTION: (False, 1.0),
    IntentCategory.QUESTION: (True, 0.3),
    IntentCategory.EXAMPLE_REQUEST: (True, 0.2),
}

_LOW_MASTERY = 0.4
_HIGH_MASTERY = 0.7


def _build_explanation(concept_name: str, mastery: float, events: list[KTEvent]) -> str:
    if not events:
        return f"No evidence yet for {concept_name} — mastery is the untouched prior."

    incorrect = [e for e in events if not e.correct]
    if mastery < _LOW_MASTERY and incorrect:
        latest = incorrect[-1]
        date = latest.created_at.strftime("%b %d")
        return (
            f"Mastery of {concept_name} is low ({mastery:.0%}) — driven by a misconception "
            f"flagged on {date} while discussing {concept_name}."
        )
    if mastery >= _HIGH_MASTERY:
        return (
            f"Mastery of {concept_name} looks strong ({mastery:.0%}), based on {len(events)} "
            "engagement(s) with no unresolved misconceptions."
        )
    return f"Mastery of {concept_name} is still uncertain ({mastery:.0%}) — only {len(events)} data point(s) so far."


class KnowledgeTracingAgent:
    """The third tutor component: turns classified dialogue turns into a
    per-concept mastery vector, fusing a Bayesian Knowledge Tracing (BKT)
    estimate — available from the very first event, no training needed —
    with a Deep Knowledge Tracing (DKT) estimate when a trained model is
    available (see DKTInferenceService), and produces a natural-language
    explanation citing the events behind the score.

    Ensemble weighting is a simple, documented default: 0.5*bkt + 0.5*dkt
    when DKT has a prediction for that concept, BKT alone otherwise —
    swappable once there's a reason (e.g. real evaluation data) to weight
    one model over the other.
    """

    def __init__(self, store: KnowledgeTracingStore, entity_repo: EntityRepository, dkt_service: DKTInferenceService) -> None:
        self._store = store
        self._entity_repo = entity_repo
        self._dkt = dkt_service

    def record_dialogue_turn(self, turn: DialogueTurn) -> list[MasteryEstimate]:
        """Derives one kt_events row per concept mentioned in `turn` (per the
        heuristic table above), persists it, and recomputes/returns the
        updated mastery estimate for each touched concept. Returns an empty
        list if the turn has no student identity, mentions no concepts, or
        its intent carries no reliable correctness signal — never raises for
        those cases, since they're expected, not exceptional."""
        if not turn.student_id or not turn.detected_concepts:
            return []
        signal = _INTENT_SIGNAL.get(turn.intent)
        if signal is None:
            return []
        correct, weight = signal

        estimates: list[MasteryEstimate] = []
        for concept_id in turn.detected_concepts:
            self._store.record_event(turn.student_id, concept_id, KTEventSource.DIALOGUE, correct, weight)
            estimate = self._recompute(turn.student_id, concept_id)
            if estimate is not None:
                estimates.append(estimate)
        return estimates

    def record_quiz_result(
        self, student_id: str, concept_id: UUID, correct: bool, weight: float = 1.0
    ) -> MasteryEstimate | None:
        """A graded quiz answer is genuine ground truth, not the dialogue
        heuristic's weak engagement proxy (_INTENT_SIGNAL above) — full
        weight by default, reusing the exact record_event -> _recompute
        pair record_dialogue_turn already uses internally, just sourced as
        KTEventSource.QUIZ."""
        self._store.record_event(student_id, concept_id, KTEventSource.QUIZ, correct, weight)
        return self._recompute(student_id, concept_id)

    def get_mastery_vector(self, student_id: str) -> list[MasteryEstimate]:
        return self._store.get_mastery(student_id)

    def _recompute(self, student_id: str, concept_id: UUID) -> MasteryEstimate | None:
        concept_events = self._store.list_events(student_id, concept_id)
        bkt_mastery = compute_mastery([(e.correct, e.weight) for e in concept_events])

        dkt_mastery: float | None = None
        top_event_ids = [e.id for e in concept_events[-3:]]
        if self._dkt.is_available:
            full_history = self._store.list_events(student_id)
            prediction = self._dkt.predict(concept_id, full_history)
            if prediction is not None:
                dkt_mastery = prediction.mastery
                top_event_ids = prediction.top_contributing_event_ids

        ensemble_mastery = 0.5 * bkt_mastery + 0.5 * dkt_mastery if dkt_mastery is not None else bkt_mastery

        entity = self._entity_repo.get(concept_id)
        concept_name = entity.canonical_name if entity is not None else "this concept"
        explanation = _build_explanation(concept_name, ensemble_mastery, concept_events)

        estimate = MasteryEstimate(
            student_id=student_id,
            concept_id=concept_id,
            bkt_mastery=bkt_mastery,
            dkt_mastery=dkt_mastery,
            ensemble_mastery=ensemble_mastery,
            explanation=explanation,
            top_contributing_event_ids=top_event_ids,
        )
        self._store.upsert_mastery(estimate)
        logger.info(
            "kt_mastery_recomputed",
            student_id=student_id,
            concept_id=str(concept_id),
            bkt_mastery=bkt_mastery,
            dkt_mastery=dkt_mastery,
            ensemble_mastery=ensemble_mastery,
        )
        return estimate
