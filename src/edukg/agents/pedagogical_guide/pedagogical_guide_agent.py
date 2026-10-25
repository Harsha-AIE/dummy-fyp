from __future__ import annotations

from uuid import UUID

from edukg.agents.pedagogical_guide.escalation_log import EscalationLogStore
from edukg.agents.pedagogical_guide.llm_intent_classifier import LLMIntentClassifier
from edukg.agents.pedagogical_guide.strategy import IntentClassificationStrategy
from edukg.agents.shared.concept_tagger import ConceptTagger
from edukg.core.config import get_settings
from edukg.core.logging_setup import get_logger
from edukg.core.models import DialogueTurn

logger = get_logger(__name__)


class PedagogicalGuideAgent:
    """
    Tiered intent/misconception classifier (Phase 3's Pedagogical Guide
    Agent): tier-1 keyword strategy handles the confident majority; below
    `thresholds.intent_confidence_min`, escalates to the tier-2 LLM
    classifier and logs the escalation. Concepts mentioned in the message are
    tagged against the live KG via the shared ConceptTagger either way.
    """

    def __init__(
        self,
        tier1_strategy: IntentClassificationStrategy,
        tier2_classifier: LLMIntentClassifier,
        concept_tagger: ConceptTagger,
        escalation_log: EscalationLogStore,
        confidence_threshold: float | None = None,
    ) -> None:
        self._tier1 = tier1_strategy
        self._tier2 = tier2_classifier
        self._concept_tagger = concept_tagger
        self._escalation_log = escalation_log
        self._threshold = (
            confidence_threshold if confidence_threshold is not None else get_settings().thresholds.intent_confidence_min
        )

    def classify(self, message: str, *, student_id: str | None = None, session_id: UUID | None = None) -> DialogueTurn:
        intent, confidence = self._tier1.classify(message)
        escalated = confidence < self._threshold

        if escalated:
            intent, confidence, _justification = self._tier2.classify(message)

        detected_concepts = [entity.id for entity in self._concept_tagger.tag(message)]

        turn = DialogueTurn(
            student_id=student_id,
            session_id=session_id,
            message=message,
            intent=intent,
            confidence=confidence,
            escalated_to_llm=escalated,
            detected_concepts=detected_concepts,
        )

        if escalated:
            self._escalation_log.save(turn)

        logger.info(
            "pedagogical_guide_classified",
            intent=intent.value,
            confidence=confidence,
            escalated=escalated,
            concept_count=len(detected_concepts),
        )
        return turn
