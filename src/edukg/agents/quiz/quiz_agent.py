from __future__ import annotations

import json
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from edukg.agents.knowledge_manager.knowledge_manager_agent import KnowledgeManagerAgent
from edukg.core.exceptions import QuizGenerationError
from edukg.core.interfaces import LLMProvider
from edukg.core.logging_setup import get_logger

logger = get_logger(__name__)

_EVIDENCE_TOP_K = 3
_EXPECTED_OPTION_COUNT = 4

_SYSTEM_PROMPT = """You write one multiple-choice diagnostic/self-check question for a student \
studying a specific concept, grounded ONLY in the given course-material evidence — never invent a \
fact the evidence doesn't support, and never ask about something the evidence doesn't actually cover.

The question must test real understanding of the concept (not a trivial word-matching or \
definition-recall question), have exactly 4 answer options with exactly ONE correct, and the \
incorrect options must be plausible (realistic misconceptions or near-misses), not obviously wrong \
filler.

Respond with ONLY a JSON object of this exact shape, no prose, no markdown fences:
{"question_text": string, "options": [string, string, string, string], "correct_index": integer \
0-3, "explanation": string explaining why the correct option is right, grounded in the evidence}
"""


@dataclass(slots=True)
class QuizQuestion:
    concept_id: UUID
    concept_name: str
    question_text: str
    options: list[str]
    correct_index: int
    explanation: str
    id: UUID = field(default_factory=uuid4)


class QuizAgent:
    """
    The Quiz & Assessment engine the personalized-learning loop needs: a
    real correctness signal (a graded answer) rather than the dialogue
    intent-classification proxy KnowledgeTracingAgent._INTENT_SIGNAL
    documents as a stopgap. Reuses KnowledgeManagerAgent.retrieve exactly
    as the Tutor Orchestrator already does — the same grounding evidence
    that answers a student's question is what grounds a fair, on-topic
    question about that concept, not a separate retrieval path.

    Deliberately takes plain (concept_id, concept_name) rather than a
    domain Entity object — agents/ must not depend on api/schemas, and
    this is the one piece of concept identity every caller already has at
    hand (pipeline entities, KT mastery rows, topic recommendations).
    """

    def __init__(self, llm_provider: LLMProvider, knowledge_manager_agent: KnowledgeManagerAgent) -> None:
        self._llm = llm_provider
        self._knowledge_manager = knowledge_manager_agent

    def generate_question(
        self, concept_id: UUID, concept_name: str, *, evidence: list[str] | None = None
    ) -> QuizQuestion:
        """`evidence`, when given, is used directly instead of calling
        KnowledgeManagerAgent.retrieve() — the Diagnostic flow (document-
        scoped) passes document-scoped evidence here (see
        modules/content/document_concepts.get_document_chunk_texts_for_concept)
        so a concept-name collision elsewhere in the corpus can't leak
        unrelated evidence into a question about this document. Self-Check
        (not tied to one document) leaves `evidence` unset and keeps the
        corpus-wide retrieve() behavior, since the best evidence for a
        concept there may legitimately live in a different document."""
        if evidence is None:
            retrieved = self._knowledge_manager.retrieve(concept_name, top_k=_EVIDENCE_TOP_K)
            evidence = [r.chunk.text for r in retrieved]
        user_prompt = self._build_user_prompt(concept_name, evidence)

        raw_response = self._llm.complete_json(
            system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt, schema_name="quiz_question"
        )
        try:
            payload = json.loads(raw_response)
            question_text = str(payload["question_text"])
            options = [str(o) for o in payload["options"]]
            correct_index = int(payload["correct_index"])
            explanation = str(payload["explanation"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            raise QuizGenerationError(f"Quiz question for '{concept_name}' was malformed: {exc}") from exc

        if len(options) != _EXPECTED_OPTION_COUNT:
            raise QuizGenerationError(
                f"Quiz question for '{concept_name}' had {len(options)} options, expected {_EXPECTED_OPTION_COUNT}"
            )
        if not 0 <= correct_index < len(options):
            raise QuizGenerationError(
                f"Quiz question for '{concept_name}' had correct_index {correct_index} out of range"
            )

        logger.info("quiz_question_generated", concept_id=str(concept_id), concept_name=concept_name)
        return QuizQuestion(
            concept_id=concept_id,
            concept_name=concept_name,
            question_text=question_text,
            options=options,
            correct_index=correct_index,
            explanation=explanation,
        )

    @staticmethod
    def _build_user_prompt(concept_name: str, evidence: list[str]) -> str:
        evidence_block = (
            "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(evidence))
            if evidence
            else "(no evidence retrieved — write a general conceptual question about this topic, clearly "
            "flagged in the explanation as not grounded in specific course material)"
        )
        return f"Concept: {concept_name}\n\nEvidence:\n{evidence_block}"
