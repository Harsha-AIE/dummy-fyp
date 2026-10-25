from __future__ import annotations

import json
from dataclasses import dataclass

from edukg.core.exceptions import TutorOrchestratorError
from edukg.core.interfaces import LLMProvider
from edukg.core.logging_setup import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are the final response layer of an AI tutoring system, sitting \
downstream of two other components that have already run: an intent classifier (tells you \
what the student is doing — asking a question, stating a misconception, requesting an \
example, or off-topic) and a retrieval system (has already found the most relevant \
grounding evidence from the course material, given to you as "Evidence" below).

Your job is to turn that into ONE reply the student actually reads, following a Socratic \
scaffolding approach rather than just stating the answer outright:
- If the intent is "misconception": do NOT simply tell them they're wrong and give the \
  correct fact. Ask a guiding question or give a hint that lets them notice the gap \
  themselves first, grounded in the Evidence. Only state the correct fact directly if the \
  evidence strongly supports it being necessary to avoid leaving a dangerous gap unresolved.
- If the intent is "question": prefer explaining via the Evidence, scaffolded to the \
  student's current mastery of the relevant concepts (given below) — lighter, more direct \
  for a concept they already have high mastery in; more guided, with a simpler first step, \
  for a concept they're weak in.
- If the intent is "example_request": give a concrete worked example grounded in the \
  Evidence, not just a restated definition.
- If the intent is "off_topic": respond briefly and warmly, and redirect toward the course \
  material without being curt.
- Never fabricate facts not supported by the Evidence. If the Evidence doesn't cover the \
  question, say so plainly rather than guessing.

After composing your reply, self-check it against the guardrail above: did you hand over the \
full answer outright for a misconception instead of guiding them to it? If so, set \
guardrail_passed to false and explain why in guardrail_note; otherwise true with a null note.

Respond with ONLY a JSON object of this exact shape, no prose, no markdown fences:
{"reply": string, "guardrail_passed": boolean, "guardrail_note": string | null}
"""


@dataclass(slots=True)
class TutorResponse:
    reply: str
    guardrail_passed: bool
    guardrail_note: str | None


class TutorResponseAgent:
    """
    The Response Generation / orchestrator layer the original architecture
    always called for (final_pro.md's MULTI-AGENT TUTOR ORCHESTRATOR diagram;
    see METHODOLOGY.md §5/§8 and MODULE_AGENTS_NOTES.md, which documented
    this as the one piece never built — the three agents returned
    independent results with no final synthesis). Merges the Pedagogical
    Guide's intent, the Knowledge Manager's retrieved evidence, and the
    Knowledge Tracing mastery signal into one reply, with a Socratic-
    scaffolding guardrail self-check baked into the same prompt rather than
    a second LLM round-trip (keeps latency reasonable for a live demo).

    Deliberately takes plain strings/tuples, not domain or API-schema
    objects — agents/ must not depend on api/schemas (api/ depends on
    agents/, never the reverse), and keeping the interface to primitives
    makes this trivially testable with a fake LLMProvider and no other
    fixtures.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    def generate(
        self,
        *,
        message: str,
        intent: str,
        concept_names: list[str],
        evidence_chunks: list[str],
        mastery_summary: list[tuple[str, float]],
    ) -> TutorResponse:
        user_prompt = self._build_user_prompt(message, intent, concept_names, evidence_chunks, mastery_summary)
        raw_response = self._llm.complete_json(
            system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt, schema_name="tutor_response"
        )
        try:
            payload = json.loads(raw_response)
            reply = str(payload["reply"])
            guardrail_passed = bool(payload["guardrail_passed"])
            guardrail_note = payload.get("guardrail_note")
            guardrail_note = str(guardrail_note) if guardrail_note is not None else None
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise TutorOrchestratorError(f"Tutor response was malformed: {exc}") from exc

        logger.info("tutor_response_generated", intent=intent, guardrail_passed=guardrail_passed)
        return TutorResponse(reply=reply, guardrail_passed=guardrail_passed, guardrail_note=guardrail_note)

    @staticmethod
    def _build_user_prompt(
        message: str,
        intent: str,
        concept_names: list[str],
        evidence_chunks: list[str],
        mastery_summary: list[tuple[str, float]],
    ) -> str:
        concepts = ", ".join(concept_names) if concept_names else "(none detected)"
        evidence = (
            "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(evidence_chunks))
            if evidence_chunks
            else "(no relevant evidence found)"
        )
        mastery = (
            "\n".join(f"- {name}: {round(value * 100)}% mastery" for name, value in mastery_summary)
            if mastery_summary
            else "(no mastery data yet for this student)"
        )
        return (
            f"Student message: {message}\n"
            f"Classified intent: {intent}\n"
            f"Concepts involved: {concepts}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Student's current mastery of these concepts:\n{mastery}"
        )
