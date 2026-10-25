from __future__ import annotations

import json

from edukg.core.interfaces import LLMProvider
from edukg.core.logging_setup import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """You identify relationships between two concepts mentioned together in a \
sentence, for a knowledge graph. You may ONLY choose from this exact list of relation names: \
{relation_names}. If none of them genuinely holds between the two concepts based on the sentence, \
say so — never guess or invent a relation not on the list.

Respond with ONLY a JSON object of this exact shape, no prose, no markdown fences:
{{"relation": one of the listed names, or null if none applies, \
"subject_is_first": true if the FIRST named concept is the relation's subject, false if the SECOND is}}
"""


class LLMRelationExtractor:
    """
    Tier 3 of the relation-extraction cascade (stage 8): for an entity pair
    the deterministic tiers (dependency parse, cue phrase) found no signal
    for, asks the LLM directly whether a known ontology relation holds
    between them, grounded in the sentence they co-occur in — a direct
    semantic judgment, not a pattern match. Only invoked by
    RelationExtractionService for ontology-verified entity pairs (same gate
    CooccurrenceFallback already uses), which bounds the added LLM cost to
    exactly the pairs that would otherwise fall through to the weakest
    signal (bare co-occurrence), not every pair in the document.

    Any failure — malformed JSON, an LLM call error, a relation name outside
    the given candidate list — degrades to None (falls through to
    co-occurrence) rather than raising. Unlike TutorResponseAgent/QuizAgent,
    this runs inside a batch pipeline stage processing hundreds of pairs per
    document; one bad response must never abort the run.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    def extract(
        self, sentence_text: str, entity_a_name: str, entity_b_name: str, candidate_relations: list[str]
    ) -> tuple[str, bool] | None:
        """Returns (relation_name, reversed) — same contract CuePhraseExtractor/
        VERB_TO_RELATION use — or None if no relation applies or anything
        went wrong. `reversed=True` means entity_a is the relation's
        *object*, not subject, despite being named first."""
        if not candidate_relations:
            return None

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(relation_names=", ".join(candidate_relations))
        user_prompt = f'Sentence: "{sentence_text}"\nFirst concept: {entity_a_name}\nSecond concept: {entity_b_name}'

        try:
            raw = self._llm.complete_json(
                system_prompt=system_prompt, user_prompt=user_prompt, schema_name="relation_extraction"
            )
            payload = json.loads(raw)
            relation = payload.get("relation")
            if relation is None or relation not in candidate_relations:
                return None
            subject_is_first = bool(payload.get("subject_is_first", True))
            logger.info(
                "llm_relation_extraction_fired",
                relation=relation,
                entity_a=entity_a_name,
                entity_b=entity_b_name,
                reversed=not subject_is_first,
            )
            return relation, not subject_is_first
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.warning("llm_relation_extraction_malformed_response", error=str(exc))
            return None
        except Exception as exc:  # noqa: BLE001 - any LLM call failure (timeout, auth outage, etc.)
            # degrades this one pair to the co-occurrence tier instead of aborting the whole batch.
            logger.warning("llm_relation_extraction_failed", error=str(exc))
            return None
