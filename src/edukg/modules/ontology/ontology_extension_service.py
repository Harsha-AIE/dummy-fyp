from __future__ import annotations

import json
from uuid import UUID

from pydantic import ValidationError

from edukg.core.exceptions import OntologyExtensionError
from edukg.core.interfaces import LLMProvider
from edukg.core.logging_setup import get_logger
from edukg.core.models import Chunk, ConfidenceScore, Document, OntologyExtension, OntologyExtensionKind
from edukg.modules.ontology.active_ontology import ActiveOntology

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an ontology engineer assisting an educational knowledge graph system.
You will be shown existing ontology classes/relations for a subject and a sample of newly \
ingested course content. Propose ONLY ontology classes, relations, aliases, or constraints that \
are NOT already covered by the existing ontology. Do not propose duplicates or near-duplicates \
of existing names or aliases.

Respond with ONLY a JSON object of this exact shape, no prose, no markdown fences:
{
  "proposals": [
    {
      "kind": "class" | "relation" | "alias" | "constraint",
      "proposed_name": string,
      "proposed_parent_class_name": string | null,
      "proposed_domain_class_name": string | null,
      "proposed_range_class_name": string | null,
      "proposed_aliases": [string, ...],
      "proposed_description": string | null,
      "justification": string,
      "confidence": number between 0.0 and 1.0
    }
  ]
}

Rules:
- "relation" proposals MUST set proposed_domain_class_name and proposed_range_class_name to a
  class name that already exists in the ontology, OR to a class name you are also proposing.
- "alias" proposals MUST set proposed_name to an alias string and proposed_parent_class_name to
  the existing class it is an alias of.
- Return an empty "proposals" list if nothing new is warranted. Prefer fewer, well-justified
  proposals over many speculative ones.
"""

_MAX_CONTENT_CHARS = 6000


class OntologyExtensionService:
    """
    Sends a bounded sample of newly ingested document content plus a summary of the
    currently active ontology to an LLMProvider, and turns its structured JSON reply
    into a list of unvalidated, unmerged OntologyExtension proposals. Nothing here
    ever writes to the ontology — OntologyValidationService and OntologyReviewService
    are the only path from proposal to merged ontology.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    def _build_user_prompt(self, active_ontology: ActiveOntology, chunks: list[Chunk]) -> str:
        content = "\n---\n".join(c.text for c in chunks)[:_MAX_CONTENT_CHARS]
        return (
            f"Subject: {active_ontology.subject.value}\n"
            f"{active_ontology.summary()}\n\n"
            f"Sample document content:\n{content}"
        )

    def propose_extensions(
        self, document: Document, chunks: list[Chunk], active_ontology: ActiveOntology
    ) -> list[OntologyExtension]:
        if not chunks:
            logger.warning("ontology_extension_skipped_no_chunks", document_id=str(document.id))
            return []

        user_prompt = self._build_user_prompt(active_ontology, chunks)
        raw_response = self._llm.complete_json(
            system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt, schema_name="ontology_extension_proposals"
        )

        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise OntologyExtensionError(f"LLM response was not valid JSON: {exc}") from exc

        raw_proposals = payload.get("proposals")
        if not isinstance(raw_proposals, list):
            raise OntologyExtensionError("LLM response missing a 'proposals' list")

        extensions: list[OntologyExtension] = []
        for raw in raw_proposals:
            try:
                extension = self._to_extension(raw, document.id, active_ontology, raw_response)
            except (ValidationError, ValueError, KeyError) as exc:
                logger.warning("ontology_extension_proposal_rejected_malformed", error=str(exc), raw=raw)
                continue
            extensions.append(extension)

        logger.info(
            "ontology_extensions_proposed",
            document_id=str(document.id),
            subject=active_ontology.subject.value,
            proposal_count=len(extensions),
        )
        return extensions

    def _to_extension(
        self, raw: dict, document_id: UUID, active_ontology: ActiveOntology, raw_llm_response: str
    ) -> OntologyExtension:
        kind = OntologyExtensionKind(raw["kind"])

        def resolve(name: str | None) -> UUID | None:
            if not name:
                return None
            existing = active_ontology.get_class(name)
            return existing.id if existing else None

        return OntologyExtension(
            kind=kind,
            subject=active_ontology.subject,
            proposed_name=raw["proposed_name"],
            proposed_parent_class_id=resolve(raw.get("proposed_parent_class_name")),
            proposed_domain_class_id=resolve(raw.get("proposed_domain_class_name")),
            proposed_range_class_id=resolve(raw.get("proposed_range_class_name")),
            proposed_aliases=list(raw.get("proposed_aliases") or []),
            proposed_description=raw.get("proposed_description"),
            justification=raw["justification"],
            confidence=ConfidenceScore(value=float(raw["confidence"]), method="llm_judge"),
            source_document_ids=[document_id],
            generator_model=self._llm.model_name(),
            raw_llm_response=raw_llm_response,
        )
