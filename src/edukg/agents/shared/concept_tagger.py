from __future__ import annotations

from uuid import uuid4

from edukg.core.models import Chunk, Entity
from edukg.core.models.enums import ExtractionMethod
from edukg.modules.entity_extraction.mention_extractor import MentionExtractor
from edukg.modules.entity_linking.linking_service import EntityLinkingService

_CHARS_PER_TOKEN = 4


class ConceptTagger:
    """
    Resolves KG entities mentioned in an arbitrary string (a student message
    or a search query) for use by the agent layer. Reuses Module 3's mention
    extraction + entity linking rather than duplicating NER logic — but in
    read-only mode: a term with no sufficiently similar existing entity is
    simply dropped, never used to mint a new KG entity. Minting entities is a
    deliberate act of the reviewed ingestion pipeline; a live user query must
    not be able to silently grow the ontology-governed graph.
    """

    def __init__(self, mention_extractor: MentionExtractor, linking_service: EntityLinkingService) -> None:
        self._mention_extractor = mention_extractor
        self._linking_service = linking_service

    def tag(self, text: str) -> list[Entity]:
        if not text.strip():
            return []

        document_id = uuid4()
        chunk = Chunk(
            document_id=document_id,
            sequence_index=0,
            text=text,
            char_start=0,
            char_end=len(text),
            extraction_method=ExtractionMethod.NATIVE_TEXT,
            token_count=max(1, len(text) // _CHARS_PER_TOKEN),
        )

        mentions = self._mention_extractor.extract(chunk)
        return self._linking_service.link(mentions, document_id, active_ontology=None, create_if_missing=False)
