from __future__ import annotations

from uuid import UUID

from edukg.agents.knowledge_manager.content_indexer import ContentIndex
from edukg.core.interfaces import EntityRepository

# A concept mentioned in nearly every one of a document's own chunks reads
# as structural/scaffolding (e.g. a term repeated in every example or
# header) rather than a distinguishing topic of that document — the same
# "too common to be a useful signal" reasoning
# knowledge_manager_agent.py._drop_generic_concepts already proven for
# corpus-wide generic concepts, scoped down to one document's own chunks
# instead of the whole corpus. _MIN_GENERIC_CHUNK_COUNT keeps a short
# document (few chunks) from having every real topic concept discounted.
_GENERIC_CONCEPT_MAX_CHUNK_FRACTION = 0.8
_MIN_GENERIC_CHUNK_COUNT = 4


def get_document_top_concepts(
    content_index: ContentIndex, entity_repo: EntityRepository, document_id: UUID, limit: int = 5
) -> list[tuple[UUID, str]]:
    """The concepts a document is actually "about", ranked by how many of
    the document's own chunks mention them — used by both the Diagnostic
    quiz (Tier 3) and the Mind Map (Tier 4) to scope "top concepts" to one
    document rather than the whole corpus, reusing
    ContentIndex.chunk_concepts (chunk_id -> entity ids) and
    Chunk.document_id, both already populated at ingestion time with no new
    provenance tracking needed. Returns (concept_id, concept_name) pairs,
    most-mentioned first; an empty list if the document has no indexed
    chunks or no linked concepts."""
    document_chunk_ids = {
        chunk_id for chunk_id, chunk in content_index.chunks_by_id.items() if chunk.document_id == document_id
    }
    if not document_chunk_ids:
        return []

    concept_chunk_counts: dict[UUID, int] = {}
    for chunk_id in document_chunk_ids:
        for concept_id in set(content_index.chunk_concepts.get(chunk_id, [])):
            concept_chunk_counts[concept_id] = concept_chunk_counts.get(concept_id, 0) + 1

    threshold = max(_MIN_GENERIC_CHUNK_COUNT, len(document_chunk_ids) * _GENERIC_CONCEPT_MAX_CHUNK_FRACTION)
    ranked_concept_ids = sorted(
        (cid for cid, count in concept_chunk_counts.items() if count <= threshold),
        key=lambda cid: concept_chunk_counts[cid],
        reverse=True,
    )

    results: list[tuple[UUID, str]] = []
    for concept_id in ranked_concept_ids[:limit]:
        entity = entity_repo.get(concept_id)
        if entity is not None:
            results.append((concept_id, entity.canonical_name))
    return results


def get_document_chunk_texts_for_concept(
    content_index: ContentIndex, document_id: UUID, concept_id: UUID, limit: int = 3
) -> list[str]:
    """The text of this document's own chunks that mention `concept_id` —
    grounding evidence for the Diagnostic quiz (Tier 3) that is scoped to
    THIS document, unlike KnowledgeManagerAgent.retrieve() which searches
    the whole corpus. Found live: a generic concept name (e.g. "process")
    shared across unrelated documents let a photosynthesis document's
    diagnostic question get grounded in an entirely different document's
    (backpropagation) evidence — retrieve() has no document boundary, so a
    name collision anywhere in the corpus can leak in. Self-Check
    (mastery page, not tied to one document) still uses retrieve() as
    before; only the document-scoped diagnostic path needs this. Ordered by
    chunk sequence_index for determinism; empty if the document has no
    chunk mentioning this concept."""
    matching_chunks = [
        chunk
        for chunk_id, chunk in content_index.chunks_by_id.items()
        if chunk.document_id == document_id and concept_id in content_index.chunk_concepts.get(chunk_id, [])
    ]
    matching_chunks.sort(key=lambda c: c.sequence_index)
    return [c.text for c in matching_chunks[:limit]]
