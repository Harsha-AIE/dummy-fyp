from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from uuid import UUID

from edukg.agents.knowledge_manager.bm25_index import BM25Index
from edukg.core.interfaces import DocumentRepository, EmbeddingProvider, EntityRepository, VectorIndexRepository
from edukg.core.logging_setup import get_logger
from edukg.core.models import Chunk

logger = get_logger(__name__)

_EMBED_BATCH_SIZE = 32


@dataclass(slots=True)
class ContentIndex:
    """In-memory index ready for KnowledgeManagerAgent to query."""

    chunks_by_id: dict[UUID, Chunk]
    dense_index: VectorIndexRepository
    lexical_index: BM25Index
    chunk_concepts: dict[UUID, list[UUID]] = field(default_factory=dict)  # chunk_id -> entity ids


class ContentIndexService:
    """
    Builds the Knowledge Manager agent's retrieval index across every
    ingested document: dense embeddings (chunk-keyed FAISS index, reusing
    the same FAISSVectorIndexRepository class entity linking uses — it's
    already generic over (UUID, embedding) pairs) + BM25 lexical index, plus
    a chunk_id -> entity_ids map derived from Entity.provenance (no new
    storage needed — every Entity already records which chunk it came from).
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        entity_repository: EntityRepository,
        embedding_provider: EmbeddingProvider,
        dense_index: VectorIndexRepository,
        lexical_index: BM25Index | None = None,
    ) -> None:
        self._documents = document_repository
        self._entities = entity_repository
        self._embeddings = embedding_provider
        self._dense_index = dense_index
        self._lexical_index = lexical_index or BM25Index()

    def build(self) -> ContentIndex:
        chunks: list[Chunk] = []
        for document in self._documents.list_all_documents():
            chunks.extend(self._documents.get_chunks(document.id))

        chunk_concepts = self._build_chunk_concept_map()

        if not chunks:
            logger.warning("content_index_build_no_chunks")
            return ContentIndex(chunks_by_id={}, dense_index=self._dense_index, lexical_index=self._lexical_index)

        texts = [c.text for c in chunks]
        for batch_chunks, batch_texts in self._batched(chunks, texts, _EMBED_BATCH_SIZE):
            vectors = self._embeddings.embed(batch_texts)
            for chunk, vector in zip(batch_chunks, vectors, strict=True):
                self._dense_index.add(chunk.id, vector)

        self._lexical_index.build([c.id for c in chunks], texts)

        logger.info("content_index_built", chunk_count=len(chunks), concept_mapped_chunks=len(chunk_concepts))
        return ContentIndex(
            chunks_by_id={c.id: c for c in chunks},
            dense_index=self._dense_index,
            lexical_index=self._lexical_index,
            chunk_concepts=chunk_concepts,
        )

    def _build_chunk_concept_map(self) -> dict[UUID, list[UUID]]:
        mapping: dict[UUID, list[UUID]] = defaultdict(list)
        for entity in self._entities.list_all():
            for provenance in entity.provenance:
                if provenance.chunk_id is not None:
                    mapping[provenance.chunk_id].append(entity.id)
        return dict(mapping)

    @staticmethod
    def _batched(chunks: list[Chunk], texts: list[str], size: int):
        for start in range(0, len(chunks), size):
            yield chunks[start : start + size], texts[start : start + size]
