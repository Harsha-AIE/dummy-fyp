from uuid import UUID, uuid4

from edukg.agents.knowledge_manager.content_indexer import ContentIndex
from edukg.core.models import Chunk
from edukg.core.models.enums import ExtractionMethod
from edukg.modules.content.document_concepts import get_document_chunk_texts_for_concept, get_document_top_concepts


class _FakeEntity:
    def __init__(self, entity_id: UUID, canonical_name: str) -> None:
        self.id = entity_id
        self.canonical_name = canonical_name


class _FakeEntityRepo:
    def __init__(self, entities: dict[UUID, _FakeEntity]) -> None:
        self._entities = entities

    def get(self, entity_id: UUID):
        return self._entities.get(entity_id)


def _chunk(document_id: UUID, sequence_index: int, text: str = "some chunk text") -> Chunk:
    return Chunk(
        document_id=document_id,
        sequence_index=sequence_index,
        text=text,
        char_start=0,
        char_end=15,
        extraction_method=ExtractionMethod.NATIVE_TEXT,
        token_count=3,
    )


def _index(chunks: list[Chunk], chunk_concepts: dict[UUID, list[UUID]]) -> ContentIndex:
    return ContentIndex(chunks_by_id={c.id: c for c in chunks}, dense_index=None, lexical_index=None, chunk_concepts=chunk_concepts)


def test_ranks_concepts_by_document_chunk_frequency_excluding_universal_ones() -> None:
    document_id = uuid4()
    chunks = [_chunk(document_id, i) for i in range(6)]
    concept_universal, concept_common, concept_rare, concept_rarest = uuid4(), uuid4(), uuid4(), uuid4()

    chunk_concepts = {}
    for chunk in chunks:
        chunk_concepts[chunk.id] = [concept_universal]  # in all 6 chunks -> excluded as structural
    for chunk in chunks[:4]:
        chunk_concepts[chunk.id].append(concept_common)  # in 4 chunks
    for chunk in chunks[:2]:
        chunk_concepts[chunk.id].append(concept_rare)  # in 2 chunks
    chunk_concepts[chunks[0].id].append(concept_rarest)  # in 1 chunk

    index = _index(chunks, chunk_concepts)
    entity_repo = _FakeEntityRepo(
        {
            concept_universal: _FakeEntity(concept_universal, "Universal"),
            concept_common: _FakeEntity(concept_common, "Common"),
            concept_rare: _FakeEntity(concept_rare, "Rare"),
            concept_rarest: _FakeEntity(concept_rarest, "Rarest"),
        }
    )

    results = get_document_top_concepts(index, entity_repo, document_id, limit=5)

    names = [name for _, name in results]
    assert "Universal" not in names
    assert names == ["Common", "Rare", "Rarest"]


def test_scopes_to_only_the_given_documents_chunks() -> None:
    doc_a, doc_b = uuid4(), uuid4()
    chunk_a = _chunk(doc_a, 0)
    chunk_b = _chunk(doc_b, 0)
    concept_a, concept_b = uuid4(), uuid4()
    index = _index([chunk_a, chunk_b], {chunk_a.id: [concept_a], chunk_b.id: [concept_b]})
    entity_repo = _FakeEntityRepo({concept_a: _FakeEntity(concept_a, "A"), concept_b: _FakeEntity(concept_b, "B")})

    results = get_document_top_concepts(index, entity_repo, doc_a, limit=5)

    assert [name for _, name in results] == ["A"]


def test_document_with_no_chunks_returns_empty() -> None:
    index = _index([], {})
    entity_repo = _FakeEntityRepo({})

    assert get_document_top_concepts(index, entity_repo, uuid4(), limit=5) == []


def test_document_with_chunks_but_no_linked_concepts_returns_empty() -> None:
    document_id = uuid4()
    chunk = _chunk(document_id, 0)
    index = _index([chunk], {})
    entity_repo = _FakeEntityRepo({})

    assert get_document_top_concepts(index, entity_repo, document_id, limit=5) == []


def test_limit_truncates_results() -> None:
    document_id = uuid4()
    chunks = [_chunk(document_id, i) for i in range(3)]
    concepts = [uuid4() for _ in range(3)]
    chunk_concepts = {chunks[i].id: [concepts[i]] for i in range(3)}
    index = _index(chunks, chunk_concepts)
    entity_repo = _FakeEntityRepo({c: _FakeEntity(c, f"Concept{i}") for i, c in enumerate(concepts)})

    results = get_document_top_concepts(index, entity_repo, document_id, limit=2)

    assert len(results) == 2


def test_get_document_chunk_texts_for_concept_returns_only_this_documents_chunks() -> None:
    doc_a, doc_b = uuid4(), uuid4()
    concept = uuid4()
    chunk_a = _chunk(doc_a, 0, text="Photosynthesis converts sunlight into chemical energy.")
    chunk_b = _chunk(doc_b, 0, text="Backpropagation computes gradients via the chain rule.")
    # Same concept id mentioned in both documents' chunks (the real-world
    # scenario a generic/colliding concept name produces) — only doc_a's
    # text must come back when asking for doc_a.
    index = _index([chunk_a, chunk_b], {chunk_a.id: [concept], chunk_b.id: [concept]})

    results = get_document_chunk_texts_for_concept(index, doc_a, concept)

    assert results == ["Photosynthesis converts sunlight into chemical energy."]


def test_get_document_chunk_texts_for_concept_orders_by_sequence_index() -> None:
    document_id = uuid4()
    concept = uuid4()
    chunk_1 = _chunk(document_id, 1, text="second")
    chunk_0 = _chunk(document_id, 0, text="first")
    index = _index([chunk_1, chunk_0], {chunk_1.id: [concept], chunk_0.id: [concept]})

    results = get_document_chunk_texts_for_concept(index, document_id, concept)

    assert results == ["first", "second"]


def test_get_document_chunk_texts_for_concept_respects_limit() -> None:
    document_id = uuid4()
    concept = uuid4()
    chunks = [_chunk(document_id, i, text=f"chunk {i}") for i in range(5)]
    index = _index(chunks, {c.id: [concept] for c in chunks})

    results = get_document_chunk_texts_for_concept(index, document_id, concept, limit=2)

    assert results == ["chunk 0", "chunk 1"]


def test_get_document_chunk_texts_for_concept_returns_empty_when_no_match() -> None:
    document_id = uuid4()
    chunk = _chunk(document_id, 0)
    index = _index([chunk], {chunk.id: [uuid4()]})

    assert get_document_chunk_texts_for_concept(index, document_id, uuid4()) == []
