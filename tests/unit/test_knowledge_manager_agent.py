from dataclasses import dataclass
from uuid import UUID, uuid4

from edukg.agents.knowledge_manager.content_indexer import ContentIndex
from edukg.agents.knowledge_manager.graph_reranker import GraphDistanceReranker
from edukg.agents.knowledge_manager.knowledge_manager_agent import KnowledgeManagerAgent
from edukg.core.models import Chunk
from edukg.core.models.enums import ExtractionMethod


class _FakeDenseIndex:
    def __init__(self, hits: list[tuple[UUID, float]]) -> None:
        self._hits = hits

    def search(self, embedding, top_k):
        return self._hits[:top_k]


class _FakeLexicalIndex:
    def __init__(self, hits: list[tuple[UUID, float]]) -> None:
        self._hits = hits

    def search(self, query, top_k):
        return self._hits[:top_k]


class _FakeEmbeddingProvider:
    def embed(self, texts):
        return [[0.0] for _ in texts]

    def dimension(self):
        return 1


@dataclass
class _FakeEntity:
    id: UUID


class _FakeConceptTagger:
    def __init__(self, entities: list[_FakeEntity]) -> None:
        self._entities = entities

    def tag(self, text):
        return self._entities


class _FakeGraphRepository:
    def __init__(self, distances: dict[tuple, int | None]) -> None:
        self._distances = distances

    def shortest_path_length(self, from_entity_id, to_entity_id, max_hops):
        return self._distances.get((from_entity_id, to_entity_id))

    def shortest_path_lengths_bulk(self, from_ids, to_ids, max_hops):
        result = {}
        for f in from_ids:
            for t in to_ids:
                d = self._distances.get((f, t))
                if d is not None:
                    result[(f, t)] = d
        return result


def _chunk(text: str) -> Chunk:
    return Chunk(
        document_id=uuid4(), sequence_index=0, text=text, char_start=0, char_end=len(text),
        extraction_method=ExtractionMethod.NATIVE_TEXT, token_count=5,
    )


def test_retrieve_reranks_by_graph_distance_to_query_concepts() -> None:
    chunk_a, chunk_b = _chunk("a"), _chunk("b")
    query_concept, concept_a, concept_b = uuid4(), uuid4(), uuid4()

    index = ContentIndex(
        chunks_by_id={chunk_a.id: chunk_a, chunk_b.id: chunk_b},
        dense_index=_FakeDenseIndex([(chunk_a.id, 0.9), (chunk_b.id, 0.8)]),
        lexical_index=_FakeLexicalIndex([(chunk_b.id, 5.0), (chunk_a.id, 4.0)]),
        chunk_concepts={chunk_a.id: [concept_a], chunk_b.id: [concept_b]},
    )
    # chunk_b's concept is adjacent to the query concept; chunk_a's is far away.
    graph = _FakeGraphRepository({(query_concept, concept_a): 5, (query_concept, concept_b): 0})
    reranker = GraphDistanceReranker(graph, max_hops=10)

    agent = KnowledgeManagerAgent(
        index, _FakeEmbeddingProvider(), _FakeConceptTagger([_FakeEntity(query_concept)]), reranker, alpha=0.3, beta=0.7
    )

    results = agent.retrieve("some query", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == chunk_b.id
    assert results[0].graph_score == 1.0
    assert results[1].chunk.id == chunk_a.id


def test_retrieve_ignores_an_overly_common_concept_for_reranking() -> None:
    # Regression test: a generic word ("selected", "first", a bare number)
    # extracted as a "concept" in many unrelated chunks was giving those
    # chunks a false perfect graph_score=1.0 whenever a query happened to
    # also tag that same overly-common concept — real content relevance had
    # nothing to do with it. A concept referenced by an unusually large
    # fraction of the corpus's chunks must be excluded from re-ranking.
    generic_concept, specific_concept = uuid4(), uuid4()
    target_chunk = _chunk("target")
    # 4 unrelated chunks all happen to share the generic concept (well above
    # the max(3, 15%-of-10-chunks) = 3 threshold for a 10-chunk corpus).
    unrelated_chunks = [_chunk(f"unrelated-{i}") for i in range(4)]
    filler_chunks = [_chunk(f"filler-{i}") for i in range(5)]  # pad corpus size to 10 total

    chunks_by_id = {target_chunk.id: target_chunk, **{c.id: c for c in unrelated_chunks + filler_chunks}}
    chunk_concepts = {c.id: [generic_concept] for c in unrelated_chunks}
    chunk_concepts[target_chunk.id] = [specific_concept]

    index = ContentIndex(
        chunks_by_id=chunks_by_id,
        dense_index=_FakeDenseIndex([(target_chunk.id, 0.9), *[(c.id, 0.5) for c in unrelated_chunks]]),
        lexical_index=_FakeLexicalIndex([(target_chunk.id, 5.0), *[(c.id, 1.0) for c in unrelated_chunks]]),
        chunk_concepts=chunk_concepts,
    )
    # The query tags both generic_concept and specific_concept directly (the
    # ConceptTagger stub below returns both), each an identical-id (distance
    # 0) match to itself — simulating the query mentioning both a real topic
    # and a generic filler word that happens to match the same entity id an
    # unrelated chunk's concept list also contains.
    graph = _FakeGraphRepository({(generic_concept, generic_concept): 0, (specific_concept, specific_concept): 0})
    reranker = GraphDistanceReranker(graph, max_hops=4)

    agent = KnowledgeManagerAgent(
        index,
        _FakeEmbeddingProvider(),
        _FakeConceptTagger([_FakeEntity(generic_concept), _FakeEntity(specific_concept)]),
        reranker,
        alpha=0.3,
        beta=0.7,
    )

    results = agent.retrieve("query mentioning both a real topic and a filler word", top_k=5)

    unrelated_result = next(r for r in results if r.chunk.id == unrelated_chunks[0].id)
    target_result = next(r for r in results if r.chunk.id == target_chunk.id)
    assert unrelated_result.graph_score == 0.0  # generic_concept excluded, no false match
    assert target_result.graph_score == 1.0  # specific_concept still works normally


def test_retrieve_on_empty_index_returns_empty() -> None:
    index = ContentIndex(chunks_by_id={}, dense_index=_FakeDenseIndex([]), lexical_index=_FakeLexicalIndex([]))
    agent = KnowledgeManagerAgent(
        index, _FakeEmbeddingProvider(), _FakeConceptTagger([]), GraphDistanceReranker(_FakeGraphRepository({}))
    )

    assert agent.retrieve("anything") == []


def test_retrieve_blank_query_returns_empty() -> None:
    chunk = _chunk("some content")
    index = ContentIndex(
        chunks_by_id={chunk.id: chunk},
        dense_index=_FakeDenseIndex([(chunk.id, 0.9)]),
        lexical_index=_FakeLexicalIndex([(chunk.id, 1.0)]),
    )
    agent = KnowledgeManagerAgent(
        index, _FakeEmbeddingProvider(), _FakeConceptTagger([]), GraphDistanceReranker(_FakeGraphRepository({}))
    )

    assert agent.retrieve("   ") == []


def test_retrieve_blends_mastery_gap_when_student_id_and_provider_given() -> None:
    chunk_weak, chunk_strong = _chunk("weak"), _chunk("strong")
    concept_weak, concept_strong = uuid4(), uuid4()

    index = ContentIndex(
        chunks_by_id={chunk_weak.id: chunk_weak, chunk_strong.id: chunk_strong},
        # Equal dense/lexical ranks so, without the mastery term, both chunks tie.
        dense_index=_FakeDenseIndex([(chunk_weak.id, 0.9), (chunk_strong.id, 0.9)]),
        lexical_index=_FakeLexicalIndex([(chunk_weak.id, 1.0), (chunk_strong.id, 1.0)]),
        chunk_concepts={chunk_weak.id: [concept_weak], chunk_strong.id: [concept_strong]},
    )
    mastery_by_concept = {concept_weak: 0.1, concept_strong: 0.9}
    agent = KnowledgeManagerAgent(
        index,
        _FakeEmbeddingProvider(),
        _FakeConceptTagger([]),
        GraphDistanceReranker(_FakeGraphRepository({})),
        gamma=0.5,
        mastery_provider=lambda student_id: mastery_by_concept,
    )

    results = agent.retrieve("query", top_k=2, student_id="student-1")

    assert results[0].chunk.id == chunk_weak.id  # bigger mastery gap ranks first


def test_retrieve_degrades_gracefully_when_mastery_provider_raises() -> None:
    # Regression test: found live — a Postgres hiccup inside the mastery
    # provider (Knowledge Tracing's store) used to propagate all the way out
    # of retrieve() uncaught, failing the entire /ask request even though
    # dense+lexical search, fusion, and graph re-ranking above it had all
    # already succeeded. The mastery-gap term is optional (gamma is inert
    # without it) — a failure here must degrade to "no mastery boost", not
    # lose the whole retrieval result.
    chunk = _chunk("content")
    concept = uuid4()
    index = ContentIndex(
        chunks_by_id={chunk.id: chunk},
        dense_index=_FakeDenseIndex([(chunk.id, 0.9)]),
        lexical_index=_FakeLexicalIndex([(chunk.id, 1.0)]),
        chunk_concepts={chunk.id: [concept]},
    )

    def _raising_provider(student_id: str) -> dict:
        raise RuntimeError("Postgres unreachable")

    agent = KnowledgeManagerAgent(
        index,
        _FakeEmbeddingProvider(),
        _FakeConceptTagger([]),
        GraphDistanceReranker(_FakeGraphRepository({})),
        gamma=0.5,
        mastery_provider=_raising_provider,
    )

    results = agent.retrieve("query", top_k=1, student_id="student-1")

    assert len(results) == 1
    assert results[0].chunk.id == chunk.id


def test_retrieve_ignores_mastery_provider_without_student_id() -> None:
    chunk = _chunk("content")
    concept = uuid4()
    index = ContentIndex(
        chunks_by_id={chunk.id: chunk},
        dense_index=_FakeDenseIndex([(chunk.id, 0.9)]),
        lexical_index=_FakeLexicalIndex([(chunk.id, 1.0)]),
        chunk_concepts={chunk.id: [concept]},
    )
    provider_calls = []
    agent = KnowledgeManagerAgent(
        index,
        _FakeEmbeddingProvider(),
        _FakeConceptTagger([]),
        GraphDistanceReranker(_FakeGraphRepository({})),
        gamma=0.5,
        mastery_provider=lambda student_id: provider_calls.append(student_id) or {},
    )

    agent.retrieve("query", top_k=1)

    assert provider_calls == []
