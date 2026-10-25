"""Router functions called directly, same convention as test_api_kt.py /
test_api_quiz.py — no FastAPI TestClient, a lightweight fake resources
object standing in for AppResources."""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from edukg.agents.knowledge_manager.content_indexer import ContentIndex
from edukg.api.routers import graph
from edukg.core.models import Chunk
from edukg.core.models.enums import ExtractionMethod


class _FakeDocument:
    def __init__(self, file_name: str) -> None:
        self.file_path = Path(file_name)


class _FakeDocumentRepo:
    def __init__(self, documents: dict[UUID, _FakeDocument] | None = None) -> None:
        self._documents = documents or {}

    def get(self, document_id: UUID):
        return self._documents.get(document_id)


class _FakeEntity:
    def __init__(self, entity_id: UUID, canonical_name: str) -> None:
        self.id = entity_id
        self.canonical_name = canonical_name


class _FakeEntityRepo:
    def __init__(self, entities: dict[UUID, _FakeEntity] | None = None) -> None:
        self._entities = entities or {}

    def get(self, entity_id: UUID):
        return self._entities.get(entity_id)


class _FakeGraphRepo:
    def __init__(self, nodes=None, edges=None) -> None:
        self.nodes = nodes or []
        self.edges = edges or []
        self.last_subgraph_call: list[UUID] | None = None

    def export_graph(self):
        return self.nodes, self.edges

    def get_subgraph(self, concept_ids: list[UUID]):
        self.last_subgraph_call = concept_ids
        wanted = {str(c) for c in concept_ids}
        nodes = [n for n in self.nodes if n["id"] in wanted]
        edges = [e for e in self.edges if e["source"] in wanted and e["target"] in wanted]
        return nodes, edges


def _chunk(document_id: UUID, sequence_index: int) -> Chunk:
    return Chunk(
        document_id=document_id,
        sequence_index=sequence_index,
        text="some chunk text",
        char_start=0,
        char_end=15,
        extraction_method=ExtractionMethod.NATIVE_TEXT,
        token_count=3,
    )


@dataclass
class _FakeResources:
    document_repo: _FakeDocumentRepo = field(default_factory=_FakeDocumentRepo)
    entity_repo: _FakeEntityRepo = field(default_factory=_FakeEntityRepo)
    graph_repo: _FakeGraphRepo = field(default_factory=_FakeGraphRepo)
    content_index: ContentIndex = field(
        default_factory=lambda: ContentIndex(chunks_by_id={}, dense_index=None, lexical_index=None, chunk_concepts={})
    )


def test_get_graph_returns_full_export() -> None:
    nodes = [{"id": "a", "name": "A", "type": "concept"}]
    edges = [{"source": "a", "target": "b", "predicate": "x", "confidence": 0.5, "evidence_count": 1, "inferred": False}]
    resources = _FakeResources(graph_repo=_FakeGraphRepo(nodes=nodes, edges=[]))

    result = graph.get_graph(resources)

    assert len(result.nodes) == 1
    assert result.nodes[0].name == "A"


def test_document_mind_map_404s_for_unknown_document() -> None:
    with pytest.raises(HTTPException) as exc_info:
        graph.get_document_mind_map(uuid4(), _FakeResources())
    assert exc_info.value.status_code == 404


def test_document_mind_map_includes_synthetic_root_node_and_edges() -> None:
    document_id = uuid4()
    chunk = _chunk(document_id, 0)
    concept = uuid4()
    content_index = ContentIndex(
        chunks_by_id={chunk.id: chunk}, dense_index=None, lexical_index=None, chunk_concepts={chunk.id: [concept]}
    )
    graph_repo = _FakeGraphRepo(nodes=[{"id": str(concept), "name": "Backpropagation", "type": "concept"}], edges=[])
    resources = _FakeResources(
        document_repo=_FakeDocumentRepo({document_id: _FakeDocument("notes.pdf")}),
        entity_repo=_FakeEntityRepo({concept: _FakeEntity(concept, "Backpropagation")}),
        content_index=content_index,
        graph_repo=graph_repo,
    )

    result = graph.get_document_mind_map(document_id, resources)

    root = next(n for n in result.nodes if n.type == "document")
    assert root.name == "notes.pdf"
    assert root.id == f"doc:{document_id}"
    assert any(n.name == "Backpropagation" for n in result.nodes)
    root_edges = [e for e in result.edges if e.source == root.id]
    assert len(root_edges) == 1
    assert root_edges[0].target == str(concept)
    assert graph_repo.last_subgraph_call == [concept]


def test_document_mind_map_scopes_subgraph_to_document_top_concepts_only() -> None:
    document_id = uuid4()
    chunk = _chunk(document_id, 0)
    in_scope = uuid4()
    out_of_scope = uuid4()
    content_index = ContentIndex(
        chunks_by_id={chunk.id: chunk}, dense_index=None, lexical_index=None, chunk_concepts={chunk.id: [in_scope]}
    )
    graph_repo = _FakeGraphRepo(
        nodes=[
            {"id": str(in_scope), "name": "InScope", "type": "concept"},
            {"id": str(out_of_scope), "name": "OutOfScope", "type": "concept"},
        ],
        edges=[],
    )
    resources = _FakeResources(
        document_repo=_FakeDocumentRepo({document_id: _FakeDocument("notes.pdf")}),
        entity_repo=_FakeEntityRepo({in_scope: _FakeEntity(in_scope, "InScope")}),
        content_index=content_index,
        graph_repo=graph_repo,
    )

    result = graph.get_document_mind_map(document_id, resources)

    names = {n.name for n in result.nodes}
    assert "InScope" in names
    assert "OutOfScope" not in names
