from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from edukg.api.schemas import GraphEdgeOut, GraphNodeOut, GraphResponse
from edukg.api.state import AppResources, get_resources
from edukg.modules.content.document_concepts import get_document_top_concepts

router = APIRouter(prefix="/api/graph", tags=["graph"])

_DEFAULT_MIND_MAP_CONCEPT_LIMIT = 12


@router.get("", response_model=GraphResponse)
def get_graph(resources: AppResources = Depends(get_resources)) -> GraphResponse:
    nodes, edges = resources.graph_repo.export_graph()
    return GraphResponse(
        nodes=[GraphNodeOut(**n) for n in nodes],
        edges=[GraphEdgeOut(**e) for e in edges],
    )


@router.get("/document/{document_id}", response_model=GraphResponse)
def get_document_mind_map(document_id: UUID, resources: AppResources = Depends(get_resources)) -> GraphResponse:
    """Mind Map (Tier 4): this document's top concepts plus the edges
    directly between them, rooted at a synthetic node representing the
    document itself — gives the radial layout a stable center to grow from
    regardless of how the real KG edges happen to connect."""
    document = resources.document_repo.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"No document {document_id}")

    top_concepts = get_document_top_concepts(
        resources.content_index, resources.entity_repo, document_id, limit=_DEFAULT_MIND_MAP_CONCEPT_LIMIT
    )
    concept_ids = [cid for cid, _ in top_concepts]
    nodes, edges = resources.graph_repo.get_subgraph(concept_ids)

    root_id = f"doc:{document_id}"
    root_node = GraphNodeOut(id=root_id, name=document.file_path.name, type="document")
    root_edges = [
        GraphEdgeOut(source=root_id, target=str(cid), predicate="covers", confidence=1.0, evidence_count=0)
        for cid, _ in top_concepts
    ]

    return GraphResponse(
        nodes=[root_node] + [GraphNodeOut(**n) for n in nodes],
        edges=root_edges + [GraphEdgeOut(**e) for e in edges],
    )
