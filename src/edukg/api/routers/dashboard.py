from __future__ import annotations

from fastapi import APIRouter, Depends

from edukg.api.schemas import DashboardSummaryResponse
from edukg.api.state import AppResources, get_resources

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummaryResponse)
def summary(resources: AppResources = Depends(get_resources)) -> DashboardSummaryResponse:
    nodes, edges = resources.graph_repo.export_graph()
    return DashboardSummaryResponse(
        entity_count=len(nodes),
        accepted_triple_count=len(edges),
        pending_triple_count=len(resources.pending_triple_store.list_pending()),
        pending_extension_count=len(resources.ontology_repo.list_pending_extensions()),
    )
