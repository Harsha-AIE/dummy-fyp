from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from edukg.api.schemas import (
    BulkActionResponse,
    BulkApproveRequest,
    BulkRejectRequest,
    PaginatedTriplesResponse,
    PendingExtensionOut,
    PendingTripleOut,
    ReviewActionResponse,
)
from edukg.api.state import AppResources, get_resources
from edukg.core.models import OntologyExtension, Triple

router = APIRouter(prefix="/api/review", tags=["review"])


def _find_pending_extension(extension_id: UUID, resources: AppResources) -> OntologyExtension:
    for extension in resources.ontology_repo.list_pending_extensions():
        if extension.id == extension_id:
            return extension
    raise HTTPException(status_code=404, detail=f"No pending ontology extension {extension_id}")


@router.get("/ontology-extensions", response_model=list[PendingExtensionOut])
def list_pending_extensions(resources: AppResources = Depends(get_resources)) -> list[PendingExtensionOut]:
    return [
        PendingExtensionOut(
            id=e.id,
            kind=e.kind.value,
            subject=e.subject.value,
            proposed_name=e.proposed_name,
            proposed_description=e.proposed_description,
            justification=e.justification,
            confidence=e.confidence.value,
        )
        for e in resources.ontology_repo.list_pending_extensions()
    ]


@router.post("/ontology-extensions/{extension_id}/approve", response_model=ReviewActionResponse)
def approve_extension(extension_id: UUID, resources: AppResources = Depends(get_resources)) -> ReviewActionResponse:
    extension = _find_pending_extension(extension_id, resources)
    active_ontology = resources.ontology_manager.get_active_ontology(extension.subject)
    resources.ontology_review_service.approve(active_ontology=active_ontology, extension=extension, reviewer_id="review-ui")
    return ReviewActionResponse(id=extension_id, status="approved")


@router.post("/ontology-extensions/{extension_id}/reject", response_model=ReviewActionResponse)
def reject_extension(extension_id: UUID, resources: AppResources = Depends(get_resources)) -> ReviewActionResponse:
    extension = _find_pending_extension(extension_id, resources)
    resources.ontology_review_service.reject(extension, reviewer_id="review-ui")
    return ReviewActionResponse(id=extension_id, status="rejected")


_MAX_PAGE_SIZE = 200


@router.get("/triples", response_model=PaginatedTriplesResponse)
def list_pending_triples(
    page: int = 1, page_size: int = 50, resources: AppResources = Depends(get_resources)
) -> PaginatedTriplesResponse:
    """Paginated — this queue can legitimately hold thousands of triples
    (most of the low-confidence co-occurrence-fallback tier's output), and
    rendering/fetching all of them at once made the Review Console
    unusably slow (a multi-thousand-row table with no pagination, plus two
    entity_repo.get() filesystem reads per triple regardless of whether it
    was ever shown). Sorted by confidence descending so the
    quickest-to-review (closest to already being good) items surface
    first."""
    if page < 1:
        raise HTTPException(status_code=400, detail="page must be >= 1")
    if not 1 <= page_size <= _MAX_PAGE_SIZE:
        raise HTTPException(status_code=400, detail=f"page_size must be between 1 and {_MAX_PAGE_SIZE}")

    all_pending = sorted(resources.pending_triple_store.list_pending(), key=lambda t: t.confidence.value, reverse=True)
    total = len(all_pending)
    start = (page - 1) * page_size
    page_items = all_pending[start : start + page_size]

    out: list[PendingTripleOut] = []
    for triple in page_items:
        subject = resources.entity_repo.get(triple.subject_entity_id)
        obj = resources.entity_repo.get(triple.object_entity_id)
        out.append(
            PendingTripleOut(
                id=triple.id,
                subject_entity_id=triple.subject_entity_id,
                subject_name=subject.canonical_name if subject else "(unknown entity)",
                predicate=triple.predicate,
                object_entity_id=triple.object_entity_id,
                object_name=obj.canonical_name if obj else "(unknown entity)",
                confidence=triple.confidence.value,
                validation_status=triple.validation_status.value,
            )
        )
    return PaginatedTriplesResponse(items=out, total=total, page=page, page_size=page_size)


def _approve_triple(triple: Triple, resources: AppResources) -> bool:
    """Writes one pending triple into the graph and removes it from the
    pending store. Returns False (does nothing else) if its subject/object
    entity no longer exists. Shared by the single-triple and bulk-approve
    endpoints — the single endpoint turns a False into an HTTP 409; bulk
    approve just skips that one triple and keeps going rather than aborting
    the whole batch over one stale entity."""
    subject = resources.entity_repo.get(triple.subject_entity_id)
    obj = resources.entity_repo.get(triple.object_entity_id)
    if subject is None or obj is None:
        return False
    resources.graph_repo.upsert_entity(subject)
    resources.graph_repo.upsert_entity(obj)
    resources.graph_repo.upsert_triple(triple, subject, obj)
    resources.pending_triple_store.remove(triple.id)
    return True


@router.post("/triples/{triple_id}/approve", response_model=ReviewActionResponse)
def approve_triple(triple_id: UUID, resources: AppResources = Depends(get_resources)) -> ReviewActionResponse:
    triple = resources.pending_triple_store.get(triple_id)
    if triple is None:
        raise HTTPException(status_code=404, detail=f"No pending triple {triple_id}")
    if not _approve_triple(triple, resources):
        raise HTTPException(status_code=409, detail="Triple's subject/object entity no longer exists")
    return ReviewActionResponse(id=triple_id, status="approved")


@router.post("/triples/{triple_id}/reject", response_model=ReviewActionResponse)
def reject_triple(triple_id: UUID, resources: AppResources = Depends(get_resources)) -> ReviewActionResponse:
    if resources.pending_triple_store.get(triple_id) is None:
        raise HTTPException(status_code=404, detail=f"No pending triple {triple_id}")
    resources.pending_triple_store.remove(triple_id)
    return ReviewActionResponse(id=triple_id, status="rejected")


@router.post("/triples/bulk-approve", response_model=BulkActionResponse)
def bulk_approve_triples(
    payload: BulkApproveRequest, resources: AppResources = Depends(get_resources)
) -> BulkActionResponse:
    """The pending-triples queue can legitimately hold thousands of rows
    (most of the low-confidence co-occurrence-fallback tier's output) —
    reviewing one at a time doesn't scale. Approves every pending triple at
    or above the given confidence in one call."""
    matching = [t for t in resources.pending_triple_store.list_pending() if t.confidence.value >= payload.min_confidence]
    approved_count = sum(1 for t in matching if _approve_triple(t, resources))
    return BulkActionResponse(count=approved_count)


@router.post("/triples/bulk-reject", response_model=BulkActionResponse)
def bulk_reject_triples(
    payload: BulkRejectRequest, resources: AppResources = Depends(get_resources)
) -> BulkActionResponse:
    matching = [t for t in resources.pending_triple_store.list_pending() if t.confidence.value <= payload.max_confidence]
    for triple in matching:
        resources.pending_triple_store.remove(triple.id)
    return BulkActionResponse(count=len(matching))
