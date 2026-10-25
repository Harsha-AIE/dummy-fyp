from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from edukg.api.pipeline_runner import PipelineRun, PipelineRunRegistry, start_pipeline_run
from edukg.api.schemas import (
    AvailableDocumentOut,
    PipelineRunCreated,
    PipelineRunRequest,
    PipelineRunStatus,
    PipelineRunSummary,
    PipelineStageEventOut,
    UploadedDocumentOut,
)
from edukg.api.state import AppResources, get_resources
from edukg.core.config import get_settings
from edukg.core.exceptions import ExternalServiceError

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

_registry = PipelineRunRegistry()
_SUPPORTED_SUFFIXES = {".pdf", ".docx"}
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB — generous for course material, not unbounded


def get_raw_dir() -> Path:
    """Depends()-injected rather than a hardcoded Path("data/raw") — that
    hardcoding was a real bug: it ignored EDUKG_PATH_RAW_DIR entirely and,
    worse, made this endpoint untestable without writing into the real
    project's data/raw directory. Routes take it as a parameter so tests can
    override it with a tmp_path."""
    return get_settings().paths.raw_dir


@router.get("/documents", response_model=list[AvailableDocumentOut])
def list_available_documents(
    resources: AppResources = Depends(get_resources), raw_dir: Path = Depends(get_raw_dir)
) -> list[AvailableDocumentOut]:
    # Maps filename -> document_id (not just a name set) so already-ingested
    # documents can be linked straight to Diagnostic/Mind Map — the Study
    # page (student-facing) needs the id, not just "is this ingested".
    ingested_ids = {doc.file_path.name: doc.id for doc in resources.document_repo.list_all_documents()}
    if not raw_dir.exists():
        return []
    return [
        AvailableDocumentOut(name=p.name, already_ingested=p.name in ingested_ids, document_id=ingested_ids.get(p.name))
        for p in sorted(raw_dir.iterdir())
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    ]


@router.post("/documents", response_model=UploadedDocumentOut)
async def upload_document(
    file: UploadFile,
    resources: AppResources = Depends(get_resources),
    raw_dir: Path = Depends(get_raw_dir),
) -> UploadedDocumentOut:
    """Accepts a browser file upload, drops it into the configured raw-
    documents directory (so it shows up in list_available_documents() and
    can be run through the pipeline — this is the piece that was missing:
    without it, the dashboard could only ever process files someone had
    already placed on the server's disk by hand), and durably persists it to
    Postgres (see PostgresDocumentStore) as the actual record of the upload.
    The local copy alone isn't good enough as "the PDF is stored" — it's
    tied to one server's disk and gone if that disk is wiped."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Path(...).name strips any directory components the client sent
    # (../../etc, absolute paths) — only ever write inside raw_dir.
    safe_name = Path(file.filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}' — expected .pdf or .docx")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit")

    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / safe_name
    destination.write_bytes(contents)

    try:
        await asyncio.to_thread(resources.postgres_store.save, safe_name, contents, file.content_type)
    except ExternalServiceError as exc:
        # The local copy already exists and is processable — the pipeline
        # isn't blocked — but the durability guarantee the user asked for
        # ("store the PDF in Postgres") wasn't met, and that must not be
        # silently swallowed.
        raise HTTPException(status_code=502, detail=f"Saved locally but failed to persist to Postgres: {exc}") from exc

    return UploadedDocumentOut(name=safe_name)


def _to_status(run: PipelineRun) -> PipelineRunStatus:
    return PipelineRunStatus(
        run_id=run.run_id,
        state=run.state,
        events=[PipelineStageEventOut(stage=e.stage, status=e.status, timestamp=e.timestamp) for e in run.events],
        error=run.error,
        summary=PipelineRunSummary(**run.summary) if run.summary else None,
    )


@router.post("/runs", response_model=PipelineRunCreated)
def create_run(payload: PipelineRunRequest, resources: AppResources = Depends(get_resources)) -> PipelineRunCreated:
    run = start_pipeline_run(resources, _registry, payload.path)
    return PipelineRunCreated(run_id=run.run_id)


@router.get("/runs/{run_id}", response_model=PipelineRunStatus)
def get_run(run_id: UUID) -> PipelineRunStatus:
    run = _registry.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No pipeline run {run_id}")
    return _to_status(run)


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: UUID) -> StreamingResponse:
    run = _registry.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No pipeline run {run_id}")

    async def event_source() -> AsyncIterator[str]:
        while True:
            event = await asyncio.to_thread(run.event_queue.get)
            if event is None:
                yield f"event: done\ndata: {json.dumps(_to_status(run).model_dump(mode='json'))}\n\n"
                break
            payload = {"stage": event.stage, "status": event.status, "timestamp": event.timestamp.isoformat()}
            yield f"event: stage\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
