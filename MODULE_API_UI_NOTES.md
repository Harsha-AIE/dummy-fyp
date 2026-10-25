# API Layer + Web UI — FastAPI backend, Next.js frontend

Builds on the completed KG pipeline and agent layer. Adds the HTTP layer
that was always the missing piece (`src/edukg/api/` was a confirmed-empty
stub) and a full Next.js/TypeScript/Tailwind frontend on top of it — a live
agent demo, a knowledge graph visualizer, a human-in-the-loop review
console, and a pipeline run dashboard with document upload.

## Backend (`src/edukg/api/`)

Runs as `uvicorn edukg.api.main:app`. Expensive resources (spaCy, KeyBERT,
Azure providers, the built content index, both agents) are constructed
**once** at process startup via FastAPI's `lifespan`, stored on
`AppResources` (`state.py`), and read off `app.state` by every route via
`Depends(get_resources)` — never rebuilt per-request.

- `routers/agents.py` — wraps `KnowledgeManagerAgent`/`PedagogicalGuideAgent` directly; `/api/agents/ask` calls both.
- `routers/graph.py` — full node/edge export from Neo4j (`Neo4jGraphRepository.export_graph()`, new).
- `routers/review.py` — approve/reject for pending ontology extensions (reuses `OntologyReviewService` as-is) and pending triples (new logic: approve writes straight to Neo4j via `upsert_entity`/`upsert_triple`, then removes from the pending store).
- `routers/pipeline.py` + `pipeline_runner.py` — document upload, an in-memory `PipelineRunRegistry` (single-process, by design — see scope decisions), and Server-Sent Events streaming of live stage progress, via a new optional `on_stage_event` callback on `PipelineOrchestrator.run()`.
- `routers/dashboard.py` — summary counts for the landing page and dashboard.

## Frontend (`frontend/`)

Next.js 15 (App Router) + TypeScript + Tailwind CSS v4 + shadcn/ui +
TanStack Query + `react-force-graph-2d`. A `(app)` route group gives the
five console screens a shared sidebar layout while the landing page at `/`
stays full-width with its own nav.

## Scope decisions

1. **`PipelineRunRegistry` is in-memory, single-process.** A run started on one API process isn't visible to another, and restarting the API loses run history. Fine for a review/demo deployment; a real deployment would move this to Redis/a DB.
2. **SSE, not WebSockets**, for pipeline progress — simpler (`StreamingResponse` + the browser's native `EventSource`), no extra client library, and pipeline progress is one-directional (server → browser), which is exactly what SSE is for.
3. **The retrieval index (dense + BM25) is rebuilt from scratch on every refresh**, not incrementally updated. Cheap enough at current scale (a handful of documents); would need incremental updates if the corpus grew large.

## Found and fixed while verifying end-to-end

- **No document upload existed.** The pipeline dashboard could only ever run documents someone had already placed by hand in `data/raw/` on the server's filesystem — there was no way for someone using the actual UI to bring their own document. This was the headline gap: added `POST /api/pipeline/documents` (multipart upload, extension-validated, filename sanitized against path traversal via `Path(...).name`) plus a real drag-and-drop upload zone on the Pipeline Runs page. Verified end-to-end through the actual browser: upload → auto-select → live 12-stage run → completion.
- **The Knowledge Manager agent's retrieval index didn't refresh after a pipeline run.** A document uploaded and processed through the dashboard would silently not be searchable from the Live Agent Demo until the API process restarted — a real "not actually real-time" gap. Fixed with `AppResources.refresh_content_index()`, called automatically after every successful pipeline run; a failed refresh logs a warning but does not mark the ingestion run itself as failed (the data is correctly in Neo4j either way — only the in-memory search index would be stale until the next refresh or restart). Verified live: the backend log shows `content_index_refreshed` firing immediately after `graph_storage` completes, with no restart in between.
- **`data/raw` was hardcoded as a relative path** in two places (the upload endpoint and the pipeline runner), silently ignoring `EDUKG_PATH_RAW_DIR` and making the upload endpoint untestable without writing into the real project's data directory. Fixed by routing both through the existing `PathSettings.raw_dir` (`get_settings().paths.raw_dir`), with the route handler taking it as a `Depends()`-injected parameter so tests can override it with a `tmp_path`.
- **Headings were silently rendering in the browser's fallback serif**, not the intended sans-serif, during the visual redesign — traced to `--font-sans` in `@theme inline` being self-referential (`next/font` actually sets `--font-geist-sans`, never `--font-sans`). Fixed by explicitly defining the `--font-sans` chain in `:root`.

## Verify

```bash
pip install -r requirements.txt
PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m uvicorn edukg.api.main:app --port 8000
cd frontend && npm install && npm run dev
```

```bash
pytest tests/unit -q   # includes routers/review.py and routers/pipeline.py tests, no real Azure/Neo4j calls
```

Then open http://localhost:3000 — upload a `.pdf`/`.docx` on the Pipeline
Runs page, watch it complete live, and confirm it's immediately answerable
from the Live Agent Demo without restarting anything.

## Next

Response Generation + an orchestrator tying the two agents together is
still the natural next slice (unchanged from `MODULE_AGENTS_NOTES.md`).
