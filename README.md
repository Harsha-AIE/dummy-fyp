# EduKG — Offline Educational Knowledge Graph System

## Module 1: Project Bootstrap & Core Foundation — DELIVERED

### What's implemented
- `src/edukg/core/config` — layered `pydantic-settings` config (paths, OpenAI, Neo4j, embeddings, OCR, chunking, thresholds, cache, logging). Nothing hardcoded; all overridable via `.env` / env vars.
- `src/edukg/core/logging_setup.py` — `structlog`-based structured (JSON) logging to console + file.
- `src/edukg/core/models/` — strongly typed Pydantic v2 domain models: `Document`, `Chunk`, `Mention`, `Entity`, `Relation`, `Triple`, `Provenance`, `ConfidenceScore`, `OntologyClass`, `OntologyRelation`, `OntologyExtension`, `ValidationResult`, `ReviewDecision`. No raw dicts cross module boundaries.
- `src/edukg/core/interfaces/` — abstract contracts (`DocumentRepository`, `OntologyRepository`, `EntityRepository`, `GraphRepository`, `VectorIndexRepository`, `CacheRepository`, `LLMProvider`, `OCRProvider`, `EmbeddingProvider`, `PipelineStageHandler`). All future modules implement these; business logic never imports Neo4j/OpenAI/FAISS directly.
- `src/edukg/core/exceptions/` — full exception hierarchy, one type per pipeline stage.
- `src/edukg/pipeline/` — `PipelineContext` (typed carrier), `PipelineOrchestrator` (enforces the fixed 12-stage order, no bypass), `CheckpointStore` (JSON checkpoint after every stage — resumable, and exportable/downloadable at any point).
- `tests/unit/` — 12 passing tests (models + orchestrator + checkpoint round-trip).
- `scripts/verify_bootstrap.py` — end-to-end smoke test proving the wiring works.

### Install (Windows PowerShell)
```powershell
.\INSTALL_WINDOWS.ps1
```

### Install (this sandbox / Linux)
```bash
pip install -r requirements.txt --break-system-packages
pip install -e . --break-system-packages
```

### Verify
```bash
pytest tests/unit -q          # 12 passed
python scripts/verify_bootstrap.py
```

### Assumptions made (flag if wrong)
1. Python 3.11+, `src/` layout, editable install via `pyproject.toml`.
2. `PipelineStage` enum + `STAGE_ORDER` tuple is the single source of truth for the mandatory pipeline order from the spec.
3. Checkpointing writes one JSON file per document under `data/cache/checkpoints/<document_id>.checkpoint.json` after every stage — this is the "download what's been generated so far" mechanism; on any interruption the last checkpoint file is a complete, valid, re-loadable `PipelineContext`.
4. Core vs. domain ontology split, and Subject enum values, follow the examples given (Mathematics, CS, ML, DL, RL, Physics, Chemistry, Biology, English) — extendable without code changes to any consumer since everything keys off the `Subject` enum + `OntologyRepository`.

### Next
**Module 2: Document Ingestion** (`modules/ingestion`) — PDF/DOCX/OCR extraction, layout-aware parsing, hashing/caching, provenance — implementing `DocumentRepository` + a `DocumentIngestionService` behind `PipelineStageHandler`.

Reply "continue" / "next module" to proceed, or flag any change needed to Module 1 first.

---

## Module 2: Ingestion, Subject Detection, Ontology — DELIVERED

See `MODULE_2_NOTES.md` for full details, assumptions, and the critical fix applied to the filesystem ontology repository.

---

## Modules 3-5: Entity Extraction/Linking, Relation Extraction, Triples, Fusion & Graph Storage — DELIVERED

Completes the pipeline: Entity Extraction (spaCy NER + KeyBERT + YAKE) → Entity
Linking (Azure OpenAI embeddings + FAISS cosine search) → Relation Extraction
(3-tier cascade: dependency parse → cue phrase → co-occurrence) → Triple
Generation → Triple Validation (domain/range checking) → Knowledge Fusion
(dedup + confidence aggregation) → Graph Storage (Neo4j). `scripts/verify_full_pipeline.py`
runs all 12 stages end-to-end against `data/raw/DNA Task 3.docx` (real
coursework content — `data/raw/amazon.pdf` is leftover test fixture output
mislabeled with a `.pdf` extension, not real source material; see
`MODULE_3_5_NOTES.md`).

See `MODULE_3_5_NOTES.md` for full details and flagged scope decisions
(notably: Azure OpenAI in place of plain OpenAI/local embeddings, and a
rule-based domain/range checker in place of full `pyshacl`).

---

## Agent Layer: Knowledge Manager (RAG) + Pedagogical Guide — DELIVERED

The first two agents of the multi-agent tutor architecture, built on top of
the completed KG: `KnowledgeManagerAgent` (hybrid dense+BM25 retrieval,
Reciprocal Rank Fusion, KG graph-distance re-ranking) and
`PedagogicalGuideAgent` (tiered keyword → LLM intent/misconception
classification, KG concept tagging). `scripts/verify_agents.py` demos both
against the live graph. See `MODULE_AGENTS_NOTES.md` for scope decisions and
real bugs found while verifying end-to-end.

---

## API Layer + Web UI — DELIVERED

FastAPI backend (`src/edukg/api/`) exposing the pipeline, agents, graph, and
review queues over HTTP — including real document upload and a document
picker, SSE-streamed live pipeline progress, and an automatic search-index
refresh after every run so newly ingested documents are searchable
immediately, no restart needed. A Next.js/TypeScript/Tailwind frontend
(`frontend/`) gives it a full UI: a landing page, a live agent demo, a
knowledge graph visualizer, a human-in-the-loop review console, and the
pipeline run dashboard. See `MODULE_API_UI_NOTES.md` for scope decisions and
real bugs found while verifying end-to-end — including the missing-upload
gap and the stale-search-index gap this pass fixed.

---

## Knowledge Tracing (BKT + DKT) — DELIVERED

The third tutor component: `KnowledgeTracingAgent` turns every classified
dialogue turn into a per-concept mastery vector, fusing a Bayesian Knowledge
Tracing (BKT) estimate — available from the first event, no training needed
— with a Deep Knowledge Tracing (DKT) estimate from a small, real,
self-attentive PyTorch model (`agents/knowledge_tracing/dkt_model.py`,
trained via `scripts/train_dkt.py`, CUDA-accelerated when available), and
produces a natural-language explanation citing the events behind the score
— DKT's attention weights, when a trained model is available, or BKT's
most-recent evidence otherwise. Mastery now also feeds back into the
Knowledge Manager agent's re-ranking (`gamma*mastery_gap_weight`), filling
the seam that agent was built with in the previous pass. Surfaced in the
frontend as a live "Student Mastery" card on the demo screen and a
dedicated `/mastery` vector view.

Since nothing in the system produces a real answer-correctness signal yet
(no quiz/assessment layer), this pass derives one from the Pedagogical
Guide's already-classified intent — a documented heuristic, not a hidden
shortcut. See `MODULE_KT_NOTES.md` for the exact heuristic table, BKT
parameters, the DKT architecture, the synthetic-bootstrap cold-start
approach, and what's explicitly still out of scope.
