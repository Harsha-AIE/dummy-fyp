# Module 2 — Document Ingestion, Subject Detection, Ontology Management

Builds strictly on Module 1 (frozen, untouched except for the two additive fixes noted below).

## Files created

**infrastructure/ (new implementations of Module 1 interfaces)**
- `filesystem/document_repository.py` — `FilesystemDocumentRepository(DocumentRepository)`
- `filesystem/ontology_repository.py` — `FilesystemOntologyRepository(OntologyRepository)`
- `cache/filesystem_cache.py` — `FilesystemCacheRepository(CacheRepository)`
- `ocr/paddleocr_provider.py` — `PaddleOCRProvider(OCRProvider)`
- `openai/openai_llm_provider.py` — `OpenAILLMProvider(LLMProvider)`

**modules/ingestion/**
- `hashing.py`, `text_cleaner.py`, `chunker.py`, `pdf_processor.py`, `pdf_rasterizer.py`,
  `layout_assembly.py`, `docx_processor.py`, `metadata_extractor.py`
- `ingestion_service.py` — `DocumentIngestionService`, the `INGESTION` `PipelineStageHandler`

**modules/subject_detection/**
- `strategy.py` — `SubjectDetectionStrategy` ABC (swappable)
- `subject_keywords.py`, `keyword_strategy.py` — default offline `KeywordSubjectDetectionStrategy`
- `subject_detection_service.py` — `SubjectDetectionStageHandler`

**modules/ontology/**
- `active_ontology.py` — `ActiveOntology` (module-local runtime view, dataclass style like `PipelineContext`)
- `core_ontology_loader.py`, `domain_ontology_loader.py`
- `ontology_manager.py` — `OntologyManager` (load, merge, version, single clean API)
- `ontology_extension_service.py` — LLM-backed proposal generation (never mutates ontology)
- `ontology_validator.py` — `OntologyValidationService`
- `ontology_merge_service.py` — `OntologyMergeService` (only path that writes class/relation/alias)
- `ontology_review_service.py` — `OntologyReviewService` (approve/reject/modify/merge + auto-review)
- `stage_handlers.py` — `OntologyLoadStageHandler`, `OntologyExtensionStageHandler`, `OntologyValidationStageHandler`

**data/ontologies/** — seeded `core.json` + `domains/{computer_science,machine_learning,mathematics,english}.json`
(generated via the real Pydantic models, not hand-written, so format is guaranteed correct).

**tests/unit/** — `test_ingestion.py`, `test_ingestion_service.py`, `test_subject_detection.py`,
`test_ontology.py`, `test_ontology_extension_pipeline.py` — 34 new tests, all passing (46/46 total with Module 1).

**scripts/verify_module2.py** — runs the *real* `PipelineOrchestrator` through
`INGESTION → SUBJECT_DETECTION → ONTOLOGY_LOAD → ONTOLOGY_EXTENSION → ONTOLOGY_VALIDATION`
with a stub `LLMProvider`/`OCRProvider` (no API key required), proving the handlers compose
correctly under the orchestrator — not just in isolation.

## One issue flagged and fixed (not a Module 1 change)

`OntologyRelation` (Module 1, frozen) has no `is_core` flag the way `OntologyClass` does, so
there was no principled place to persist a *core* relation (e.g. `is_prerequisite_of`) versus
a domain relation. Rather than touch the frozen model, `FilesystemOntologyRepository.add_relation`
adopts a convention — `subject == Subject.UNKNOWN` means "core relation," written to `core.json`
— consistent with how the seed data is generated. This is additive/workaround, not a redesign,
and is isolated entirely inside the new repository implementation. Flagging for your confirmation;
if a different convention is preferred, only this one method needs to change.

## Design decisions (stated, not silently assumed)

1. **No subject-detection methodology was specified** in the attached documents, so a
   dependency-free, offline **keyword-frequency strategy** is the default
   `SubjectDetectionStrategy`. It's fully swappable (see `strategy.py`) — an
   embedding/classifier-based strategy can be dropped in later with zero changes to the
   stage handler or pipeline.
2. **Ontology versioning** is implemented via `CacheRepository` (already a Module 1
   interface) rather than adding new methods to `OntologyRepository` — `OntologyManager`
   tracks a per-subject revision counter (`ontology_version:<subject>`) and formats it as
   `1.<revision>.0`. No interface changes required.
3. **`ONTOLOGY_VALIDATION` stage auto-reviews.** The mandatory pipeline order places
   `ONTOLOGY_VALIDATION` before `ENTITY_EXTRACTION`, but the spec also says ontology
   extension review is a Teacher/Student UI concern with no UI in this module. Resolution:
   `OntologyReviewService.auto_review()` acts as a `ReviewerRole.SYSTEM` reviewer for
   offline/batch runs — only extensions that are `ValidationStatus.VALID` (not just
   `WARNING`) **and** clear `thresholds.ontology_extension_confidence_min` are merged;
   everything else stays pending in the repository for a human reviewer via the same
   `OntologyReviewService.approve/reject/modify/merge_with` API a future UI will call.
   Nothing bypasses validation or the review/decision audit trail — this only supplies
   the reviewer for the case where no human is available yet.
4. **Ontology extension proposals are drawn from raw ingested chunk text**, not from
   extracted entities (which don't exist yet at this pipeline position) — matching both
   the mandatory stage order and the spec's instruction to "send extracted document
   knowledge to the OpenAI API."

## Verify

```bash
pip install -r requirements.txt --break-system-packages   # adds pypdf, python-docx, openai, paddleocr, etc.
pytest tests/unit -q                                       # 46 passed
python scripts/verify_module2.py                           # real orchestrator, stub LLM/OCR
```

## Next

**Module 3: Entity Extraction + Entity Linking** — NER/keyword/concept extraction against the
now-active ontology, FAISS-backed embedding similarity search, alias handling, canonical
resolution — implementing `EntityRepository` + `VectorIndexRepository`.

Reply "continue" to proceed, or flag anything in Module 2 first.
