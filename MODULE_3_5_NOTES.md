# Modules 3-5 — Entity Extraction/Linking, Relation Extraction/Triples, Fusion & Graph Storage

Builds strictly on Modules 1-2 (frozen, untouched). Completes the pipeline
spec's remaining 7 stages: Entity Extraction, Entity Linking, Relation
Extraction, Triple Generation, Triple Validation, Knowledge Fusion, Graph
Storage.

## Files created

**infrastructure/ (new implementations of Module 1 interfaces)**
- `openai/azure_openai_llm_provider.py` — `AzureOpenAILLMProvider(LLMProvider)`
- `openai/azure_openai_embedding_provider.py` — `AzureOpenAIEmbeddingProvider(EmbeddingProvider)`
- `faiss/faiss_vector_index.py` — `FAISSVectorIndexRepository(VectorIndexRepository)`
- `filesystem/entity_repository.py` — `FilesystemEntityRepository(EntityRepository)`
- `neo4j/neo4j_graph_repository.py` — `Neo4jGraphRepository(GraphRepository)`

**modules/entity_extraction/**, **modules/entity_linking/**,
**modules/relation_extraction/**, **modules/triple_generation/**,
**modules/triple_validation/**, **modules/fusion/**, **modules/storage/** —
one `PipelineStageHandler` + service per stage (6 through 12), following the
same shape Module 2 established: stage handler wires from `PipelineContext`
and the active ontology, service methods take plain domain objects.

**scripts/verify_full_pipeline.py** — runs the real `PipelineOrchestrator`
through all 12 stages against `data/raw/amazon.pdf` with real providers
(Azure OpenAI, PaddleOCR, FAISS, Neo4j).

**tests/unit/** — `test_entity_extraction.py`, `test_entity_linking.py`,
`test_relation_extraction.py`, `test_triple_generation_validation.py`,
`test_fusion.py`. **tests/integration/test_graph_storage.py** — real Neo4j,
skips automatically if unreachable.

## Assumptions and scope decisions (flagged, not silently made)

1. **Azure OpenAI, not plain OpenAI.** The team supplied an Azure deployment
   (chat: `gpt-4o`, embeddings: `text-embedding-3-large`) instead of a plain
   OpenAI key. `AzureOpenAILLMProvider`/`AzureOpenAIEmbeddingProvider` are
   drop-in `LLMProvider`/`EmbeddingProvider` implementations — no other code
   depends on which provider is wired in. `AzureOpenAISettings` uses the
   `AZURE_OPENAI_` env prefix (not `EDUKG_`) to match the deployment's own
   variable names.
2. **Azure embeddings replace local sentence-transformers for entity
   linking.** Higher quality, no local model download. `FAISSVectorIndexRepository`
   is sized from `embedding_provider.dimension()` (3072) at construction time,
   not the `EmbeddingSettings.dimension` default (768, which stays relevant
   only if a local-model `EmbeddingProvider` is wired in later). KeyBERT still
   uses its own small local model internally for keyphrase scoring — that's
   an internal implementation detail of KeyBERT, unrelated to entity linking.
3. **Triple Validation (stage 10) uses a rule-based domain/range checker,
   now layered with real `pyshacl`.** The rule-based check (reusing the exact
   `ValidationResult`/`ValidationIssue` pattern already proven in
   `ontology_validator.py`) runs first; `shacl_validator.py` then compiles
   the single relevant relation into a SHACL shape per triple and runs
   `pyshacl.validate()` against it, adding any violation as an additional
   `SHACL_DOMAIN_RANGE_VIOLATION` issue — WARNING, not INVALID, for the same
   coarse-entity-typing reason in point 4 below (pyshacl hits the identical
   limitation; this doesn't change the judgment call, just how the same
   check is performed). See `shacl_validator.py`'s docstring for why it's
   built per-triple rather than once for a whole batch.
4. **Entities get a coarse ontology class at creation time** (stage 7):
   spaCy's `PERSON`/`ORG` labels map to the core `Person`/no-core-class-exists
   respectively, everything else defaults to the core `Concept` class. This is
   intentionally coarse — classifying an entity into a precise *domain* class
   (e.g. `Model` vs `Dataset`) needs a real classifier, out of scope here.
   Consequence: triple validation's domain/range check is meaningful for core
   relations (`is_prerequisite_of`, `is_related_to`, `part_of`, which is most
   of what relation extraction produces) but will usually WARN, not fail, on
   domain-specific relations like `trained_on` — by design (see
   `domain_range_validator.py` docstring), not a bug.
5. **Relation extraction's dependency-parse tier (tier 1) maps verb lemmas to
   ontology relation names via a small, extensible lookup table**
   (`VERB_TO_RELATION` in `relation_extraction_service.py`), not a learned
   model — consistent with the spec's "verb/auxiliary relationships" framing,
   and the same "explicit lookup table, not a black box" pattern the cue-phrase
   tier already uses.
6. **Knowledge Fusion's confidence formula** (base + extractor-agreement
   bonus + evidence-count bonus, capped at 1.0) implements the additive
   formula named in the pipeline spec's Phase 5; exact bonus values
   (`_AGREEMENT_BONUS`, `_EVIDENCE_BONUS_PER_EXTRA`, `_MAX_EVIDENCE_BONUS` in
   `fusion_service.py`) are a reasonable starting point, not derived from any
   specified constant — tune them against real eval data later.
7. **Graph Storage relationship type is fixed to `RELATES_TO`** in Neo4j, with
   the ontology predicate stored as a `predicate` property, rather than using
   the predicate string as the Cypher relationship type — keeps every query
   fully parameterized (no dynamic Cypher type interpolation) at the cost of
   one extra property filter in downstream queries.
8. **Transitive reasoning** (multi-hop prerequisite chaining, PDF Phase 5) was
   not implemented — flagged as a stretch item, not attempted this pass.

## Found and fixed while verifying end-to-end

- **Relation extraction was pairing every entity in a whole chunk, not just
  within a sentence.** On a real document this produced 447 relations from
  51 mentions — a combinatorial flood of cross-paragraph "co-occurrence"
  noise that Knowledge Fusion correctly filtered back out (5 accepted, 442
  pending). Scoping pairing to entities that co-occur in the *same sentence*
  (`relation_extraction_service.py`, using `doc.sents`) cut that to 50
  relations for the same input with no loss of the real signal — a stronger,
  more linguistically sound signal in the first place.
- **`FilesystemEntityRepository`'s name-index sanitizer didn't strip
  newlines/control characters.** A KeyBERT keyphrase spanning a paragraph
  break (e.g. `"Novel Antibiotics\nGenome"`) produced an invalid Windows
  filename and crashed the pipeline mid-run. Fixed with a proper unsafe-
  character sweep; regression test added (`test_entity_repository.py`).
- **`data/raw/amazon.pdf` is not real content.** It's a Word document
  (mislabeled `.pdf`) whose text is the exact two synthetic sentences
  `verify_module2.py`'s `_make_sample_docx()` fixture writes — leftover test
  output, not source material. `data/raw/DNA Task 3.docx` (32 paragraphs,
  real coursework on genome sequencing / antibiotic discovery) is what
  `verify_full_pipeline.py` now demos against.
- **Windows faiss+torch OpenMP crash.** Loading `faiss` in the same process
  as `torch`/`sentence-transformers`-family packages (pulled in transitively
  by KeyBERT) aborts the interpreter with `OMP: Error #15` unless
  `KMP_DUPLICATE_LIB_OK=TRUE` is set before `faiss` is imported — now set
  defensively inside `faiss_vector_index.py`.

**Remaining tuning opportunity, not fixed this pass:** on the real demo
document only 5 of 50 relations cleared `triple_confidence_min` (0.5) — the
`CUE_PHRASES`/`VERB_TO_RELATION` tables in `relation_extraction_service.py`
were written with ML/CS vocabulary ("trained_on", "minimizes") and don't
cover biology or other domains, so most same-sentence pairs fall through to
the weak 0.35-confidence co-occurrence tier. Broadening those tables with
domain-agnostic academic phrasing (or adding domain-specific tables per
`Subject`) would raise the accepted-triple yield without weakening fusion's
filtering.

## Verify

```bash
pip install -r requirements.txt        # adds keybert, yake
python -m spacy download en_core_web_sm
pytest tests/unit -q                    # existing 46 + new tests, no external calls
python scripts/verify_full_pipeline.py  # real Azure OpenAI + Neo4j, against data/raw/amazon.pdf
```

## Next

**FastAPI `/review` UI** (human-in-the-loop console for both ontology
extensions and the triple review queue) and real `pyshacl` validation were
the two pieces of the original spec still open after this pass — both have
since been built (see `MODULE_API_UI_NOTES.md` for the review console; point
3 above for `pyshacl`).
