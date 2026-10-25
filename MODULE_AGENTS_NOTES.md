# Agent Layer — Knowledge Manager (RAG) + Pedagogical Guide

Builds on the completed KG pipeline (Modules 1-5, stages 1-12). Adds the
first two of the multi-agent tutor's agents described in the project's
architecture notes. Response Generation/orchestrator and Knowledge Tracing
(DKT/BKT) were explicitly out of scope for this pass (see rationale below) —
Knowledge Tracing was built in a later pass (`MODULE_KT_NOTES.md`), and the
Response Generation/orchestrator layer (`TutorResponseAgent`, merging all
three agents into one Socratic-scaffolded reply via `/api/agents/ask`) has
since been built too — see its docstring in
`agents/tutor_orchestrator/tutor_response_agent.py`.

## Files created

**`core/models/session.py`** — `DialogueTurn` (matches the `dialogue_turns`
table in the data architecture notes). **`core/models/enums.py`** — new
`IntentCategory` enum.

**Additive interface methods** (existing ABCs, both implemented only by the
filesystem/Neo4j classes that already exist):
- `DocumentRepository.list_all_documents()` — `FilesystemDocumentRepository`
- `GraphRepository.shortest_path_length()` — `Neo4jGraphRepository`, via a
  hop-bounded `shortestPath` Cypher query

**`agents/shared/concept_tagger.py`** — `ConceptTagger`, wraps
`MentionExtractor` + `EntityLinkingService` (Module 3) for tagging KG
entities mentioned in a query/message string, in read-only mode.

**`agents/knowledge_manager/`** — `BM25Index`, `reciprocal_rank_fusion`,
`GraphDistanceReranker`, `ContentIndexService`/`ContentIndex`,
`KnowledgeManagerAgent`.

**`agents/pedagogical_guide/`** — `IntentClassificationStrategy` +
`KeywordIntentStrategy` (tier 1), `LLMIntentClassifier` (tier 2),
`EscalationLogStore`, `PedagogicalGuideAgent`.

**`scripts/verify_agents.py`** — real end-to-end demo against the live
Neo4j graph + already-ingested documents.

## Scope decisions

1. **No Knowledge Tracing agent exists**, so the RAG re-ranking formula's
   `gamma*mastery_gap_weight` term is dropped — `KnowledgeManagerAgent` only
   combines fused retrieval score + KG graph-distance. Swappable: a third
   term can be added later without touching the fusion/rerank wiring.
2. **Pedagogical Guide's tier-1 "local classifier" is a keyword strategy**,
   not a trained model — no labeled intent data exists. Same swappable-
   strategy shape as `SubjectDetectionStrategy`. Tier 2 escalates to the
   already-built `AzureOpenAILLMProvider`.
3. **`ConceptTagger` is strictly read-only**: an unmatched query term is
   dropped, never minted as a new KG entity, and — found and fixed during
   verification — a *matched* entity's mention/alias bookkeeping is also
   never persisted from a query. Only the reviewed ingestion pipeline is
   allowed to write to entity storage. (`EntityLinkingService.link(...,
   create_if_missing=False)` now gates persistence, not just entity
   creation.)
4. **Escalations are logged, not fed into a live retraining loop** — no
   trainer exists. `EscalationLogStore` writes each escalation to
   `data/agent_logs/escalations/<id>.json`; the "falling escalation rate"
   metric is computable from this trail once a trainer exists, not before.
5. **Content index is rebuilt in memory on every agent startup**, not
   persisted — fine at current scale (a handful of documents); persisting
   it (the underlying `FAISSVectorIndexRepository`/`BM25Index` both already
   support save/load) is a follow-up once re-embedding every chunk on every
   startup becomes noticeably slow.

## Found and fixed while verifying end-to-end

- **BM25's classic (non-smoothed) idf can be exactly 0** for a term
  appearing in about half a small corpus — `BM25Index.search()` was
  filtering out non-positive scores, which silently dropped valid results
  on small corpora (including in tests). Removed the filter: this index
  only ever feeds Reciprocal Rank Fusion, which uses rank position, not the
  raw score value, so the sign of the score was never meaningful downstream.
- **Plain substring matching in `KeywordIntentStrategy` produced false
  positives**: "why do" (a QUESTION-tier phrase) matched inside "doesn't"
  (its first two letters), making "I'm confused, why doesn't this work?"
  score as QUESTION instead of MISCONCEPTION. Fixed with `\b` boundaries at
  both ends of each compiled phrase pattern.
- **`ConceptTagger`'s "read-only" mode was incomplete** — see scope
  decision 3 above. Confirmed via a real agent run: two live queries about
  "antibiotics"/"microorganisms" had inflated an already-ingested entity's
  `mention_ids` from 1/2 to 5/4 before the fix; a clean pipeline + agent
  re-run afterward confirmed the counts stayed at the correct 1/2.

## Verify

```bash
pip install -r requirements.txt        # adds rank_bm25
pytest tests/unit tests/integration -q  # 98 passed
python scripts/verify_full_pipeline.py  # regenerates the KG if needed
python scripts/verify_agents.py         # real Azure OpenAI + Neo4j, both agents
```

## Next

Response Generation + a thin orchestrator (ties `PedagogicalGuideAgent`'s
intent + `KnowledgeManagerAgent`'s ranked context into a final LLM reply,
with the Socratic-scaffolding guardrail from the architecture notes) was the
natural next slice — now built as `TutorResponseAgent`, wired into
`POST /api/agents/ask`. Knowledge Tracing (DKT/BKT) needed real student
interaction data before it could be more than a stub — built once the
synthetic-bootstrap cold-start approach made that tractable; see
`MODULE_KT_NOTES.md`.
