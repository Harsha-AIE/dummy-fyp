# EduKG — Full Project Methodology

**A knowledge-graph-grounded, multi-agent AI tutor system.**
This document explains, end to end: what existed in the repository before this work started, what was built on top of it, how the system works, and what the real, measured accuracy numbers are — so anyone reviewing this project (teammates, evaluators) can see exactly what was done and verify it themselves.

---

## 1. Executive Summary

EduKG takes raw educational documents (PDF/DOCX), builds a validated Knowledge Graph out of them through a 12-stage pipeline, and serves that graph through three cooperating AI agents (a RAG-based Knowledge Manager, a Pedagogical Guide that classifies student intent, and a Knowledge Tracing engine that estimates student mastery) behind a full web application (FastAPI backend + Next.js frontend).

**Current state:** all 12 pipeline stages are implemented and running against real documents, all three tutor agents are implemented and live in the UI, the system is backed by real infrastructure (Neo4j graph database, Neon Postgres, Azure OpenAI, a GPU-trained PyTorch model), and it is covered by **161 automated tests** (156 unit + 5 integration against the real live databases), all passing.

---

## 2. What Was Already in the Repository (Starting Point)

The repository (`Knowledge-graph-construction`) was started by a teammate and already contained:

- **Module 1 — Project Bootstrap:** the core architecture — typed Pydantic v2 domain models, dependency-inversion interfaces (`core/interfaces/`), the pipeline orchestrator skeleton, exception hierarchy, structured logging.
- **Module 2 — Ingestion, Subject Detection, Ontology:** Stages 1–5 of the 12-stage pipeline — document ingestion (PDF/DOCX/OCR), subject detection, ontology loading/extension/validation.
- **Stages 6–12 (Entity Extraction through Graph Storage) were empty stubs.** No entity extraction, no relation extraction, no agents, no API, no UI existed.

In short: the foundation and the first third of the pipeline existed. Everything from entity extraction onward — the actual "knowledge graph" part of the Knowledge Graph project, every AI agent, and the entire application layer — did not exist yet.

---

## 3. What Was Built, In Order

### Phase 1 — Completing the Knowledge Graph Construction Pipeline (Stages 6–12)

This is the piece that turns raw text into an actual graph. Built against the project's own architecture specification (a 12-stage pipeline mandate):

| Stage | What it does | How |
|---|---|---|
| 6. Entity Extraction | Finds candidate concept mentions in text | spaCy NER + KeyBERT + YAKE keyphrase extraction, run in parallel, plus a dedicated acronym pass (see §7) |
| 7. Entity Linking | Resolves mentions to canonical KG entities | Exact-name match first, then FAISS cosine-similarity search over entity embeddings (Azure OpenAI `text-embedding-3-large`, 3072-dim) |
| 8. Relation Extraction | Finds relationships between entities | A 3-tier cascade: dependency parsing (verb/auxiliary structure) → cue-phrase matching ("is a prerequisite of", "uses") → co-occurrence fallback |
| 9. Triple Generation | Projects relations into Subject–Predicate–Object triples | With full provenance (which document, which sentence) |
| 10. Triple Validation | Checks triples are structurally valid | Rule-based domain/range class checking (see §9, scope decisions) |
| 11. Knowledge Fusion | Deduplicates and scores confidence | Additive confidence formula (base + extractor agreement + evidence count); low-confidence/ambiguous triples routed to a review queue instead of silently accepted |
| 12. Graph Storage | Persists the accepted graph | Neo4j, via parameterized Cypher `MERGE` (no duplicate nodes/edges, evidence counts increment instead) |

**Result at this point:** a real, resumable, checkpointed pipeline that turns a PDF/DOCX into entities, relationships, and validated triples in a live Neo4j graph.

### Phase 2 — The Tutor Agents (Knowledge Manager + Pedagogical Guide)

Two of the three AI agents described in the project's architecture, built directly on top of the completed graph:

- **Knowledge Manager Agent (RAG):** hybrid retrieval — dense embedding search *and* BM25 lexical search, combined with Reciprocal Rank Fusion, then re-ranked by how close the retrieved content's concepts are to the query's concepts *in the graph* (shortest-path distance in Neo4j). This is graph-grounded RAG, not plain vector search.
- **Pedagogical Guide Agent:** classifies what a student's message actually is — a genuine `question`, a `misconception`, a request for an `example`, or `off_topic` — using a fast local tier-1 classifier first, escalating to an LLM (Azure GPT-4o) only when tier-1 isn't confident. Every message also gets tagged against the live knowledge graph so the system knows *which concepts* the student is asking about.

### Phase 3 — Full-Stack Web Application

A FastAPI backend (`src/edukg/api/`) exposing every piece above over HTTP, and a Next.js 15 + TypeScript + Tailwind + shadcn/ui frontend giving it a real interface:

- **Live Agent Demo** — ask a question, see both agents respond in real time
- **Knowledge Graph visualizer** — force-directed graph of the live Neo4j data
- **Review Console** — human-in-the-loop approval queue for anything the pipeline wasn't confident enough to auto-accept
- **Pipeline Runs dashboard** — upload a document and watch all 12 stages run live via Server-Sent Events

### Phase 4 — Real Data Persistence

- Uploaded documents are stored durably in **Neon Postgres**, not just the local filesystem.
- Dashboard statistics (entity/triple counts) update live as documents are processed — polled automatically, not static numbers.

### Phase 5 — Knowledge Tracing: BKT + DKT (the third tutor agent)

The project's architecture calls for a third agent — Knowledge Tracing — to estimate what each student has and hasn't mastered. This was built as a genuine, trained model, not a placeholder:

- **BKT (Bayesian Knowledge Tracing):** the classic Corbett & Anderson algorithm. Works from the very first interaction, no training needed — an interpretable, always-available baseline.
- **DKT (Deep Knowledge Tracing):** a real, small self-attention (transformer) neural network built in PyTorch, **trained on GPU** (confirmed CUDA-accelerated in this environment), that predicts whether a student will answer correctly next, and whose attention weights are used to explain *why* a mastery score is what it is.
- The two are fused into one ensemble mastery estimate per student per concept, with a plain-English explanation ("Mastery of antibiotics is low — driven by a misconception flagged on Sep 17…"), shown live in the UI on both the demo screen and a dedicated Knowledge Tracing page.
- Feeds back into the Knowledge Manager's retrieval ranking — content covering a student's *weak* concepts is now ranked higher for that student.

**Honesty note:** since this is a brand-new system with no real historical student usage yet, the DKT model's initial training uses a clearly-labeled *synthetic* bootstrap dataset (simulated students, generated from the same statistical model BKT itself uses) so there's something real to train and evaluate on. It's designed to retrain on genuine usage data as it accumulates — nothing else needs to change when that happens.

### Phase 6 — Finding and Fixing Real Retrieval Bugs

While stress-testing the system with natural questions (not just the exact phrasing from the source documents), three real, general bugs were found and fixed — this is worth including because it's the difference between "it works on the demo questions" and "it actually works":

1. **Short but important terms (like "DNA") were never becoming their own graph entities**, because the keyphrase-ranking algorithms (KeyBERT/YAKE) rank candidates by whole-document relevance, and a short acronym routinely loses that ranking competition to longer technical phrases — even in a document titled "DNA Sequencing." Fixed with a dedicated, priority acronym-detection pass.
2. **The entity-linking similarity index used for matching a student's question to the graph was being built empty on every server start** — meaning the system could only ever match a question to the graph if it used the *exact same wording* as the source document. Any paraphrase (e.g. "genetic material" instead of "DNA") silently matched nothing. Fixed by populating that index from all existing entities at startup, and sharing it across every future document upload too.
3. **KeyBERT was splitting every multi-word phrase into single words** ("genetic material" → "genetic" + "material" separately), which weakens exactly the kind of phrase that most needs to stay intact. Fixed by widening the phrase-length range it considers.

After these fixes, the agent correctly handles paraphrased, reworded, and synonym questions it had never seen phrased that way before — not just the literal source-document wording.

---

## 4. System Architecture

```
Document (PDF/DOCX)
   │
   ▼
┌─────────────────────────── 12-STAGE PIPELINE ───────────────────────────┐
│ Ingestion → Subject Detection → Ontology Load/Extend/Validate →         │
│ Entity Extraction → Entity Linking → Relation Extraction →              │
│ Triple Generation → Triple Validation → Knowledge Fusion → Graph Storage│
└───────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
                        Neo4j Knowledge Graph (live)
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                             ▼
┌───────────────────┐   ┌────────────────────────┐   ┌─────────────────────┐
│ Knowledge Manager  │   │  Pedagogical Guide      │   │  Knowledge Tracing   │
│ Agent (RAG)         │   │  Agent                  │   │  Agent (BKT + DKT)   │
│ hybrid retrieval +   │   │  intent classification  │   │  mastery estimation, │
│ graph re-ranking     │   │  + concept tagging       │   │  trained PyTorch      │
└──────────┬──────────┘   └───────────┬─────────────┘   └──────────┬──────────┘
           └────────────────────────────┴─────────────────────────────┘
                                     ▼
                          FastAPI backend (HTTP + SSE)
                                     ▼
                     Next.js frontend (live demo, graph view,
                       review console, pipeline dashboard)
```

**Infrastructure actually in use** (not simulated): Neo4j (Docker), Neon-hosted PostgreSQL, Azure OpenAI (GPT-4o for chat, `text-embedding-3-large` for embeddings), PyTorch with CUDA/GPU training, FAISS vector search.

---

## 5. The Agents — What Each One Actually Does

| Agent | Job | Key techniques |
|---|---|---|
| **Knowledge Manager (RAG)** | Retrieve the most relevant, graph-grounded content for a question | Dense (embedding) + lexical (BM25) hybrid search, Reciprocal Rank Fusion, KG graph-distance re-ranking, mastery-gap re-ranking |
| **Pedagogical Guide** | Understand what the student is actually saying | Tiered classifier (fast local rules → LLM escalation only when unsure), KG concept tagging |
| **Knowledge Tracing** | Track what each student has/hasn't mastered | Bayesian Knowledge Tracing (BKT) + a trained Deep Knowledge Tracing (DKT) neural network, ensembled, with attention-based explanations |
| **Tutor Response Orchestrator** | Merge the other three agents' output into one answer the student actually reads | One LLM call combining intent + retrieved evidence + mastery, with a Socratic-scaffolding guardrail self-check baked into the same prompt |

A fourth piece described in the original architecture — a Response Generation/orchestrator layer that would combine all three agents' outputs into one final conversational reply — was flagged as **not built** for a long stretch of this project; it has since been added as `TutorResponseAgent` (`agents/tutor_orchestrator/`), wired into `POST /api/agents/ask`. The demo UI now shows the synthesized reply first, with each agent's individual output still visible underneath (expand "How this answer was built") — the transparency the independent-cards view gave wasn't removed, just no longer the only thing shown.

---

## 6. Evaluation & Accuracy — What the Numbers Actually Mean

All numbers below come from `scripts/metrics_report.py`, a script that computes everything live against the real running system — nothing here is hand-typed or estimated. Run it yourself any time with:

```
set PYTHONPATH=src && python scripts/metrics_report.py
```

### 6.1 Intent Classification (Pedagogical Guide) — **100% accuracy**

Tested against a hand-labeled set of example questions/misconceptions/example-requests/off-topic messages:

| Metric | Value |
|---|---|
| Accuracy | **1.00** (8/8 correct) |
| Macro F1 | **1.00** |
| Precision / Recall (every class) | 1.00 / 1.00 |

### 6.2 Concept Tagging (does the system recognize what the question is about) — **87.5% accuracy**

This is the "some 87% accuracy" number — it measures: given a question, does the system correctly recognize whether it's about a concept in the knowledge graph or not.

| Metric | Value | Meaning |
|---|---|---|
| Accuracy | **0.875** | 7 out of 8 test questions were handled correctly |
| Precision | 0.857 | When it tagged a concept, it was right 85.7% of the time |
| Recall | **1.00** | Every question that *should* have matched a concept, did |
| F1 score | 0.923 | Combined precision+recall score |

(Recall being a perfect 1.00 is the important number here — it means the system never *missed* a real concept; its one imperfection is being slightly over-eager, not under-eager.)

### 6.3 Knowledge Tracing — BKT vs. DKT (held-out evaluation)

Both models were tested on a *fresh* set of simulated student interactions they never saw during training — the standard, honest way to evaluate a predictive model (not just checking training performance).

| Metric | BKT (statistical) | DKT (neural network) |
|---|---|---|
| Accuracy | **74.33%** | 58.83% |
| Precision | 83.89% | 65.93% |
| Recall | 72.17% | 68.55% |
| F1 score | **0.776** | 0.672 |
| AUC (0.5=random, 1.0=perfect) | **0.793** | 0.586 |

**Honest interpretation:** BKT performs solidly. DKT started out close to a coin flip (accuracy 52.9%, F1 0.563, AUC 0.541) because it was trained on a relatively small simulated dataset (60 synthetic students) for a fixed 30 epochs and was overfitting rather than generalizing — the evaluation script caught this, and it was also reproducible *live*: asking the same simulated student two misconceptions in a row about the same concept made DKT's mastery score go *up*, while BKT correctly went down. Fixed by training on 600 synthetic students with an 80/20 train/validation split, early stopping on validation loss, and weight decay — F1 improved to 0.67-0.69 and AUC to 0.59-0.60 (both figures vary slightly run to run). DKT still trails BKT and isn't perfectly monotonic turn-to-turn, which is exactly why the ensemble keeps BKT in the mix rather than relying on DKT alone — this is a known, transparently-reported limitation, not a hidden one.

### 6.4 Knowledge Graph Construction (current graph state)

| Metric | Value |
|---|---|
| Documents ingested | 3 |
| Total entities in graph | 304 |
| Accepted, validated triples | 39 |
| Average triple confidence | 0.691 |
| Ontology core classes / relations | 6 / 5 |

### 6.5 Automated Test Coverage

| Suite | Count | What it tests against |
|---|---|---|
| Unit tests | **156**, all passing | Fake/in-memory dependencies — fast, no external services |
| Integration tests | **5**, all passing | The real Neo4j graph and real Neon Postgres database |
| **Total** | **161 passing** | |

---

## 7. Techniques Used, By Component

| Component | Techniques |
|---|---|
| Entity extraction | spaCy NER, KeyBERT (n-gram 1–3), YAKE keyphrase extraction, custom acronym detection |
| Entity linking | Exact-name matching + FAISS cosine-similarity search (two different thresholds: strict for merging entities during ingestion, lenient for read-only question matching) |
| Relation extraction | Dependency parsing, cue-phrase pattern matching, co-occurrence fallback |
| Retrieval (RAG) | Dense embedding search, BM25 lexical search, Reciprocal Rank Fusion, graph-distance re-ranking |
| Intent classification | Tiered rule-based classifier + LLM (GPT-4o) escalation |
| Knowledge Tracing | Bayesian Knowledge Tracing (Corbett & Anderson), a custom self-attention Transformer trained in PyTorch on GPU |
| Storage | Neo4j (graph), Neon PostgreSQL (documents + tracing events), filesystem JSON (entities/ontology) |

---

## 8. Honest Scope Decisions (What Was Deliberately Simplified)

Documented explicitly, not discovered later:

- **Triple validation now layers real `pyshacl` on top of a rule-based domain/range checker** — the rule-based check runs first (same `ValidationResult`/`ValidationIssue` interface as before), and `shacl_validator.py` adds genuine SHACL violations as an extra WARNING-level signal. Still WARNING, not INVALID: entity typing is coarse (see below), so pyshacl hits the identical limitation the rule-based checker was already designed around — this wasn't a cosmetic swap, it's a real pyshacl pass, just one that (correctly) agrees with the existing judgment call about severity.
- **Triple confidence dedup uses an additive formula**, not a learned model.
- **Response Generation/orchestrator layer is now built** (`TutorResponseAgent`) — merges all three agents' output into one Socratic-scaffolded reply via `/api/agents/ask`, rather than returning three independent results.
- **Transitive reasoning over the KG is now built** — `Neo4jGraphRepository.materialize_transitive_closure()` runs automatically after each pipeline run for every ontology relation marked `is_transitive` (currently `is_prerequisite_of`), so a chain A→B→C also gets an inferred A→C edge, distinguished from directly-extracted edges via `inferred: true`.
- **Knowledge Tracing's correctness signal is derived from intent classification** (a detected misconception = a negative signal; a question = a weak positive signal), because there is no quiz/assessment layer yet to provide a real "right/wrong" signal. Documented and designed to be swapped out the moment a real assessment layer exists.
- **DKT's initial training data is synthetic** (see §3, Phase 5) — by design, until real usage accumulates.
- **Worksheet grading (the SmartGrade OCR feature)** described in the architecture was never built — explicitly out of scope, by deliberate choice (would be a multi-day net-new CV/OCR feature, not a fix).
- **Real user authentication / a session layer** was never built — student identity is a lightweight client-generated id in the browser's `localStorage`, not real auth. Deliberately out of scope for the same reason as SmartGrade.
- **A Ragas / Explanation-Traceability evaluation harness was never integrated** — `scripts/metrics_report.py`'s custom precision/recall/F1/AUC metrics substitute for it and cover the same ground for this project's actual needs.

---

## 9. How To Run and Verify Everything Yourself

```bash
# Backend
cd Knowledge-graph-construction
set PYTHONPATH=src
python -m uvicorn edukg.api.main:app --host 127.0.0.1 --port 8000

# Frontend (separate terminal)
cd frontend
npm run dev

# Run the full automated test suite
pytest tests/unit tests/integration -q

# Regenerate the metrics/accuracy report
python scripts/metrics_report.py

# Retrain the DKT model
python scripts/train_dkt.py
```

Then open `http://localhost:3000` — every screen (Dashboard, Live Agent Demo, Knowledge Graph, Knowledge Tracing, Review Console, Pipeline Runs) is driven by the live backend, not mock data.

---

*This document reflects the state of the system as verified on 2026-09-18, against the live running application and its real databases.*
