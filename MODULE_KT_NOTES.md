# Knowledge Tracing — BKT + DKT

Builds on the Agent Layer (`MODULE_AGENTS_NOTES.md`), which explicitly left
Knowledge Tracing out: "needs real student interaction data before it can be
more than a stub." This pass builds it anyway, by being explicit about the
one constraint that made it a stub before — no correctness signal exists —
and designing around it rather than around real assessment data that
doesn't exist yet.

## The core problem: no correctness signal

Both DKT and BKT need a sequence of (concept, correct/incorrect)
observations. Nothing in the system produces one: `PedagogicalGuideAgent`
classifies *intent* (question / misconception / example-request /
off-topic), not answer correctness, and there is no quiz/assessment layer.

`MISCONCEPTION` intent is the one exception — the classifier has already
determined the student expressed a wrong belief, which *is* a checked,
negative correctness signal. Everything else is, at best, a weak proxy.
`agents/knowledge_tracing/knowledge_tracing_agent.py`'s `_INTENT_SIGNAL`
table, applied per `detected_concept` on every classified `DialogueTurn`:

| Intent | `correct` | `weight` | Rationale |
|---|---|---|---|
| `MISCONCEPTION` | `False` | `1.0` | A checked negative signal — trust it fully. |
| `QUESTION` | `True` | `0.3` | Engagement isn't evidence of understanding; weak positive. |
| `EXAMPLE_REQUEST` | `True` | `0.2` | Weaker still — the least informative engagement signal. |
| `OFF_TOPIC` / `UNKNOWN` | — | — | No event emitted; no reliable signal either way. |

This is flagged here the same way every other scope decision in this
project has been (coarse entity typing at extraction time, etc.) — not
hidden, and designed to be
replaced without touching anything downstream: the moment a real
assessment/orchestrator layer produces a genuine `correct: bool`, it's one
more producer writing into the same `kt_events` table.

## BKT (available immediately, no training)

`agents/knowledge_tracing/bkt.py` — standard Corbett & Anderson (1994)
update: Bayes' rule turns an observation into a posterior, then a fixed
`P(T)` transition step lets mastery move up between opportunities. Default
parameters (`p_init=0.3, p_transit=0.1, p_slip=0.1, p_guess=0.2`) are
literature defaults, not fit against real EduKG data — there isn't enough
of it yet. `weight` (from the table above) dampens partial-confidence
observations by blending the Bayes-updated posterior back toward the prior
before the transition step, so a "weak positive" question doesn't move
mastery as much as a confirmed misconception moves it down.

This is what makes the feature usable from the very first event, not only
after data accumulates — BKT has no training step at all.

## DKT (real, trained, degrades gracefully)

`agents/knowledge_tracing/dkt_model.py` — a small self-attentive (SAKT-
style) model: `(concept, correct)` interaction pairs embedded and fed
through one causally-masked `TransformerEncoderLayer`, sized deliberately
small (`d_model=32`, 2 heads) — this project's interaction volume doesn't
justify more, and a bigger model would just overfit the bootstrap data
faster. `DKTInferenceService` loads a trained checkpoint from
`data/models/` if one exists, extracts attention weights at inference time
to identify which prior events most influenced a prediction (the "top
contributing events" in the mastery explanation — a real attribution, not a
templated guess), and returns `None` — never raises — whenever no model is
loaded or a concept wasn't in the training vocabulary. `BKT` is always
available as a fallback, so a missing or stale DKT model is never fatal.

**Cold start**: a brand-new deployment has zero real usage history.
`scripts/train_dkt.py` trains on real `kt_events` once at least 200 exist;
until then it generates a clearly-logged **synthetic bootstrap dataset** —
simulated students sampled from the same BKT generative process as
`bkt.py`'s defaults — so the model has something real to learn from for a
demo. Confirmed in this environment: PyTorch is installed with CUDA
available (`torch 2.7.1+cu118`), so training genuinely runs on GPU here,
not just in principle. Re-running the script later retrains on real usage
as it accumulates — same "improves with real data" shape as the
Pedagogical Guide's escalation log.

**Found and fixed via live testing (not just the held-out eval):** the
original fixed-30-epoch, 60-synthetic-student training run drove training
loss to ~0.04 while held-out accuracy stayed near chance (~53%, AUC 0.54)
— overfitting to a too-small dataset. This wasn't just a metrics-script
number: it was reproducible live — running the same student through two
consecutive misconceptions about the same concept, DKT's mastery estimate
*increased* instead of decreasing (BKT correctly decreased throughout).
Fixed by training on 600 synthetic students (up from 60), an 80/20
train/validation split, early stopping on validation loss instead of a
fixed epoch count, and Adam `weight_decay=1e-4`. Result: held-out F1
0.56→0.67-0.69 (varies slightly run to run — the synthetic generator has no
fixed seed at training time), AUC 0.54→0.59-0.60, log-loss 0.95→0.66-0.67.
Live re-test: the same two-consecutive-misconceptions scenario now shows a
much smaller, largely immaterial fluctuation (~0.03, down from ~0.13) with
mastery correctly landing and staying in "low" territory throughout, rather
than climbing into "uncertain"/"strong" territory after negative evidence.
Not perfectly monotonic per-step even now — a neural sequence model isn't
mathematically guaranteed to be, unlike BKT's closed-form update — which is
exactly why the ensemble keeps BKT in the mix rather than relying on DKT
alone.

**Ensemble**: `0.5*bkt + 0.5*dkt` when DKT has a prediction for that
concept, `bkt` alone otherwise. A simple, documented default — swappable
once there's real evaluation data to justify a different weighting.

## Filling the Knowledge Manager's mastery seam

`KnowledgeManagerAgent`/`GraphDistanceReranker` were built in the previous
pass with the `gamma*mastery_gap_weight` term explicitly documented and
left at zero, "swappable once a mastery vector exists." It now does:
`KnowledgeManagerAgent` takes an optional `mastery_provider` (student_id ->
{concept_id: mastery}) and `gamma` (default `0.15` where wired from the API
layer, `0.0`/inert by default on the class itself so existing callers are
unaffected); `retrieve()` takes an optional `student_id` and blends
`gamma * average(1 - mastery)` over each candidate chunk's concepts that
intersect the student's known concepts — chunks covering more of the
student's weak spots rank higher. `GraphDistanceReranker` itself is
untouched; the term lives directly in `KnowledgeManagerAgent`, next to
`alpha`/`beta`.

## Correctness/schema notes

- `core/models/knowledge_tracing.py` — `KTEvent` (matches the `kt_events`
  table: student_id, concept_id, source, correct, weight),
  `MasteryEstimate` (matches `mastery_estimates`: adds `ensemble_mastery`
  and `explanation`/`top_contributing_event_ids`, not in the original
  table sketch, but cheap to persist and needed by the API/frontend).
- `infrastructure/postgres/kt_store.py` — `KnowledgeTracingStore`, mirrors
  `PostgresDocumentStore`'s conventions exactly (per-call connections, no
  pooling, `ensure_schema()` from `AppResources`, `ExternalServiceError` on
  failure). `kt_events.id` is a UUID primary key (matching the domain
  model's own id), not the `SERIAL` sketched in early planning — a
  same-purpose, implementation-level refinement.
- `student_id` was already a field on `ClassifyRequest`/`DialogueTurn` but
  silently dropped by the API router before this pass — fixed as part of
  wiring Knowledge Tracing in, not as a separate bug.

## Explicitly out of scope (flagged now, not discovered later)

- **Worksheet grading integration** (`source='worksheet'` rows) — the
  SmartGrade feature itself was never built.
- **A session/orchestrator layer** — student identity is a lightweight
  client-generated id in the browser's `localStorage`
  (`frontend/lib/student-id.ts`), not real auth.
- **Ragas / Explanation-Traceability evaluation harness.**
- **Background auto-retraining** — `scripts/train_dkt.py` is a manual,
  rerun-able script, matching how pipeline runs are already manually
  triggered, not a scheduled job.

## Verify

```bash
pip install -r requirements.txt        # adds torch (already present transitively via sentence-transformers)
pytest tests/unit tests/integration -q
python scripts/train_dkt.py            # trains DKT — real data if enough exists, synthetic bootstrap otherwise
```

Then, with both servers running: ask a question containing a clear
misconception on the Live Agent Demo, confirm the "Student Mastery" card
appears with a lowered score and an explanation citing that turn; ask a
follow-up question about the same concept and confirm the score moves;
check `/mastery` for the full vector; `GET /api/kt/events/{student_id}`
should match what was submitted.
