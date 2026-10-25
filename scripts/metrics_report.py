"""Run: python scripts/metrics_report.py

Comprehensive, live metrics report for EduKG: Knowledge Graph construction
quality, Knowledge Tracing (BKT + DKT) quality, and agent/retrieval quality.

Every number is computed against the real system, not hardcoded:
- KG construction + Knowledge Tracing sections read directly from the real
  filesystem entity/ontology stores, Neo4j, and Postgres.
- The DKT-vs-BKT evaluation section runs a FRESH synthetic held-out set
  (same generator as scripts/train_dkt.py's cold-start bootstrap, but a
  different run — never seen during training) through both models and
  reports next-response prediction accuracy/log-loss/AUC, the standard way
  knowledge-tracing models are evaluated. Clearly labeled synthetic, since
  there may not be enough real usage yet to evaluate on.
- The agent/retrieval section calls the *live* backend over HTTP
  (http://127.0.0.1:8000 by default) with a fixed benchmark query set,
  reusing its already-built content index/agents rather than rebuilding
  them here — this is why the backend needs to be running for this
  section (every other section works standalone).

Prints a formatted report to stdout and saves the same data as JSON to
data/reports/metrics_<timestamp>.json.
"""

from __future__ import annotations

import json
import math
import random
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

from edukg.agents.knowledge_tracing.bkt import DEFAULT_BKT_PARAMS, update_mastery
from edukg.agents.knowledge_tracing.dkt_service import DKTInferenceService
from edukg.agents.knowledge_tracing.synthetic import Interaction, generate_synthetic_sequences
from edukg.agents.pedagogical_guide.escalation_log import EscalationLogStore
from edukg.core.config import get_settings
from edukg.core.logging_setup import configure_logging, get_logger
from edukg.infrastructure.filesystem.document_repository import FilesystemDocumentRepository
from edukg.infrastructure.filesystem.entity_repository import FilesystemEntityRepository
from edukg.infrastructure.filesystem.ontology_repository import FilesystemOntologyRepository
from edukg.infrastructure.neo4j.neo4j_graph_repository import Neo4jGraphRepository
from edukg.infrastructure.postgres.kt_store import KnowledgeTracingStore
from edukg.modules.fusion.pending_triple_store import PendingTripleStore

configure_logging()
logger = get_logger("metrics_report")

BACKEND_BASE_URL = "http://127.0.0.1:8000"
HELD_OUT_STUDENT_COUNT = 40
HELD_OUT_INTERACTIONS_RANGE = (20, 50)
HELD_OUT_SEED = 20260101  # fixed for a reproducible report, distinct from train_dkt.py's own runs

# A small hand-labeled benchmark: (query, the intent a human would call it,
# whether it should tag at least one KG concept). This is what makes
# precision/recall/F1 possible below — without a labeled expectation to
# compare against, "accuracy" has nothing to be accurate *about*.
BENCHMARK_QUERIES: list[dict] = [
    {"query": "What is the role of genome sequencing in discovering new antibiotics?", "expected_intent": "question", "expects_concepts": True},
    {"query": "What is DNA?", "expected_intent": "question", "expects_concepts": True},
    {"query": "I'm confused about why microorganisms would produce antibiotics in the first place.", "expected_intent": "misconception", "expects_concepts": True},
    {"query": "I think antibiotics are made by viruses, not bacteria.", "expected_intent": "misconception", "expects_concepts": True},
    {"query": "Can you give me an example of how antibiotics work?", "expected_intent": "example_request", "expects_concepts": True},
    {"query": "Can you show me an example of a biosynthetic gene cluster?", "expected_intent": "example_request", "expects_concepts": True},
    {"query": "hi", "expected_intent": "off_topic", "expects_concepts": False},
    {"query": "what's the weather like today", "expected_intent": "off_topic", "expects_concepts": False},
]


# --------------------------------------------------------------------------
# Section 1: Knowledge Graph construction quality (local reads)
# --------------------------------------------------------------------------
def kg_construction_metrics() -> dict:
    entity_repo = FilesystemEntityRepository()
    document_repo = FilesystemDocumentRepository()
    ontology_repo = FilesystemOntologyRepository()
    pending_triple_store = PendingTripleStore()

    entities = entity_repo.list_all()
    documents = document_repo.list_all_documents()
    total_chunks = sum(len(document_repo.get_chunks(d.id)) for d in documents)
    mention_total = sum(len(e.mention_ids) for e in entities)
    core_classes, core_relations = ontology_repo.load_core_ontology()

    result: dict = {
        "documents_ingested": len(documents),
        "total_chunks": total_chunks,
        "total_entities": len(entities),
        "entities_by_type": dict(Counter(e.entity_type.value for e in entities)),
        "avg_mentions_per_entity": round(mention_total / len(entities), 3) if entities else 0.0,
        "ontology_core_classes": len(core_classes),
        "ontology_core_relations": len(core_relations),
        "pending_ontology_extensions": len(ontology_repo.list_pending_extensions()),
        "pending_triples": len(pending_triple_store.list_pending()),
    }

    try:
        graph_repo = Neo4jGraphRepository()
        try:
            nodes, edges = graph_repo.export_graph()
            result["graph_nodes"] = len(nodes)
            result["accepted_triples_in_graph"] = len(edges)
            result["relations_by_predicate"] = dict(Counter(e["predicate"] for e in edges))
            result["avg_accepted_triple_confidence"] = (
                round(sum(e["confidence"] for e in edges) / len(edges), 3) if edges else 0.0
            )
        finally:
            graph_repo.close()
    except Exception as exc:  # noqa: BLE001
        result["graph_error"] = f"Neo4j unreachable: {exc}"

    return result


# --------------------------------------------------------------------------
# Section 2: Knowledge Tracing store metrics — real usage (Postgres)
# --------------------------------------------------------------------------
def kt_store_metrics() -> dict:
    store = KnowledgeTracingStore()
    try:
        store.ensure_schema()
        events = store.list_all_events()
        estimates = store.list_all_mastery()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}

    result: dict = {
        "available": True,
        "total_events": len(events),
        "unique_students": len({e.student_id for e in events}),
        "unique_concepts_tracked": len({e.concept_id for e in events}),
        "events_by_source": dict(Counter(e.source.value for e in events)),
        "events_correct_rate": round(sum(1 for e in events if e.correct) / len(events), 3) if events else 0.0,
        "total_mastery_estimates": len(estimates),
    }
    if estimates:
        result["avg_bkt_mastery"] = round(sum(e.bkt_mastery for e in estimates) / len(estimates), 3)
        dkt_values = [e.dkt_mastery for e in estimates if e.dkt_mastery is not None]
        result["avg_dkt_mastery"] = round(sum(dkt_values) / len(dkt_values), 3) if dkt_values else None
        result["avg_ensemble_mastery"] = round(sum(e.ensemble_mastery for e in estimates) / len(estimates), 3)
        result["low_mastery_count"] = sum(1 for e in estimates if e.ensemble_mastery < 0.4)
        result["high_mastery_count"] = sum(1 for e in estimates if e.ensemble_mastery >= 0.7)
    return result


# --------------------------------------------------------------------------
# Section 3: DKT vs. BKT held-out predictive quality (fresh synthetic set)
# --------------------------------------------------------------------------
def _bkt_next_step_predictions(sequence: list[Interaction]) -> list[float]:
    """BKT's P(correct) for sequence[t+1]'s concept, predicted from only
    sequence[0..t] — mirrors DKT's 'predict the next response' objective
    exactly, so the two are directly comparable."""
    mastery_by_concept: dict[int, float] = {}
    preds: list[float] = []
    for t in range(len(sequence) - 1):
        obs = sequence[t]
        prior = mastery_by_concept.get(obs.concept_idx, DEFAULT_BKT_PARAMS.p_init)
        mastery_by_concept[obs.concept_idx] = update_mastery(prior, obs.correct, 1.0, DEFAULT_BKT_PARAMS)
        next_concept = sequence[t + 1].concept_idx
        preds.append(mastery_by_concept.get(next_concept, DEFAULT_BKT_PARAMS.p_init))
    return preds


def _auc(preds: list[float], targets: list[bool]) -> float | None:
    """Mann-Whitney U / rank-sum AUC — no sklearn dependency needed."""
    pos = [p for p, t in zip(preds, targets, strict=True) if t]
    neg = [p for p, t in zip(preds, targets, strict=True) if not t]
    if not pos or not neg:
        return None
    order = sorted(range(len(preds)), key=lambda i: preds[i])
    ranks = [0.0] * len(preds)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and preds[order[j]] == preds[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    sum_ranks_pos = sum(ranks[idx] for idx, t in enumerate(targets) if t)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _binary_metrics(preds: list[float], targets: list[bool]) -> dict:
    """Standard binary-classification metrics for the "will the student
    answer correctly next?" prediction task — a prediction >=0.5 counts as
    predicting "correct" (the positive class).

    accuracy: fraction of predictions (correct/incorrect) that matched what
      actually happened. Simplest metric, but misleading if one outcome
      (e.g. "correct") is much more common than the other.
    precision: of every time the model predicted "correct", the fraction
      that actually were correct. High precision = few false alarms.
    recall: of every time the student was actually correct, the fraction
      the model successfully predicted. High recall = few missed cases.
    f1_score: the harmonic mean of precision and recall — one number that
      only looks good if both precision AND recall are good (a model that
      just always predicts "correct" would score high on one but not both).
    confusion_matrix: the four raw counts precision/recall/f1 are built
      from — true_positive (predicted correct, was correct), false_positive
      (predicted correct, was wrong), true_negative (predicted incorrect,
      was incorrect), false_negative (predicted incorrect, was actually
      correct — a "missed" correct answer).
    log_loss / brier_score / auc: how well-CALIBRATED the predicted
      probabilities are, not just the yes/no call — lower log_loss/brier is
      better, AUC (0.5=random guessing, 1.0=perfect ranking) measures
      whether higher-predicted-probability cases really are more often
      correct.
    """
    n = len(preds)
    eps = 1e-7
    predicted_positive = [p >= 0.5 for p in preds]

    true_positive = sum(1 for pp, t in zip(predicted_positive, targets, strict=True) if pp and t)
    false_positive = sum(1 for pp, t in zip(predicted_positive, targets, strict=True) if pp and not t)
    true_negative = sum(1 for pp, t in zip(predicted_positive, targets, strict=True) if not pp and not t)
    false_negative = sum(1 for pp, t in zip(predicted_positive, targets, strict=True) if not pp and t)

    accuracy = (true_positive + true_negative) / n
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    log_loss = (
        -sum(
            math.log(max(p, eps)) if t else math.log(max(1 - p, eps))
            for p, t in zip(preds, targets, strict=True)
        )
        / n
    )
    brier = sum((p - float(t)) ** 2 for p, t in zip(preds, targets, strict=True)) / n

    return {
        "n_predictions": n,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "log_loss": round(log_loss, 4),
        "brier_score": round(brier, 4),
        "auc": round(auc, 4) if (auc := _auc(preds, targets)) is not None else None,
    }


def dkt_bkt_evaluation_metrics() -> dict:
    vocab_path = get_settings().paths.data_dir / "models" / "dkt_vocab.json"
    if not vocab_path.exists():
        return {"available": False, "reason": "no trained DKT model — run scripts/train_dkt.py first"}
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    # Same {"vocab": ..., "d_model": ...} shape DKTInferenceService reads —
    # see its _load() for why d_model travels alongside the vocab now.
    vocab: dict[str, int] = payload["vocab"] if isinstance(payload, dict) and "vocab" in payload else payload
    vocab_size = len(vocab)

    dkt_service = DKTInferenceService()
    if not dkt_service.is_available:
        return {"available": False, "reason": "DKT model failed to load"}

    random.seed(HELD_OUT_SEED)
    held_out = generate_synthetic_sequences(
        list(vocab.values()), student_count=HELD_OUT_STUDENT_COUNT, interactions_range=HELD_OUT_INTERACTIONS_RANGE
    )

    bkt_preds: list[float] = []
    bkt_targets: list[bool] = []
    dkt_preds: list[float] = []
    dkt_targets: list[bool] = []

    for seq in held_out:
        if len(seq) < 2:
            continue
        bkt_preds.extend(_bkt_next_step_predictions(seq))
        bkt_targets.extend(i.correct for i in seq[1:])

        interaction_ids = [i.concept_idx + (vocab_size if i.correct else 0) for i in seq]
        probs = dkt_service.forward_probs(interaction_ids)
        if probs is None:
            continue
        for t in range(len(seq) - 1):
            dkt_preds.append(probs[t][seq[t + 1].concept_idx])
            dkt_targets.append(seq[t + 1].correct)

    agreement = None
    if dkt_preds and len(dkt_preds) == len(bkt_preds):
        agreement = round(sum(abs(a - b) for a, b in zip(bkt_preds, dkt_preds, strict=True)) / len(bkt_preds), 4)

    return {
        "available": True,
        "note": "SYNTHETIC held-out set (BKT-generative-process), never used during training",
        "held_out_students": len(held_out),
        "held_out_seed": HELD_OUT_SEED,
        "vocab_size": vocab_size,
        "bkt": _binary_metrics(bkt_preds, bkt_targets),
        "dkt": _binary_metrics(dkt_preds, dkt_targets),
        "avg_abs_bkt_dkt_disagreement": agreement,
    }


# --------------------------------------------------------------------------
# Section 4: Pedagogical Guide escalation log (local read)
# --------------------------------------------------------------------------
def escalation_log_metrics() -> dict:
    turns = EscalationLogStore().list_all()
    if not turns:
        return {"total_escalations": 0}
    return {
        "total_escalations": len(turns),
        "escalations_by_intent": dict(Counter(t.intent.value for t in turns)),
        "avg_confidence_at_escalation": round(sum(t.confidence for t in turns) / len(turns), 3),
    }


# --------------------------------------------------------------------------
# Section 5: Live agent/retrieval quality (HTTP against the running backend)
# --------------------------------------------------------------------------
def _http_post(path: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND_BASE_URL}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _multiclass_precision_recall_f1(y_true: list[str], y_pred: list[str]) -> dict:
    """Per-class precision/recall/F1 for intent classification, plus a
    macro average (each class weighted equally, regardless of how many
    benchmark examples it had — appropriate here since the benchmark is
    small and deliberately balanced, not a large real-world sample)."""
    labels = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, dict] = {}
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "support": sum(1 for t in y_true if t == label),
        }
        f1s.append(f1)
    accuracy = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / len(y_true)
    return {
        "accuracy": round(accuracy, 3),
        "macro_f1": round(sum(f1s) / len(f1s), 3) if f1s else 0.0,
        "per_class": per_class,
    }


def _binary_precision_recall_f1(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and not p)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and not p)
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "confusion_matrix": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
    }


def agent_retrieval_metrics() -> dict:
    try:
        urllib.request.urlopen(f"{BACKEND_BASE_URL}/api/health", timeout=5)
    except urllib.error.URLError as exc:
        return {"available": False, "reason": f"backend not reachable at {BACKEND_BASE_URL}: {exc}"}

    true_intents: list[str] = []
    predicted_intents: list[str] = []
    expects_concepts: list[bool] = []
    got_concepts: list[bool] = []
    escalation_count = 0
    confidences: list[float] = []
    retrieval_hit_count = 0
    top_scores: list[float] = []
    answered = 0

    for case in BENCHMARK_QUERIES:
        try:
            resp = _http_post("/api/agents/ask", {"message": case["query"], "top_k": 5})
        except Exception as exc:  # noqa: BLE001
            logger.error("metrics_benchmark_query_failed", query=case["query"], error=str(exc))
            continue
        answered += 1
        classification = resp["classification"]
        true_intents.append(case["expected_intent"])
        predicted_intents.append(classification["intent"])
        expects_concepts.append(case["expects_concepts"])
        got_concepts.append(bool(classification["detected_concepts"]))
        confidences.append(classification["confidence"])
        if classification["escalated_to_llm"]:
            escalation_count += 1
        results = resp["retrieval"]["results"]
        if results:
            retrieval_hit_count += 1
            top_scores.append(results[0]["final_score"])

    if answered == 0:
        return {"available": False, "reason": "no benchmark query got a response from the backend"}

    return {
        "available": True,
        "benchmark_query_count": answered,
        "intent_classification": _multiclass_precision_recall_f1(true_intents, predicted_intents),
        # The metric that would have caught the "what is DNA -> no concepts
        # matched" gap before a human had to notice it manually.
        "concept_tagging": _binary_precision_recall_f1(expects_concepts, got_concepts),
        "avg_classification_confidence": round(sum(confidences) / answered, 3),
        "escalation_rate": round(escalation_count / answered, 3),
        "retrieval_hit_rate": round(retrieval_hit_count / answered, 3),
        "avg_top_retrieval_score": round(sum(top_scores) / len(top_scores), 3) if top_scores else 0.0,
    }


# --------------------------------------------------------------------------
# Report assembly / printing
# --------------------------------------------------------------------------
def _print_section(title: str, data: dict) -> None:
    print(f"\n=== {title} ===")
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k2, v2 in value.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    print("EduKG Metrics Report")
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kg_construction": kg_construction_metrics(),
        "knowledge_tracing_store": kt_store_metrics(),
        "dkt_vs_bkt_evaluation": dkt_bkt_evaluation_metrics(),
        "pedagogical_guide_escalations": escalation_log_metrics(),
        "agent_retrieval_quality": agent_retrieval_metrics(),
    }

    _print_section("Knowledge Graph Construction", report["kg_construction"])
    _print_section("Knowledge Tracing Store (real usage)", report["knowledge_tracing_store"])
    _print_section("DKT vs. BKT Held-Out Evaluation (synthetic)", report["dkt_vs_bkt_evaluation"])
    _print_section("Pedagogical Guide Escalations", report["pedagogical_guide_escalations"])
    _print_section("Agent/Retrieval Quality (live benchmark)", report["agent_retrieval_quality"])

    reports_dir = get_settings().paths.data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"metrics_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved full report to {out_path}")


if __name__ == "__main__":
    main()
