"""Run: python scripts/train_dkt.py
Trains the Deep Knowledge Tracing model (src/edukg/agents/knowledge_tracing/
dkt_model.py) and saves it to data/models/dkt_model.pt (+ dkt_vocab.json)
for KnowledgeTracingAgent/DKTInferenceService to load at API startup.

Cold-start handling (see MODULE_KT_NOTES.md): a brand-new EduKG deployment
has zero real student interaction history. If fewer than MIN_REAL_EVENTS
real `kt_events` rows exist in Postgres, this script generates a clearly
logged SYNTHETIC bootstrap dataset — simulated students sampled from the
same BKT generative process used elsewhere in this codebase (bkt.py) — so
the model has something real to learn from for a demo. Re-run this script
after real usage accumulates (via the Live Agent Demo) to retrain on
genuine data; nothing else needs to change when that happens.

Requires the KG to already be populated (run scripts/verify_full_pipeline.py
first) — the concept vocabulary is read from the live entity repository.

Trains with an 80/20 train/validation split and early stopping on
validation loss (not a fixed epoch count) — an earlier fixed-30-epoch
version drove training loss to ~0.04 while held-out accuracy stayed near
chance (~53%), a textbook overfitting signature caught by
scripts/metrics_report.py's held-out evaluation and confirmed live (mastery
moved the wrong direction after a second misconception). Synthetic student
count and model width (d_model) both scale with the live KG's vocabulary
size (see SYNTHETIC_TOUCHES_PER_CONCEPT_TARGET and _model_dim_for_vocab
below) rather than being fixed constants — a KG that grows over time
(documents get re-run, new ones added) otherwise silently starves this
script of the density/capacity it needs, exactly as happened when this
project's KG grew from 313 to 1380 entities. Adam weight_decay=1e-4
further reduces memorization risk.
"""

from __future__ import annotations

import json
import random

import torch
import torch.nn.functional as F

from edukg.agents.knowledge_tracing.dkt_model import DKTModel
from edukg.agents.knowledge_tracing.synthetic import Interaction, generate_synthetic_sequences
from edukg.core.config import get_settings
from edukg.core.logging_setup import configure_logging, get_logger
from edukg.core.models.knowledge_tracing import KTEvent
from edukg.infrastructure.filesystem.entity_repository import FilesystemEntityRepository
from edukg.infrastructure.postgres.kt_store import KnowledgeTracingStore

configure_logging()
logger = get_logger("train_dkt")

MIN_REAL_EVENTS = 200
# A fixed student count (previously 600) was tuned against whatever the KG's
# entity count happened to be at the time (313) — it silently goes stale as
# the KG grows, since the vocabulary scripts/train_dkt.py trains against is
# read live from FilesystemEntityRepository, not pinned to that number.
# Found live: retraining after the KG grew to 1380 entities with the old
# fixed 600 produced ~1.7 synthetic training touches per concept on average
# (600 students x ~4.5 concepts each / 1380 concepts) and validation loss
# sat at chance level (~0.693, ln 2) from epoch 1 onward — nowhere near
# enough signal across a vocabulary four times larger than it was tuned
# for. Scaling student count by the live vocab size, targeting a fixed
# average touches-per-concept density instead, keeps this correct as the
# KG keeps growing. MIN_SYNTHETIC_STUDENTS is a floor for a small/early-
# stage KG, matching the previously-validated 600 baseline.
SYNTHETIC_TOUCHES_PER_CONCEPT_TARGET = 10
SYNTHETIC_CONCEPTS_PER_STUDENT_RANGE = (3, 10)
MIN_SYNTHETIC_STUDENTS = 600
SYNTHETIC_INTERACTIONS_PER_STUDENT = (20, 60)
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VALIDATION_FRACTION = 0.2


def _model_dim_for_vocab(vocab_size: int) -> int:
    """DKTModel's hidden width (d_model) must scale with vocab_size: every
    concept's mastery is read off the SAME fixed-size hidden state through
    one shared `readout: Linear(d_model, vocab_size)` layer, so a hidden
    state too small to represent that many concepts' trajectories at once
    caps achievable accuracy regardless of how much training data exists —
    a capacity bottleneck, not a data problem. Found live: retraining at
    vocab_size=1380 with the previous fixed d_model=32 (tuned for
    vocab_size=313) produced validation loss stuck at chance (~0.693, ln 2)
    from epoch 1 onward even after scaling synthetic data volume up 3.5x —
    ruling out "not enough data" and pointing at capacity instead.
    Uses the standard embedding-size heuristic (Howard & Gugger's fastai
    rule, min(600, round(1.6 * n_categories**0.56))), rounded to the
    nearest multiple of 16 for clean attention-head division. Sanity check:
    this formula maps vocab_size=313 to 32 — exactly the value this
    project had already hand-picked for that vocab size — so it reproduces
    the existing, previously-validated choice rather than just inventing a
    new one."""
    raw = 1.6 * (vocab_size**0.56)
    return max(32, round(raw / 16) * 16)


def _load_vocab() -> dict[str, int]:
    entities = FilesystemEntityRepository().list_all()
    vocab = {str(entity.id): idx for idx, entity in enumerate(entities)}
    return vocab


def _real_sequences(vocab: dict[str, int]) -> tuple[list[list[Interaction]], int]:
    events = KnowledgeTracingStore().list_all_events()
    by_student: dict[str, list[KTEvent]] = {}
    for event in events:
        by_student.setdefault(event.student_id, []).append(event)
    sequences: list[list[Interaction]] = []
    for student_events in by_student.values():
        student_events.sort(key=lambda e: e.created_at)
        seq = [Interaction(vocab[str(e.concept_id)], e.correct) for e in student_events if str(e.concept_id) in vocab]
        if len(seq) >= 2:
            sequences.append(seq)
    return sequences, len(events)


def _sequence_loss(model, sequence: list[Interaction], vocab_size: int, device) -> torch.Tensor | None:
    """Standard DKT training objective: at each timestep, predict the NEXT
    interaction's correctness from everything seen so far. Shared by
    training and validation — only whether gradients are tracked differs."""
    if len(sequence) < 2:
        return None

    interaction_ids = [i.concept_idx + (vocab_size if i.correct else 0) for i in sequence]
    tensor = torch.tensor([interaction_ids], dtype=torch.long, device=device)
    probs = model(tensor)[0]  # (seq_len, vocab_size)

    preds = torch.stack([probs[t, sequence[t + 1].concept_idx] for t in range(len(sequence) - 1)])
    targets = torch.tensor([float(sequence[t + 1].correct) for t in range(len(sequence) - 1)], device=device)
    return F.binary_cross_entropy(preds, targets)


def _train_step(model, optimizer, sequence: list[Interaction], vocab_size: int, device) -> float | None:
    loss = _sequence_loss(model, sequence, vocab_size, device)
    if loss is None:
        return None
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def _validate(model, sequences: list[list[Interaction]], vocab_size: int, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for seq in sequences:
            loss = _sequence_loss(model, seq, vocab_size, device)
            if loss is not None:
                losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def main() -> None:
    vocab = _load_vocab()
    if not vocab:
        logger.error("train_dkt_empty_vocab")
        print("No entities found in the KG — run scripts/verify_full_pipeline.py first.")
        return

    real_sequences, real_event_count = _real_sequences(vocab)
    if real_event_count >= MIN_REAL_EVENTS:
        sequences = real_sequences
        logger.info("train_dkt_using_real_data", event_count=real_event_count, student_count=len(sequences))
        print(f"Training on {real_event_count} real kt_events across {len(sequences)} students.")
    else:
        avg_concepts_per_student = sum(SYNTHETIC_CONCEPTS_PER_STUDENT_RANGE) / 2
        synthetic_student_count = max(
            MIN_SYNTHETIC_STUDENTS,
            int(len(vocab) * SYNTHETIC_TOUCHES_PER_CONCEPT_TARGET / avg_concepts_per_student),
        )
        sequences = generate_synthetic_sequences(
            list(vocab.values()),
            student_count=synthetic_student_count,
            interactions_range=SYNTHETIC_INTERACTIONS_PER_STUDENT,
            concepts_per_student_range=SYNTHETIC_CONCEPTS_PER_STUDENT_RANGE,
        )
        logger.info(
            "train_dkt_using_synthetic_data",
            real_event_count=real_event_count,
            threshold=MIN_REAL_EVENTS,
            vocab_size=len(vocab),
            synthetic_students=len(sequences),
        )
        print(
            f"Only {real_event_count} real kt_events exist (need {MIN_REAL_EVENTS}+) — "
            f"training on {len(sequences)} SYNTHETIC bootstrap students instead. "
            "Re-run this script later once real usage accumulates."
        )

    random.shuffle(sequences)
    split = max(1, int(len(sequences) * (1 - VALIDATION_FRACTION)))
    train_sequences, val_sequences = sequences[:split], sequences[split:]
    print(f"Train/validation split: {len(train_sequences)} / {len(val_sequences)} sequences")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    d_model = _model_dim_for_vocab(len(vocab))
    print(f"Model width: d_model={d_model} (scaled for vocab_size={len(vocab)})")
    model = DKTModel(vocab_size=len(vocab), d_model=d_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Early stopping on held-out validation loss: the earlier fixed-30-epoch
    # version drove *training* loss to ~0.04 while held-out accuracy stayed
    # near chance (~53%, confirmed via scripts/metrics_report.py, and
    # observed live: mastery moved the wrong direction after a second
    # misconception) — a textbook overfitting signature on a too-small
    # dataset. More synthetic students (600, up from 60) plus stopping at
    # the best validation checkpoint instead of a fixed epoch count directly
    # targets that, rather than just training longer on the same data.
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        random.shuffle(train_sequences)
        train_losses = [
            loss
            for seq in train_sequences
            if (loss := _train_step(model, optimizer, seq, len(vocab), device)) is not None
        ]
        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else float("nan")
        val_loss = _validate(model, val_sequences, len(vocab), device)
        logger.info("train_dkt_epoch", epoch=epoch, train_loss=avg_train_loss, val_loss=val_loss)
        print(f"epoch {epoch:3d}/{MAX_EPOCHS}  train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Stopping early — no validation improvement for {EARLY_STOP_PATIENCE} epochs "
                      f"(best val_loss={best_val_loss:.4f} at epoch {epoch - epochs_without_improvement}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_dir = get_settings().paths.data_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "dkt_model.pt")
    # d_model is saved alongside the vocab (not just the vocab) because
    # DKTInferenceService must reconstruct the exact same architecture
    # before load_state_dict() — a mismatched d_model is a tensor-shape
    # load error, not a silently-wrong-but-running model.
    (model_dir / "dkt_vocab.json").write_text(json.dumps({"vocab": vocab, "d_model": d_model}), encoding="utf-8")
    logger.info(
        "train_dkt_saved", model_dir=str(model_dir), vocab_size=len(vocab), d_model=d_model, best_val_loss=best_val_loss
    )
    print(f"Saved best model (val_loss={best_val_loss:.4f}) + vocab to {model_dir}")


if __name__ == "__main__":
    main()
