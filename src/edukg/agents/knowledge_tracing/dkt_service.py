from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from edukg.core.config import get_settings
from edukg.core.logging_setup import get_logger
from edukg.core.models.knowledge_tracing import KTEvent

logger = get_logger(__name__)

_MAX_SEQUENCE_LEN = 512


@dataclass(slots=True)
class DKTPrediction:
    mastery: float
    top_contributing_event_ids: list[UUID]


class DKTInferenceService:
    """Loads a trained DKTModel (produced by scripts/train_dkt.py) if one
    exists on disk and serves inference. Degrades to "no DKT available"
    (predict() returns None, is_available is False) rather than raising
    whenever no model file exists yet, the checkpoint fails to load, or a
    concept isn't in the trained vocabulary — KnowledgeTracingAgent always
    has BKT as a fallback, so a missing or stale DKT model is never fatal to
    the feature."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self._model_dir = model_dir or (get_settings().paths.data_dir / "models")
        self._model = None
        self._device = None
        self._vocab: dict[str, int] | None = None
        self._load()

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        model_path = self._model_dir / "dkt_model.pt"
        vocab_path = self._model_dir / "dkt_vocab.json"
        if not model_path.exists() or not vocab_path.exists():
            logger.info("dkt_model_not_found", model_dir=str(self._model_dir))
            return
        try:
            import torch

            from edukg.agents.knowledge_tracing.dkt_model import DKTModel

            payload = json.loads(vocab_path.read_text(encoding="utf-8"))
            # scripts/train_dkt.py saves {"vocab": ..., "d_model": ...} so
            # the exact trained architecture can be reconstructed before
            # load_state_dict() — a mismatched d_model is a tensor-shape
            # load error, not a silently-wrong model. Falls back to the
            # legacy flat-vocab-dict format (d_model always 32) for a model
            # file saved before d_model started scaling with vocab size.
            if isinstance(payload, dict) and "vocab" in payload and "d_model" in payload:
                vocab = payload["vocab"]
                d_model = payload["d_model"]
            else:
                vocab = payload
                d_model = 32
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DKTModel(vocab_size=len(vocab), d_model=d_model)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            self._vocab = vocab
            self._device = device
            self._model = model
            logger.info("dkt_model_loaded", vocab_size=len(vocab), d_model=d_model, device=str(device))
        except Exception as exc:  # noqa: BLE001 - any load failure just means "no DKT this run"
            logger.error("dkt_model_load_failed", error=str(exc))
            self._model = None

    def predict(self, concept_id: UUID, events: list[KTEvent]) -> DKTPrediction | None:
        """`events` is the student's full chronological history across every
        concept (not just `concept_id`) — DKT reasons about the whole
        interaction sequence, not one concept in isolation. Returns None if
        no model is loaded, the concept was never seen during training, or
        there's no usable history."""
        if self._model is None or self._vocab is None:
            return None
        concept_key = str(concept_id)
        if concept_key not in self._vocab:
            return None

        recent = events[-_MAX_SEQUENCE_LEN:]
        known_events = [e for e in recent if str(e.concept_id) in self._vocab]
        if not known_events:
            return None

        import torch

        vocab_size = len(self._vocab)
        interaction_ids = [self._vocab[str(e.concept_id)] + (vocab_size if e.correct else 0) for e in known_events]
        tensor = torch.tensor([interaction_ids], dtype=torch.long, device=self._device)

        with torch.no_grad():
            probs = self._model(tensor)
            attn = self._model.attention_weights(tensor)

        target_idx = self._vocab[concept_key]
        mastery = float(probs[0, -1, target_idx].item())

        last_step_attention = attn[0, -1].tolist()
        ranked = sorted(range(len(known_events)), key=lambda i: last_step_attention[i], reverse=True)
        top_event_ids = [known_events[i].id for i in ranked[:3]]

        return DKTPrediction(mastery=mastery, top_contributing_event_ids=top_event_ids)

    def forward_probs(self, interaction_ids: list[int]) -> list[list[float]] | None:
        """Raw per-timestep, per-concept P(correct) for an already
        vocab-indexed interaction sequence (`concept_idx + vocab_size *
        correct` codes) — the same forward pass predict() uses internally,
        exposed directly for sequence-level held-out evaluation
        (scripts/metrics_report.py) rather than predict()'s single
        "mastery of one concept right now" shape. None if no model is
        loaded."""
        if self._model is None:
            return None
        import torch

        tensor = torch.tensor([interaction_ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            probs = self._model(tensor)
        return probs[0].tolist()
