from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from edukg.core.config import get_settings
from edukg.core.logging_setup import get_logger
from edukg.core.models import ConfidenceScore, Triple
from edukg.core.models.enums import ValidationStatus

logger = get_logger(__name__)

# Additive confidence formula per the pipeline spec's Phase 5: base confidence
# (the strongest single extraction) plus a bonus when multiple independent
# extractor methods agreed on the same fact, plus a small bonus per extra
# piece of supporting evidence, capped so fusion can never inflate a triple
# past full confidence.
_AGREEMENT_BONUS = 0.10
_EVIDENCE_BONUS_PER_EXTRA = 0.05
_MAX_EVIDENCE_BONUS = 0.20

_STATUS_SEVERITY = {
    ValidationStatus.INVALID: 0,
    ValidationStatus.PENDING: 1,
    ValidationStatus.WARNING: 2,
    ValidationStatus.VALID: 3,
}


class KnowledgeFusionService:
    """
    Stage 11: deduplicates triples by natural key (subject, predicate,
    object), aggregating confidence and evidence across duplicate
    extractions, then splits the fused triples into accepted (proceeds to
    Graph Storage) vs. pending review (INVALID validation status, or fused
    confidence below thresholds.triple_confidence_min) — mirrors the
    ontology extension pipeline's auto-review/pending split.
    """

    def __init__(self, min_confidence: float | None = None) -> None:
        self._min_confidence = (
            min_confidence if min_confidence is not None else get_settings().thresholds.triple_confidence_min
        )

    def fuse(self, triples: list[Triple]) -> tuple[list[Triple], list[Triple]]:
        groups: dict[tuple, list[Triple]] = defaultdict(list)
        for triple in triples:
            groups[triple.natural_key()].append(triple)

        accepted: list[Triple] = []
        pending: list[Triple] = []
        for group in groups.values():
            fused = self._fuse_group(group)
            if fused.validation_status == ValidationStatus.INVALID or fused.confidence.value < self._min_confidence:
                pending.append(fused)
            else:
                accepted.append(fused)

        logger.info(
            "knowledge_fusion_done",
            input_count=len(triples),
            fused_count=len(groups),
            accepted=len(accepted),
            pending=len(pending),
        )
        return accepted, pending

    def _fuse_group(self, group: list[Triple]) -> Triple:
        base = max(group, key=lambda t: t.confidence.value)
        methods = {t.confidence.method for t in group}
        agreement_bonus = _AGREEMENT_BONUS if len(methods) > 1 else 0.0
        evidence_bonus = min(_EVIDENCE_BONUS_PER_EXTRA * (len(group) - 1), _MAX_EVIDENCE_BONUS)
        fused_value = min(1.0, base.confidence.value + agreement_bonus + evidence_bonus)

        return base.model_copy(
            update={
                "confidence": ConfidenceScore(
                    value=fused_value,
                    method="fusion",
                    reason=f"{len(group)} supporting extraction(s); methods={sorted(methods)}",
                ),
                "validation_status": self._worst_status(t.validation_status for t in group),
            }
        )

    @staticmethod
    def _worst_status(statuses: Iterable[ValidationStatus]) -> ValidationStatus:
        return min(statuses, key=lambda s: _STATUS_SEVERITY[s])
