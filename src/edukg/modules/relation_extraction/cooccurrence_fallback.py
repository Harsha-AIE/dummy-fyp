from __future__ import annotations

from edukg.core.models import Entity


class CooccurrenceFallback:
    """Tier 3 (weakest) of the relation-extraction cascade (stage 8): two
    entities that co-occur in the same chunk but showed no dependency-parse
    or cue-phrase signal are still linked, at low confidence, as related_to —
    better to surface a weak candidate for downstream fusion/review than to
    silently drop the co-occurrence signal.

    Gated on both entities having resolved to *some* ontology class first —
    an unresolved entity isn't a real concept the KG can reason about yet, so
    pairing it on bare textual co-occurrence alone is pure noise, not a weak
    signal worth keeping. This is a cheaper, structural check than
    TripleValidationService's later domain/range check (deliberately WARNING-
    not-INVALID there, given coarse entity typing — see its docstring): this
    one only asks "does this entity have any resolved class at all", not
    "the *correct* one", so it's safe to apply as a hard skip here rather
    than a downstream warning.
    """

    RELATION_NAME = "is_related_to"

    def extract(self, entity_a: Entity | None = None, entity_b: Entity | None = None) -> str | None:
        if entity_a is not None and entity_b is not None:
            if entity_a.ontology_class_id is None or entity_b.ontology_class_id is None:
                return None
        return self.RELATION_NAME
