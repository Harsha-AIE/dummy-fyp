from __future__ import annotations

# Cue phrase -> (ontology relation name, reversed). Matched case-insensitively
# against the text spanning between two co-occurring entity mentions (tier 2
# of the stage-8 cascade). The caller always finds mention_a before mention_b
# in document order; `reversed=True` means mention_a is semantically the
# relation's *object*, not subject, despite appearing first in the text — the
# caller must swap subject/object before storing the Relation. Needed because
# not every phrase reads subject-first: "A requires B" / "A builds on B" / "A
# assumes knowledge of B" all put the PREREQUISITE (B) after the dependent
# concept (A), unlike "A is a prerequisite of B" which already reads
# subject-first. Same issue for "A consists of B" (A is the whole, B is the
# part — part_of's subject is the part, so B is the subject here, not A).
# Found via a direct cross-check against the real project's relation
# semantics after a teammate's independent, separately-built codebase
# flagged "prerequisite direction could be reversed" — verified the claim by
# reading this table, not by trusting the report.
CUE_PHRASES: dict[str, tuple[str, bool]] = {
    "is a prerequisite of": ("is_prerequisite_of", False),
    "is a prerequisite for": ("is_prerequisite_of", False),
    "prerequisite of": ("is_prerequisite_of", False),
    "requires": ("is_prerequisite_of", True),
    "builds on": ("is_prerequisite_of", True),
    "assumes knowledge of": ("is_prerequisite_of", True),
    "is part of": ("part_of", False),
    "is a component of": ("part_of", False),
    "consists of": ("part_of", True),
    "defines": ("defines", False),
    "is defined as": ("defines", False),
    "trained on": ("trained_on", False),
    "is trained using": ("trained_on", False),
    "minimizes": ("minimizes", False),
    "uses": ("is_related_to", False),
    "is related to": ("is_related_to", False),
    "is associated with": ("is_related_to", False),
}


class CuePhraseExtractor:
    """Tier 2 of the relation-extraction cascade (stage 8): looks for a known
    linking phrase in the text between two entity mentions and maps it
    directly to an ontology relation name, plus whether the caller's default
    text-order subject/object assignment needs to be swapped (see
    CUE_PHRASES's docstring)."""

    def __init__(self, cue_phrases: dict[str, tuple[str, bool]] | None = None) -> None:
        self._cue_phrases = cue_phrases or CUE_PHRASES

    def extract(self, gap_text: str) -> tuple[str, bool] | None:
        lowered = gap_text.lower()
        for phrase, (relation_name, reversed_) in self._cue_phrases.items():
            if phrase in lowered:
                return relation_name, reversed_
        return None
