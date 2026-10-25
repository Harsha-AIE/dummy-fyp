from __future__ import annotations

from spacy.language import Language
from spacy.tokens import Doc


class DependencyRelationExtractor:
    """
    Tier 1 of the relation-extraction cascade (stage 8): finds a verb/auxiliary
    token that governs both entity mentions within the same sentence via
    spaCy's dependency parse. Returns the verb's lemma as a free-form
    predicate candidate — RelationExtractionService maps that lemma onto a
    real ontology relation name; this class knows nothing about ontologies.
    """

    def __init__(self, nlp: Language) -> None:
        self._nlp = nlp

    def extract(self, doc: Doc, span_a: tuple[int, int], span_b: tuple[int, int]) -> str | None:
        token_a = self._token_at(doc, span_a)
        token_b = self._token_at(doc, span_b)
        if token_a is None or token_b is None or token_a.sent != token_b.sent:
            return None

        verb = self._common_governing_verb(token_a, token_b)
        return verb.lemma_.lower() if verb is not None else None

    def _token_at(self, doc: Doc, span: tuple[int, int]):
        char_span = doc.char_span(span[0], span[1], alignment_mode="expand")
        return char_span.root if char_span is not None else None

    def _common_governing_verb(self, token_a, token_b):
        ancestors_a = {token_a, *token_a.ancestors}
        ancestors_b = {token_b, *token_b.ancestors}
        shared_verbs = [t for t in (ancestors_a & ancestors_b) if t.pos_ in ("VERB", "AUX")]
        if not shared_verbs:
            return None
        return min(shared_verbs, key=lambda t: abs(t.i - token_a.i) + abs(t.i - token_b.i))
