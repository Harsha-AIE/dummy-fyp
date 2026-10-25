from uuid import uuid4

from edukg.core.models import Chunk
from edukg.core.models.enums import ExtractionMethod
from edukg.modules.entity_extraction.mention_extractor import MentionExtractor


class _FakeEnt:
    def __init__(self, text: str, start_char: int, end_char: int, label_: str) -> None:
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.label_ = label_


class _FakeDoc:
    def __init__(self, ents: list[_FakeEnt]) -> None:
        self.ents = ents


class _FakeNLP:
    """Stub spaCy pipeline: returns a fixed set of entities regardless of input,
    so tests don't need the real en_core_web_sm model installed."""

    def __init__(self, ents: list[_FakeEnt]) -> None:
        self._ents = ents

    def __call__(self, text: str) -> _FakeDoc:
        return _FakeDoc(self._ents)


class _FakeKeyphraseExtractor:
    def __init__(self, keywords: list[tuple[str, float]]) -> None:
        self._keywords = keywords

    def extract_keywords(self, text: str, **kwargs) -> list[tuple[str, float]]:
        return self._keywords


def _make_chunk(text: str) -> Chunk:
    return Chunk(
        document_id=uuid4(),
        sequence_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
        extraction_method=ExtractionMethod.NATIVE_TEXT,
        token_count=len(text) // 4,
    )


def test_extractor_combines_ner_keybert_yake_without_overlap() -> None:
    text = "Gradient descent minimizes the loss function during training."
    chunk = _make_chunk(text)

    nlp = _FakeNLP([_FakeEnt("Gradient descent", 0, 17, "CONCEPT")])
    keybert = _FakeKeyphraseExtractor([("loss function", 0.9), ("gradient descent", 0.8)])
    yake = _FakeKeyphraseExtractor([("training", 0.1)])

    extractor = MentionExtractor(nlp, keybert, yake)
    mentions = extractor.extract(chunk)

    surface_forms = {m.surface_form.lower() for m in mentions}
    assert "gradient descent" in surface_forms  # from NER
    assert "loss function" in surface_forms  # from KeyBERT
    assert "training" in surface_forms  # from YAKE
    # KeyBERT's "gradient descent" overlaps the NER span and must not duplicate it.
    assert sum(1 for m in mentions if m.surface_form.lower() == "gradient descent") == 1


def test_extractor_sets_candidate_type_from_ner_label() -> None:
    text = "Ada Lovelace wrote the first algorithm."
    chunk = _make_chunk(text)

    nlp = _FakeNLP([_FakeEnt("Ada Lovelace", 0, 12, "PERSON")])
    extractor = MentionExtractor(nlp, _FakeKeyphraseExtractor([]), _FakeKeyphraseExtractor([]))

    mentions = extractor.extract(chunk)

    assert len(mentions) == 1
    assert mentions[0].candidate_type == "PERSON"
    assert mentions[0].chunk_id == chunk.id
    assert mentions[0].document_id == chunk.document_id


def test_extractor_returns_empty_for_blank_chunk_text() -> None:
    chunk = _make_chunk("   ")
    extractor = MentionExtractor(_FakeNLP([]), _FakeKeyphraseExtractor([]), _FakeKeyphraseExtractor([]))

    assert extractor.extract(chunk) == []


def test_extractor_captures_short_acronym_ner_and_keyphrases_miss() -> None:
    # Regression test: "DNA" never became a KG entity because it lost out to
    # longer, more "representative" phrases in KeyBERT/YAKE's capped top-N —
    # even though the source document was literally titled "DNA Sequencing".
    text = "DNA Sequencing Task: investigate the role of genome sequencing."
    chunk = _make_chunk(text)

    extractor = MentionExtractor(_FakeNLP([]), _FakeKeyphraseExtractor([]), _FakeKeyphraseExtractor([]))
    mentions = extractor.extract(chunk)

    assert "DNA" in {m.surface_form for m in mentions}


def test_acronym_overlapping_a_ner_span_is_not_duplicated() -> None:
    text = "DNA Sequencing Task starts now."
    chunk = _make_chunk(text)

    nlp = _FakeNLP([_FakeEnt("DNA Sequencing", 0, 14, "ORG")])
    extractor = MentionExtractor(nlp, _FakeKeyphraseExtractor([]), _FakeKeyphraseExtractor([]))
    mentions = extractor.extract(chunk)

    # "DNA" (chars 0-3) overlaps the NER span (0-14) and must be skipped,
    # not extracted a second time as a separate mention.
    assert sum(1 for m in mentions if m.surface_form == "DNA") == 0
    assert sum(1 for m in mentions if m.surface_form == "DNA Sequencing") == 1


def test_acronym_coexists_with_non_overlapping_keybert_yake_candidates() -> None:
    text = "PCR amplifies target sequences for downstream analysis."
    chunk = _make_chunk(text)

    keybert = _FakeKeyphraseExtractor([("downstream analysis", 0.9)])
    extractor = MentionExtractor(_FakeNLP([]), keybert, _FakeKeyphraseExtractor([]))
    mentions = extractor.extract(chunk)

    surface_forms = {m.surface_form for m in mentions}
    assert "PCR" in surface_forms
    assert "downstream analysis" in surface_forms


def test_bare_stopword_candidates_from_keybert_and_yake_are_dropped() -> None:
    # Regression test: a query containing "...in the first place" tagged the
    # bare word "first" as a KG concept (YAKE's stopword filtering let it
    # through even though KeyBERT's own stop_words='english' wouldn't have),
    # which then polluted graph-distance re-ranking — any other chunk that
    # also happened to contain the word "first" got a false perfect
    # graph_score, regardless of actual topical relevance.
    text = "Genome sequencing helps discover new antibiotics."
    chunk = _make_chunk(text)

    keybert = _FakeKeyphraseExtractor([("first", 0.5), ("genome sequencing", 0.9)])
    yake = _FakeKeyphraseExtractor([("one", 0.05), ("antibiotics", 0.1)])
    extractor = MentionExtractor(_FakeNLP([]), keybert, yake)
    mentions = extractor.extract(chunk)

    surface_forms = {m.surface_form.lower() for m in mentions}
    assert "first" not in surface_forms
    assert "one" not in surface_forms
    assert "genome sequencing" in surface_forms
    assert "antibiotics" in surface_forms


def test_multiword_phrase_containing_a_stopword_is_kept() -> None:
    # Only a *bare* single-word stopword candidate is dropped — a multi-word
    # phrase that happens to contain one stays intact and meaningful.
    text = "This is the state of the art approach for this task."
    chunk = _make_chunk(text)

    keybert = _FakeKeyphraseExtractor([("state of the art", 0.9)])
    extractor = MentionExtractor(_FakeNLP([]), keybert, _FakeKeyphraseExtractor([]))
    mentions = extractor.extract(chunk)

    assert "state of the art" in {m.surface_form.lower() for m in mentions}


def test_ner_ordinal_and_cardinal_labels_are_dropped() -> None:
    # Regression test: "...in the first place" kept tagging "first" as a
    # concept even after the KeyBERT/YAKE low-information filter existed,
    # because spaCy's NER tags it ORDINAL and NER results bypass that check
    # entirely (they never go through KeyBERT/YAKE at all).
    text = "We enrolled 46 participants; Ada Lovelace led the first cohort."
    chunk = _make_chunk(text)

    nlp = _FakeNLP(
        [
            _FakeEnt("46", 12, 14, "CARDINAL"),
            _FakeEnt("Ada Lovelace", 30, 42, "PERSON"),
            _FakeEnt("first", 51, 56, "ORDINAL"),
        ]
    )
    extractor = MentionExtractor(nlp, _FakeKeyphraseExtractor([]), _FakeKeyphraseExtractor([]))
    mentions = extractor.extract(chunk)

    surface_forms = {m.surface_form for m in mentions}
    assert "46" not in surface_forms
    assert "first" not in surface_forms
    assert "Ada Lovelace" in surface_forms  # a genuinely meaningful NER label is untouched


def test_bare_number_candidates_are_dropped_but_numbers_in_phrases_are_kept() -> None:
    # Same failure mode as bare stopwords, found immediately after fixing
    # that one: a standalone digit ("1", "46", "1,112") is never a
    # meaningful KG concept on its own, but a number that's part of a
    # genuine multi-word term (e.g. "Type 2 diabetes") is fine.
    text = "Filtering excluded subjects; 867 of 1,112 remained across sites with Type 2 diabetes."
    chunk = _make_chunk(text)

    keybert = _FakeKeyphraseExtractor([("46", 0.5), ("Type 2 diabetes", 0.9)])
    yake = _FakeKeyphraseExtractor([("1,112", 0.05), ("867", 0.1)])
    extractor = MentionExtractor(_FakeNLP([]), keybert, yake)
    mentions = extractor.extract(chunk)

    surface_forms = {m.surface_form.lower() for m in mentions}
    assert "46" not in surface_forms
    assert "1,112" not in surface_forms
    assert "867" not in surface_forms
    assert "type 2 diabetes" in surface_forms
