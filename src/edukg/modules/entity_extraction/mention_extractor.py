from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from edukg.core.logging_setup import get_logger
from edukg.core.models import Chunk, Mention

logger = get_logger(__name__)

# KeyBERT's own stop_words='english' filters these out of *its* candidates,
# but YAKE's stopword handling is looser and let bare function words like
# "first"/"one"/"two" through as standalone "concept" entities — found via a
# real, live symptom: a query containing "...in the first place" tagged
# "first" as a KG concept, which then gave an unrelated document's chunk a
# perfect graph_score=1.0 in RAG re-ranking, purely because that chunk
# happened to also contain the word "first" (both mapped to the same junk
# entity). A bare number ("1", "46", "1,112") is the same failure in
# practice — a standalone digit is never a meaningful "concept" for this
# KG — found the same way, immediately after fixing the stopword case
# (a query containing "...in the first place" also tagged "1" once "first"
# stopped being extracted). Both checks apply only to single-word/single-
# token candidates — a multi-word phrase that happens to contain a
# stopword or a number (e.g. "state of the art", "Type 2 diabetes") is
# still meaningful as a whole and is left alone.
_STOPWORDS = ENGLISH_STOP_WORDS
_BARE_NUMBER_PATTERN = re.compile(r"^[\d.,%±]+$")

# spaCy NER labels that are structurally never a meaningful standalone KG
# concept — found via the exact same live symptom as the bare-number filter
# above: "...in the first place" kept tagging "first" as a concept even
# after the KeyBERT/YAKE filter existed, because spaCy's NER tags it
# ORDINAL and NER results go straight into `mentions` without passing
# through _is_low_information_candidate at all (that check only runs on
# KeyBERT/YAKE candidates). CARDINAL/ORDINAL/PERCENT/QUANTITY/MONEY/TIME
# entities are numbers and units by definition, not domain concepts;
# PERSON/ORG/GPE/PRODUCT/EVENT/WORK_OF_ART etc. are left untouched since
# those genuinely are meaningful entities.
_LOW_VALUE_NER_LABELS = frozenset({"CARDINAL", "ORDINAL", "PERCENT", "QUANTITY", "MONEY", "TIME"})

# Standalone 2-6 letter all-caps tokens (DNA, RNA, PCR, BGC, ...) — almost
# always a meaningful domain acronym in academic/technical text, but
# KeyBERT/YAKE rank by whole-document representativeness, so a short
# acronym routinely loses out to longer, more "salient" multi-word phrases
# for a spot in their capped top-N and never gets extracted at all (found
# via a real gap: "DNA" never became a KG entity despite being literally in
# the ingested document's title, because it didn't survive KeyBERT/YAKE's
# top-10 cut in a 512-token chunk full of more specific technical phrases).
# Extracted with guaranteed priority — before KeyBERT/YAKE, right after
# spaCy NER — so ranking competition can no longer drop it.
_ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,6}\b")


class MentionExtractor:
    """
    Stage 6: runs spaCy NER, an acronym pass, KeyBERT, and YAKE over a
    Chunk's text to capture candidate concept mention spans, per the
    pipeline spec's Phase 3. spaCy NER runs first (most precise, gives a
    candidate_type label, and skips low-value labels — see
    _LOW_VALUE_NER_LABELS), then the acronym pass (see _ACRONYM_PATTERN);
    KeyBERT and YAKE keyphrases fill in domain concepts neither would catch
    (e.g. "gradient descent"), but are skipped wherever they'd overlap a
    span already produced, or where the candidate itself is low-information
    (see _is_low_information_candidate) — so overlapping duplicate mentions
    are never emitted, and neither is content-free noise.
    """

    def __init__(
        self,
        nlp,
        keybert_model,
        yake_extractor,
        *,
        keybert_top_n: int = 10,
        keybert_ngram_range: tuple[int, int] = (1, 3),
    ) -> None:
        self._nlp = nlp
        self._keybert = keybert_model
        self._yake = yake_extractor
        self._keybert_top_n = keybert_top_n
        # KeyBERT defaults to keyphrase_ngram_range=(1, 1) — unigrams only —
        # which was silently fragmenting every multi-word technical phrase it
        # produced into separate single words (e.g. "genetic material" ->
        # "genetic" + "material", each individually a much weaker, more
        # ambiguous match than the phrase together). Widened to match YAKE's
        # own n=3 setting so both extractors consider the same phrase
        # lengths.
        self._keybert_ngram_range = keybert_ngram_range

    def extract(self, chunk: Chunk) -> list[Mention]:
        text = chunk.text
        if not text.strip():
            return []

        mentions: list[Mention] = []
        occupied: list[tuple[int, int]] = []

        doc = self._nlp(text)
        for ent in doc.ents:
            if ent.label_ in _LOW_VALUE_NER_LABELS:
                continue
            span = (ent.start_char, ent.end_char)
            occupied.append(span)
            mentions.append(self._make_mention(chunk, ent.text, span, ent.label_))

        for match in _ACRONYM_PATTERN.finditer(text):
            span = match.span()
            if any(self._overlaps(span, other) for other in occupied):
                continue
            occupied.append(span)
            mentions.append(self._make_mention(chunk, match.group(), span, None))

        for phrase, _score in self._keybert.extract_keywords(
            text, top_n=self._keybert_top_n, keyphrase_ngram_range=self._keybert_ngram_range
        ):
            if self._is_low_information_candidate(phrase):
                continue
            span = self._locate(text, phrase, occupied)
            if span is None:
                continue
            occupied.append(span)
            mentions.append(self._make_mention(chunk, text[span[0] : span[1]], span, None))

        for phrase, _score in self._yake.extract_keywords(text):
            if self._is_low_information_candidate(phrase):
                continue
            span = self._locate(text, phrase, occupied)
            if span is None:
                continue
            occupied.append(span)
            mentions.append(self._make_mention(chunk, text[span[0] : span[1]], span, None))

        logger.info("mentions_extracted", chunk_id=str(chunk.id), mention_count=len(mentions))
        return mentions

    def _make_mention(self, chunk: Chunk, surface_form: str, span: tuple[int, int], candidate_type: str | None) -> Mention:
        return Mention(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            surface_form=surface_form,
            span_start=span[0],
            span_end=span[1],
            candidate_type=candidate_type,
        )

    def _locate(self, text: str, phrase: str, occupied: list[tuple[int, int]]) -> tuple[int, int] | None:
        idx = text.lower().find(phrase.lower())
        if idx == -1:
            return None
        span = (idx, idx + len(phrase))
        if any(self._overlaps(span, other) for other in occupied):
            return None
        return span

    @staticmethod
    def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    @staticmethod
    def _is_low_information_candidate(phrase: str) -> bool:
        normalized = phrase.strip()
        if " " in normalized:
            return False  # multi-word phrases are never rejected, even if a word inside is common
        if normalized.lower() in _STOPWORDS:
            return True
        return bool(_BARE_NUMBER_PATTERN.match(normalized))
