from __future__ import annotations

import re
import unicodedata

_MULTI_WHITESPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Lines that are document furniture (page/slide numbers, chapter headers,
# "Prepared by: ..." bylines, a standalone institution name used as a
# header) rather than course content — these were flowing straight into
# chunk text and then into entity/relation extraction as noise. Found via a
# cross-check against a teammate's independently-built codebase, which had
# already hit and fixed this; this version is written fresh for our own
# style rather than copied, and deliberately narrower in one place: a line
# is metadata either because it starts with an explicit label (page/
# chapter/author/etc — those are structurally never prose) or, for the
# broader "mentions an institution name" case, only when it's ALSO short
# and has no sentence-ending punctuation — a genuine sentence that happens
# to mention a university ("...researchers at MIT found...") must not be
# silently deleted just because it contains that word.
_METADATA_LABEL_PREFIX = re.compile(
    r"^\s*(?:page|slide|p\.)\s*\d+(?:\s*(?:of|/)\s*\d+)?\s*$"
    r"|^\s*(?:chapter|section|unit|module|part)\s+\d+(?:\.\d+)*(?:\s*[:.\-].*)?$"
    r"|^\s*(?:author|instructor|lecturer|faculty|prepared\s+by|submitted\s+by)\s*[:\-].*$",
    re.IGNORECASE,
)
_INSTITUTION_MENTION = re.compile(r"\b(?:college|university|institute\s+of\s+technology)\b", re.IGNORECASE)
_SENTENCE_PUNCTUATION = re.compile(r"[.!?]")
_MAX_HEADER_LINE_LENGTH = 120


class TextCleaner:
    """Normalizes raw extracted text (from PDF/DOCX/OCR) into clean, chunkable prose.
    Pure function transform — no I/O, fully unit-testable in isolation."""

    def clean(self, text: str) -> str:
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = _CONTROL_CHARS.sub("", normalized)
        normalized = _HYPHEN_LINEBREAK.sub(r"\1\2", normalized)  # de-hyphenate line-wrapped words
        normalized = "\n".join(line for line in normalized.splitlines() if not self._is_metadata_line(line))
        normalized = _MULTI_WHITESPACE.sub(" ", normalized)
        normalized = _MULTI_NEWLINE.sub("\n\n", normalized)
        lines = [line.strip() for line in normalized.split("\n")]
        return "\n".join(lines).strip()

    @staticmethod
    def _is_metadata_line(line: str) -> bool:
        if _METADATA_LABEL_PREFIX.match(line):
            return True
        stripped = line.strip()
        if not _INSTITUTION_MENTION.search(stripped):
            return False
        return len(stripped) <= _MAX_HEADER_LINE_LENGTH and not _SENTENCE_PUNCTUATION.search(stripped)
