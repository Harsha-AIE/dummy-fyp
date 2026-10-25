from pathlib import Path
from uuid import uuid4

from edukg.core.config import ChunkingSettings
from edukg.core.models.enums import ExtractionMethod
from edukg.modules.ingestion.chunker import ChunkGenerator
from edukg.modules.ingestion.hashing import compute_file_hash
from edukg.modules.ingestion.text_cleaner import TextCleaner


def test_text_cleaner_dehyphenates_and_normalizes_whitespace() -> None:
    cleaner = TextCleaner()
    raw = "This is a hyphen-\nated word.   Multiple   spaces.\n\n\n\nToo many newlines."
    cleaned = cleaner.clean(raw)
    assert "hyphenated" in cleaned
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


def test_text_cleaner_handles_empty_input() -> None:
    assert TextCleaner().clean("") == ""
    assert TextCleaner().clean(None or "") == ""


def test_text_cleaner_strips_labeled_metadata_lines() -> None:
    # Regression test: page/slide numbers, chapter headers, and "Prepared
    # by:" bylines used to flow straight into chunk text and then into
    # entity/relation extraction as noise (e.g. "Prepared by" -> a PERSON
    # mention, "Page 3 of 40" -> spurious numeric candidates).
    raw = "Page 3 of 40\nChapter 2: Neural Networks\nPrepared by: Dr. Jane Smith\nBackpropagation computes gradients."
    cleaned = TextCleaner().clean(raw)
    assert "Page 3 of 40" not in cleaned
    assert "Chapter 2" not in cleaned
    assert "Prepared by" not in cleaned
    assert "Backpropagation computes gradients." in cleaned


def test_text_cleaner_strips_a_short_standalone_institution_header_line() -> None:
    raw = "XYZ Institute of Technology\nIntroduction to Machine Learning\nSupervised learning uses labeled data."
    cleaned = TextCleaner().clean(raw)
    assert "XYZ Institute of Technology" not in cleaned
    assert "Supervised learning uses labeled data." in cleaned


def test_text_cleaner_keeps_a_real_sentence_that_happens_to_mention_a_university() -> None:
    # The institution-mention check must not delete real prose just because
    # it contains "university"/"college" — only a short, punctuation-free
    # standalone header line is metadata; a full sentence is content.
    raw = "Researchers at Stanford University published a widely-cited study on this topic in 2021."
    cleaned = TextCleaner().clean(raw)
    assert "Stanford University" in cleaned
    assert "widely-cited study" in cleaned


def test_chunk_generator_respects_page_boundaries() -> None:
    generator = ChunkGenerator(ChunkingSettings(chunk_size_tokens=50, chunk_overlap_tokens=5))
    document_id = uuid4()
    page_texts = [
        (0, "Alpha concept explanation. " * 20, ExtractionMethod.NATIVE_TEXT),
        (1, "Beta concept explanation. " * 20, ExtractionMethod.NATIVE_TEXT),
    ]
    chunks = generator.generate(document_id, page_texts)

    assert len(chunks) > 2
    assert all(c.document_id == document_id for c in chunks)
    page0_chunks = [c for c in chunks if c.page_number == 0]
    page1_chunks = [c for c in chunks if c.page_number == 1]
    assert all("Alpha" in c.text for c in page0_chunks)
    assert all("Beta" in c.text for c in page1_chunks)


def test_chunk_generator_skips_blank_pages() -> None:
    generator = ChunkGenerator()
    chunks = generator.generate(uuid4(), [(0, "   ", ExtractionMethod.NATIVE_TEXT)])
    assert chunks == []


def test_chunk_sequence_index_is_monotonic() -> None:
    generator = ChunkGenerator(ChunkingSettings(chunk_size_tokens=30, chunk_overlap_tokens=5))
    chunks = generator.generate(uuid4(), [(0, "word " * 200, ExtractionMethod.NATIVE_TEXT)])
    indices = [c.sequence_index for c in chunks]
    assert indices == sorted(indices)
    assert indices == list(range(len(chunks)))


def test_compute_file_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("hello world")
    file_b.write_text("hello world!")

    hash_a1 = compute_file_hash(file_a)
    hash_a2 = compute_file_hash(file_a)
    hash_b = compute_file_hash(file_b)

    assert hash_a1 == hash_a2
    assert hash_a1 != hash_b
    assert len(hash_a1) == 64
