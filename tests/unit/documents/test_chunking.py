from __future__ import annotations

from itertools import pairwise

from academic_chatbot.documents.chunking import (
    LEXICAL_CHUNK_PROFILE_ID,
    LexicalChunk,
    chunk_canonical_page,
)
from academic_chatbot.documents.models import CanonicalPdfWord


def _canonical_words(texts: list[str]) -> tuple[str, tuple[CanonicalPdfWord, ...]]:
    text = " ".join(texts)
    cursor = 0
    words: list[CanonicalPdfWord] = []
    for index, word in enumerate(texts):
        start = cursor
        end = start + len(word)
        words.append(
            CanonicalPdfWord(
                text=word,
                char_start=start,
                char_end=end,
                x0=float(index * 10),
                top=0.0,
                x1=float(index * 10 + 9),
                bottom=9.0,
            )
        )
        cursor = end + 1
    return text, tuple(words)


def _chunks(texts: list[str], *, page_id: str = "page-1") -> tuple[LexicalChunk, ...]:
    text, words = _canonical_words(texts)
    return chunk_canonical_page(
        document_generation_id="generation-1",
        page_id=page_id,
        canonical_text=text,
        canonical_words=words,
    )


def test_nonfinal_120_word_range_without_tail_punctuation_cuts_at_exactly_120_words() -> None:
    """Would fail if an older punctuation mark or an arbitrary boundary shrank the chunk."""
    words = [f"w{index:03d}" for index in range(1, 122)]
    chunks = _chunks(words)

    assert len(chunks) == 2
    assert chunks[0].text == " ".join(words[:120])
    assert (chunks[0].start_offset, chunks[0].end_offset) == (0, len(chunks[0].text))
    assert chunks[0].lexical_word_count == 120
    assert chunks[1].text == "w121"


def test_last_sentence_like_boundary_in_the_90_to_120_tail_is_selected() -> None:
    """Would fail if the preferred tail did not choose its last eligible punctuation boundary."""
    words = [f"w{index:03d}" for index in range(1, 131)]
    words[96] = 'first."'
    words[112] = "last!)"
    chunks = _chunks(words)

    assert chunks[0].text == " ".join(words[:113])
    assert chunks[0].lexical_word_count == 113
    assert chunks[1].text == " ".join(words[113:])


def test_punctuation_before_word_90_does_not_force_an_early_nonfinal_chunk() -> None:
    """Would fail if the implementation searched the full 120-word candidate window."""
    words = [f"w{index:03d}" for index in range(1, 122)]
    words[79] = "old."
    chunks = _chunks(words)

    assert chunks[0].text == " ".join(words[:120])
    assert chunks[0].lexical_word_count == 120


def test_final_remainder_below_90_words_is_preserved() -> None:
    """Would fail if the minimum preferred boundary incorrectly discarded a short final range."""
    words = [f"w{index:03d}" for index in range(1, 141)]
    chunks = _chunks(words)

    assert [chunk.lexical_word_count for chunk in chunks] == [120, 20]
    assert chunks[1].text == " ".join(words[120:])


def test_repeated_content_uses_exact_constructed_offsets_and_distinct_chunk_identities() -> None:
    """Would fail if repeated canonical text were bound through a first-occurrence search."""
    words = ["Repeat."] * 121
    text, _ = _canonical_words(words)
    chunks = _chunks(words)

    assert chunks[0].text == text[chunks[0].start_offset : chunks[0].end_offset]
    assert chunks[1].text == text[chunks[1].start_offset : chunks[1].end_offset]
    assert (chunks[0].start_offset, chunks[0].end_offset) == (0, len(" ".join(words[:120])))
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_chunks_are_page_bounded_non_overlapping_and_identical_across_reruns() -> None:
    """Would fail if chunks crossed pages, overlapped, or depended on mutable runtime state."""
    first = _chunks([f"a{index:03d}" for index in range(1, 122)], page_id="page-1")
    second = _chunks([f"b{index:03d}" for index in range(1, 122)], page_id="page-2")
    repeated = _chunks([f"a{index:03d}" for index in range(1, 122)], page_id="page-1")

    assert all(chunk.page_id == "page-1" for chunk in first)
    assert all(chunk.page_id == "page-2" for chunk in second)
    assert all(left.end_offset < right.start_offset for left, right in pairwise(first))
    assert first == repeated
    assert [chunk.chunk_id for chunk in first] != [chunk.chunk_id for chunk in second]
    assert {chunk.processing_profile_id for chunk in first} == {LEXICAL_CHUNK_PROFILE_ID}
