from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import pytest

from academic_chatbot.embeddings.models import EmbeddingProfile, EmbeddingRole
from academic_chatbot.embeddings.repository import EmbeddingSpanStatus
from academic_chatbot.embeddings.spans import (
    CanonicalWordRange,
    ChunkSourceEvidence,
    SpanConstructionError,
    construct_document_spans,
)
from academic_chatbot.embeddings.tokenizer import EmbeddingInputTooLongError
from tests.unit.embeddings.conftest import frozen_profile_payload as _frozen_profile_payload


@dataclass(frozen=True)
class _BudgetTokenizer:
    """A real boundary-shaped fake: only the token budget is replaced."""

    maximum_words: int

    def prepare(self, role: EmbeddingRole, texts: tuple[str, ...]) -> object:
        assert role is EmbeddingRole.DOCUMENT
        for text in texts:
            if len(text.split(" ")) > self.maximum_words:
                raise EmbeddingInputTooLongError("document exceeds embedding profile token budget")
        return object()


def _profile() -> EmbeddingProfile:
    return EmbeddingProfile.model_validate(_frozen_profile_payload.__wrapped__())  # type: ignore[attr-defined]


def _chunk(*words: str, start: int = 0, chunk_id: str = "chunk-one") -> ChunkSourceEvidence:
    page_text = "prefix " + " ".join(words) + " suffix"
    chunk_start = len("prefix ") + start
    ranges: list[CanonicalWordRange] = []
    cursor = chunk_start
    for word in words:
        ranges.append(
            CanonicalWordRange(text=word, start_offset=cursor, end_offset=cursor + len(word))
        )
        cursor += len(word) + 1
    return ChunkSourceEvidence(
        document_generation_id="generation-one",
        chunk_id=chunk_id,
        page_id="page-one",
        page_text=page_text,
        chunk_ordinal=0,
        chunk_start_offset=chunk_start,
        chunk_end_offset=ranges[-1].end_offset,
        words=tuple(ranges),
    )


def test_constructs_one_exact_span_when_the_chunk_fits() -> None:
    source = _chunk("alpha", "beta", "gamma")

    spans = construct_document_spans(
        source=source, profile=_profile(), tokenizer=_BudgetTokenizer(maximum_words=3)
    )

    assert len(spans) == 1
    span = spans[0]
    assert span.status is EmbeddingSpanStatus.EMBEDDABLE
    assert (span.identity.start_offset, span.identity.end_offset) == (
        source.chunk_start_offset,
        source.chunk_end_offset,
    )
    assert (
        source.page_text[span.identity.start_offset : span.identity.end_offset]
        == "alpha beta gamma"
    )


def test_splits_contiguously_at_the_last_tokenizer_valid_word() -> None:
    source = _chunk("one", "two", "three", "four", "five")

    spans = construct_document_spans(
        source=source, profile=_profile(), tokenizer=_BudgetTokenizer(maximum_words=2)
    )

    assert [(span.identity.start_offset, span.identity.end_offset) for span in spans] == [
        (source.words[0].start_offset, source.words[1].end_offset),
        (source.words[2].start_offset, source.words[3].end_offset),
        (source.words[4].start_offset, source.words[4].end_offset),
    ]
    assert all(span.status is EmbeddingSpanStatus.EMBEDDABLE for span in spans)
    assert all(
        left.identity.end_offset < right.identity.start_offset
        for left, right in pairwise(spans)
    )


def test_repeated_occurrences_keep_distinct_exact_embedding_span_ids() -> None:
    source = _chunk("repeat", "repeat")

    spans = construct_document_spans(
        source=source, profile=_profile(), tokenizer=_BudgetTokenizer(maximum_words=1)
    )

    assert [
        source.page_text[item.identity.start_offset : item.identity.end_offset] for item in spans
    ] == [
        "repeat",
        "repeat",
    ]
    assert spans[0].embedding_span_id != spans[1].embedding_span_id


def test_preserves_unicode_urls_and_identifier_heavy_source_slices() -> None:
    source = _chunk("naïve", "東京", "https://example.org/x?y=1", "id_42")

    spans = construct_document_spans(
        source=source, profile=_profile(), tokenizer=_BudgetTokenizer(maximum_words=2)
    )

    assert [
        source.page_text[item.identity.start_offset : item.identity.end_offset] for item in spans
    ] == ["naïve 東京", "https://example.org/x?y=1 id_42"]


def test_persists_an_exact_excluded_span_for_one_overbudget_word() -> None:
    source = _chunk("oversized")

    spans = construct_document_spans(
        source=source, profile=_profile(), tokenizer=_BudgetTokenizer(maximum_words=0)
    )

    assert len(spans) == 1
    assert spans[0].status is EmbeddingSpanStatus.EXCLUDED_UNEMBEDDABLE
    assert (
        source.page_text[spans[0].identity.start_offset : spans[0].identity.end_offset]
        == "oversized"
    )


def test_rejects_noncontiguous_or_outside_persisted_anchor_ranges() -> None:
    source = _chunk("alpha", "beta")
    with pytest.raises(SpanConstructionError, match="canonical"):
        ChunkSourceEvidence(
            document_generation_id=source.document_generation_id,
            chunk_id=source.chunk_id,
            page_id=source.page_id,
            page_text=source.page_text,
            chunk_ordinal=source.chunk_ordinal,
            chunk_start_offset=source.chunk_start_offset,
            chunk_end_offset=source.chunk_end_offset,
            words=(
                source.words[0],
                CanonicalWordRange(
                    text="beta",
                    start_offset=source.words[1].start_offset + 1,
                    end_offset=source.words[1].end_offset + 1,
                ),
            ),
        )
