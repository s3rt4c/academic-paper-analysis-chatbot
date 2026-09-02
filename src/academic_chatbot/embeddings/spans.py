"""Construction-derived semantic spans from immutable canonical word anchors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from academic_chatbot.embeddings.models import (
    FROZEN_SPAN_POLICY_ID,
    EmbeddingProfile,
    EmbeddingRole,
    EmbeddingSpanIdentity,
)
from academic_chatbot.embeddings.repository import (
    EmbeddingSpan,
    EmbeddingSpanStatus,
)
from academic_chatbot.embeddings.tokenizer import EmbeddingInputTooLongError


class SpanConstructionError(ValueError):
    """Raised when durable canonical evidence cannot support an exact span."""


class _DocumentTokenizer(Protocol):
    def prepare(self, role: EmbeddingRole, texts: Sequence[str]) -> object: ...


@dataclass(frozen=True, slots=True)
class CanonicalWordRange:
    """One persisted canonical-word range, recovered from one page anchor."""

    text: str
    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        if not self.text or self.start_offset < 0 or self.end_offset <= self.start_offset:
            raise SpanConstructionError("canonical word ranges must be non-empty")
        if self.end_offset - self.start_offset != len(self.text):
            raise SpanConstructionError("canonical word range does not match its text")


@dataclass(frozen=True, slots=True)
class ChunkSourceEvidence:
    """One lexical chunk plus its exact, ordered persisted word anchors."""

    document_generation_id: str
    chunk_id: str
    page_id: str
    page_text: str
    chunk_ordinal: int
    chunk_start_offset: int
    chunk_end_offset: int
    words: tuple[CanonicalWordRange, ...]

    def __post_init__(self) -> None:
        if (
            not self.document_generation_id
            or not self.chunk_id
            or not self.page_id
            or self.chunk_ordinal < 0
            or self.chunk_start_offset < 0
            or self.chunk_end_offset <= self.chunk_start_offset
            or self.chunk_end_offset > len(self.page_text)
            or not self.words
        ):
            raise SpanConstructionError("chunk evidence is incomplete")
        cursor = self.chunk_start_offset
        for index, word in enumerate(self.words):
            if word.start_offset != cursor or word.end_offset > self.chunk_end_offset:
                raise SpanConstructionError("canonical word ranges are not contiguous in the chunk")
            if self.page_text[word.start_offset : word.end_offset] != word.text:
                raise SpanConstructionError(
                    "canonical word text does not match the page source slice"
                )
            cursor = word.end_offset + (1 if index < len(self.words) - 1 else 0)
        if self.words[-1].end_offset != self.chunk_end_offset:
            raise SpanConstructionError("canonical words do not cover the full chunk range")
        if self.page_text[self.chunk_start_offset : self.chunk_end_offset] != " ".join(
            word.text for word in self.words
        ):
            raise SpanConstructionError("chunk source is not the canonical word construction slice")


def construct_document_spans(
    *,
    source: ChunkSourceEvidence,
    profile: EmbeddingProfile,
    tokenizer: _DocumentTokenizer,
) -> tuple[EmbeddingSpan, ...]:
    """Greedily split one lexical chunk without ever rediscovering text offsets.

    The tokenizer is used only to judge whether the exact source slice can be
    embedded.  Offset boundaries always originate from persisted canonical
    words, never tokenizer offsets or text search.
    """

    if profile.span_policy != FROZEN_SPAN_POLICY_ID:
        raise SpanConstructionError("embedding profile does not use the frozen span policy")

    spans: list[EmbeddingSpan] = []
    start_index = 0
    while start_index < len(source.words):
        end_index = start_index
        while end_index < len(source.words):
            candidate = _source_slice(source, start_index, end_index + 1)
            try:
                tokenizer.prepare(EmbeddingRole.DOCUMENT, (candidate,))
            except EmbeddingInputTooLongError:
                break
            end_index += 1

        if end_index == start_index:
            # The exact, indivisible canonical word is coverage, not a vector.
            spans.append(
                _span(
                    source=source,
                    profile=profile,
                    start_index=start_index,
                    end_index=start_index + 1,
                    status=EmbeddingSpanStatus.EXCLUDED_UNEMBEDDABLE,
                )
            )
            start_index += 1
            continue

        spans.append(
            _span(
                source=source,
                profile=profile,
                start_index=start_index,
                end_index=end_index,
                status=EmbeddingSpanStatus.EMBEDDABLE,
            )
        )
        start_index = end_index
    return tuple(spans)


def source_text_for_span(*, source: ChunkSourceEvidence, span: EmbeddingSpan) -> str:
    """Return a checked exact source slice for a span constructed from *source*."""

    identity = span.identity
    if (
        identity.document_generation_id != source.document_generation_id
        or identity.chunk_id != source.chunk_id
        or identity.page_id != source.page_id
        or identity.start_offset < source.chunk_start_offset
        or identity.end_offset > source.chunk_end_offset
    ):
        raise SpanConstructionError("embedding span does not belong to the chunk source evidence")
    text = source.page_text[identity.start_offset : identity.end_offset]
    if not text:
        raise SpanConstructionError("embedding span has no exact source text")
    return text


def _source_slice(source: ChunkSourceEvidence, start_index: int, end_index: int) -> str:
    start_offset = source.words[start_index].start_offset
    end_offset = source.words[end_index - 1].end_offset
    return source.page_text[start_offset:end_offset]


def _span(
    *,
    source: ChunkSourceEvidence,
    profile: EmbeddingProfile,
    start_index: int,
    end_index: int,
    status: EmbeddingSpanStatus,
) -> EmbeddingSpan:
    identity = EmbeddingSpanIdentity(
        document_generation_id=source.document_generation_id,
        chunk_id=source.chunk_id,
        page_id=source.page_id,
        start_offset=source.words[start_index].start_offset,
        end_offset=source.words[end_index - 1].end_offset,
        embedding_profile_id=profile.embedding_profile_id,
    )
    # Make offset integrity explicit at the production boundary too.
    if not source_text_for_span(source=source, span=EmbeddingSpan(identity, status)):
        raise SpanConstructionError("embedding span source slice is empty")
    return EmbeddingSpan(identity=identity, status=status)
