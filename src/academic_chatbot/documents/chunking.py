"""Deterministic page-scoped lexical chunks for the Phase 1A FTS pipeline."""

from __future__ import annotations

import hashlib
import json
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from academic_chatbot.documents.models import CanonicalPdfWord

LEXICAL_CHUNK_PROFILE_ID = "lexical-chunk-v1"
MAX_WORDS = 120
PREFERRED_BOUNDARY_MIN_WORDS = 90
OVERLAP_WORDS = 0
_SENTENCE_LIKE_SUFFIXES = (".", "!", "?", '."', '?"', '!"', ".)", "?)", "!)")


class LexicalChunkError(ValueError):
    """Raised when canonical page data cannot produce exact lexical chunks."""


class LexicalChunk(BaseModel):
    """One contiguous canonical page-text range ready for FTS publication."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(pattern=r"^chunk-sha256-[0-9a-f]{64}$")
    document_generation_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    text: str = Field(min_length=1)
    lexical_word_count: int = Field(ge=1, le=MAX_WORDS)
    processing_profile_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.end_offset <= self.start_offset:
            raise ValueError("chunk offsets must form a non-empty half-open range")
        if self.lexical_word_count != len(self.text.split(" ")):
            raise ValueError("lexical word count must equal the exact chunk text word count")
        return self


def chunk_canonical_page(
    *,
    document_generation_id: str,
    page_id: str,
    canonical_text: str,
    canonical_words: tuple[CanonicalPdfWord, ...],
    processing_profile_id: str = LEXICAL_CHUNK_PROFILE_ID,
) -> tuple[LexicalChunk, ...]:
    """Construct non-overlapping chunks from existing Task 3 word ranges only.

    ``max_words`` is a deterministic processing-profile choice, not a resource
    safety limit.  The function never normalizes text or searches it to recover
    offsets; each range is copied directly from canonical word construction.
    """

    _validate_canonical_words(canonical_text, canonical_words)
    chunks: list[LexicalChunk] = []
    start_index = 0
    while start_index < len(canonical_words):
        candidate_end = min(start_index + MAX_WORDS, len(canonical_words))
        end_index = candidate_end
        if candidate_end < len(canonical_words):
            tail_start = start_index + PREFERRED_BOUNDARY_MIN_WORDS - 1
            for index in range(candidate_end - 1, tail_start - 1, -1):
                if _is_sentence_like_boundary(canonical_words[index].text):
                    end_index = index + 1
                    break
        start_offset = canonical_words[start_index].char_start
        end_offset = canonical_words[end_index - 1].char_end
        text = canonical_text[start_offset:end_offset]
        chunks.append(
            LexicalChunk(
                chunk_id=_chunk_id(
                    document_generation_id=document_generation_id,
                    page_id=page_id,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    processing_profile_id=processing_profile_id,
                ),
                document_generation_id=document_generation_id,
                page_id=page_id,
                ordinal=len(chunks),
                start_offset=start_offset,
                end_offset=end_offset,
                text=text,
                lexical_word_count=end_index - start_index,
                processing_profile_id=processing_profile_id,
            )
        )
        start_index = end_index
    return tuple(chunks)


def _validate_canonical_words(
    canonical_text: str, canonical_words: tuple[CanonicalPdfWord, ...]
) -> None:
    if not canonical_words:
        if canonical_text:
            raise LexicalChunkError("canonical text requires canonical words")
        return
    cursor = 0
    for index, word in enumerate(canonical_words):
        if (
            word.char_start != cursor
            or word.char_end != cursor + len(word.text)
            or canonical_text[word.char_start : word.char_end] != word.text
        ):
            raise LexicalChunkError("canonical words must retain Task 3 construction offsets")
        cursor = word.char_end + (1 if index < len(canonical_words) - 1 else 0)
    if canonical_text != " ".join(word.text for word in canonical_words):
        raise LexicalChunkError("canonical text must remain unchanged from Task 3")


def _is_sentence_like_boundary(text: str) -> bool:
    return text.endswith(_SENTENCE_LIKE_SUFFIXES)


def _chunk_id(
    *,
    document_generation_id: str,
    page_id: str,
    start_offset: int,
    end_offset: int,
    processing_profile_id: str,
) -> str:
    payload = {
        "document_generation_id": document_generation_id,
        "page_id": page_id,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "processing_profile_id": processing_profile_id,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return "chunk-sha256-" + hashlib.sha256(encoded).hexdigest()
