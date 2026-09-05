"""Immutable contracts for later evidence-preserving hybrid retrieval.

These models only hold already-selected channel facts.  They deliberately do
not calculate RRF scores, collapse semantic spans, retrieve data, or detect
channel availability.
"""

from __future__ import annotations

import math
from enum import StrEnum
from math import gcd
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.service import RetrievalHit


class ChunkCandidateIdentity(BaseModel):
    """Occurrence-specific parent-chunk identity shared by both search channels."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_id: str = Field(min_length=1)
    document_generation_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)


class HybridParentChunkContext(BaseModel):
    """Validated parent-chunk context, without relabelling it as semantic evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity: ChunkCandidateIdentity
    paper_id: str = Field(min_length=1)
    file_version_id: str = Field(min_length=1)
    physical_page_index: int = Field(ge=0)
    display_page_number: int = Field(ge=1)
    printed_page_label: str | None
    chunk_ordinal: int = Field(ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    chunk_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.end_offset <= self.start_offset:
            raise ValueError("parent chunk offsets must form a non-empty half-open range")
        if self.display_page_number != self.physical_page_index + 1:
            raise ValueError("display_page_number must equal physical_page_index + 1")
        return self


class HybridLexicalContribution(BaseModel):
    """One lexical packet, preserving the established chunk evidence contract.

    Embedding :class:`RetrievalHit` avoids copying or weakening its exact
    range, text, anchor, rank, raw-BM25, and lineage facts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lexical_hit: RetrievalHit

    @property
    def identity(self) -> ChunkCandidateIdentity:
        """Return the canonical parent identity represented by this packet."""

        hit = self.lexical_hit
        return ChunkCandidateIdentity(
            project_id=hit.project_id,
            document_generation_id=hit.document_generation_id,
            page_id=hit.page_id,
            chunk_id=hit.chunk_id,
        )

    @model_validator(mode="after")
    def _validate_evidence(self) -> Self:
        hit = self.lexical_hit
        if not math.isfinite(hit.raw_bm25_score):
            raise ValueError("raw BM25 score must be finite")
        if hit.end_offset <= hit.start_offset:
            raise ValueError("lexical chunk offsets must form a non-empty half-open range")
        for anchor in hit.anchors:
            if (
                anchor.file_version_id != hit.file_version_id
                or anchor.physical_page_index != hit.physical_page_index
                or anchor.display_page_number != hit.display_page_number
                or anchor.printed_page_label != hit.printed_page_label
                or anchor.char_start < hit.start_offset
                or anchor.char_end > hit.end_offset
            ):
                raise ValueError("lexical anchors must be scoped to the lexical chunk evidence")
            if anchor.canonical_page_text[hit.start_offset : hit.end_offset] != hit.chunk_text:
                raise ValueError("lexical chunk text must equal its canonical page range")
        return self


class HybridSemanticContribution(BaseModel):
    """One future-selected semantic representative, with span-scoped evidence.

    The embedded :class:`SemanticRetrievalHit` preserves the channel's native
    span, profile, vector generation, raw cosine, rank, and anchors.  This
    model does not select the representative or retain extra span votes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    semantic_hit: SemanticRetrievalHit

    @property
    def identity(self) -> ChunkCandidateIdentity:
        """Return the canonical parent identity represented by this span."""

        hit = self.semantic_hit
        return ChunkCandidateIdentity(
            project_id=hit.project_id,
            document_generation_id=hit.document_generation_id,
            page_id=hit.page_id,
            chunk_id=hit.chunk_id,
        )

    @model_validator(mode="after")
    def _validate_evidence(self) -> Self:
        hit = self.semantic_hit
        if not math.isfinite(hit.raw_semantic_score):
            raise ValueError("raw semantic score must be finite")
        if hit.end_offset <= hit.start_offset:
            raise ValueError("semantic span offsets must form a non-empty half-open range")
        for anchor in hit.anchors:
            if (
                anchor.file_version_id != hit.file_version_id
                or anchor.physical_page_index != hit.physical_page_index
                or anchor.display_page_number != hit.display_page_number
                or anchor.printed_page_label != hit.printed_page_label
                or anchor.char_start < hit.start_offset
                or anchor.char_end > hit.end_offset
            ):
                raise ValueError("semantic anchors must be scoped to the semantic span evidence")
            if (
                anchor.canonical_page_text[hit.start_offset : hit.end_offset]
                != hit.embedding_span_text
            ):
                raise ValueError("semantic span text must equal its canonical page range")
        return self


class HybridChannelMembership(StrEnum):
    """Stable explicit membership of the independently preserved channels."""

    LEXICAL_ONLY = "lexical_only"
    SEMANTIC_ONLY = "semantic_only"
    BOTH = "both"


class HybridChannelState(StrEnum):
    """Successful channel states; unavailable, stale, and invalid remain errors."""

    HEALTHY_RESULTS = "healthy_results"
    HEALTHY_EMPTY = "healthy_empty"


class ExactRationalScore(BaseModel):
    """Canonical exact score payload for deterministic RRF ordering.

    This value object deliberately stores a precomputed reduced fraction.  It
    does not calculate a score from ranks; Task 2 owns that behavior.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    numerator: int = Field(gt=0)
    denominator: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_reduced_fraction(self) -> Self:
        if gcd(self.numerator, self.denominator) != 1:
            raise ValueError("exact fusion score must be a reduced fraction")
        return self


class HybridRankingTrace(BaseModel):
    """Auditable ranking facts; the fusion score is a ranking signal only.

    ``fusion_score`` is neither a probability nor a confidence value.  Raw
    channel scores remain in their separately labelled contribution packets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fusion_profile_id: str = Field(min_length=1)
    fusion_score: ExactRationalScore = Field(
        description="Exact RRF ranking signal only; not probability or confidence."
    )
    fusion_rank: int = Field(ge=1)
    channel_membership: HybridChannelMembership
    lexical_rank: int | None = Field(default=None, ge=1)
    semantic_rank: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_membership_ranks(self) -> Self:
        has_lexical = self.lexical_rank is not None
        has_semantic = self.semantic_rank is not None
        expected = _membership_for(has_lexical=has_lexical, has_semantic=has_semantic)
        if self.channel_membership is not expected:
            raise ValueError("channel membership must exactly match the supplied channel ranks")
        return self


class HybridRetrievalHit(BaseModel):
    """One fused parent-chunk candidate with separately labelled support packets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity: ChunkCandidateIdentity
    parent_chunk: HybridParentChunkContext
    lexical_contribution: HybridLexicalContribution | None
    semantic_contribution: HybridSemanticContribution | None
    trace: HybridRankingTrace

    @property
    def channel_membership(self) -> HybridChannelMembership:
        """Expose the validated trace membership without score-sentinel logic."""

        return self.trace.channel_membership

    @model_validator(mode="after")
    def _validate_candidate(self) -> Self:
        lexical = self.lexical_contribution
        semantic = self.semantic_contribution
        if lexical is None and semantic is None:
            raise ValueError("a hybrid retrieval hit requires at least one channel contribution")
        if self.parent_chunk.identity != self.identity:
            raise ValueError("parent chunk context must use the hybrid candidate identity")

        expected_membership = _membership_for(
            has_lexical=lexical is not None,
            has_semantic=semantic is not None,
        )
        if self.trace.channel_membership is not expected_membership:
            raise ValueError("ranking trace membership must match channel contributions")

        if lexical is not None:
            if lexical.identity != self.identity:
                raise ValueError("lexical contribution must belong to the hybrid candidate")
            if self.trace.lexical_rank != lexical.lexical_hit.rank:
                raise ValueError(
                    "ranking trace lexical rank must equal the lexical contribution rank"
                )
            _validate_parent_against_lexical(self.parent_chunk, lexical.lexical_hit)
        elif self.trace.lexical_rank is not None:
            raise ValueError("ranking trace cannot include a lexical rank without lexical evidence")

        if semantic is not None:
            if semantic.identity != self.identity:
                raise ValueError("semantic contribution must belong to the hybrid candidate")
            if self.trace.semantic_rank != semantic.semantic_hit.rank:
                raise ValueError(
                    "ranking trace semantic rank must equal the semantic contribution rank"
                )
            _validate_parent_against_semantic(self.parent_chunk, semantic.semantic_hit)
        elif self.trace.semantic_rank is not None:
            raise ValueError(
                "ranking trace cannot include a semantic rank without semantic evidence"
            )
        return self


class HybridRetrievalResults(BaseModel):
    """Immutable successful hybrid results for one project and fusion profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_id: str = Field(min_length=1)
    query: str
    fusion_profile_id: str = Field(min_length=1)
    lexical_state: HybridChannelState
    semantic_state: HybridChannelState
    hits: tuple[HybridRetrievalHit, ...]

    @model_validator(mode="after")
    def _validate_hit_scope(self) -> Self:
        for hit in self.hits:
            if hit.identity.project_id != self.project_id:
                raise ValueError("hybrid result hit must belong to the requested project")
            if hit.trace.fusion_profile_id != self.fusion_profile_id:
                raise ValueError("hybrid result hit must use the requested fusion profile")
        return self


HybridCandidateKey = ChunkCandidateIdentity
"""Compatibility spelling for the canonical parent-chunk identity."""

HybridRetrievalResult = HybridRetrievalHit
"""Singular-result spelling for :class:`HybridRetrievalHit`."""


def _membership_for(*, has_lexical: bool, has_semantic: bool) -> HybridChannelMembership:
    if has_lexical and has_semantic:
        return HybridChannelMembership.BOTH
    if has_lexical:
        return HybridChannelMembership.LEXICAL_ONLY
    if has_semantic:
        return HybridChannelMembership.SEMANTIC_ONLY
    raise ValueError("channel membership requires at least one contribution")


def _validate_parent_against_lexical(
    parent: HybridParentChunkContext, hit: RetrievalHit
) -> None:
    if (
        parent.paper_id != hit.paper_id
        or parent.file_version_id != hit.file_version_id
        or parent.physical_page_index != hit.physical_page_index
        or parent.display_page_number != hit.display_page_number
        or parent.printed_page_label != hit.printed_page_label
        or parent.chunk_ordinal != hit.chunk_ordinal
        or parent.start_offset != hit.start_offset
        or parent.end_offset != hit.end_offset
        or parent.chunk_text != hit.chunk_text
    ):
        raise ValueError("parent chunk context must exactly match lexical chunk evidence")


def _validate_parent_against_semantic(
    parent: HybridParentChunkContext, hit: SemanticRetrievalHit
) -> None:
    if (
        parent.paper_id != hit.paper_id
        or parent.file_version_id != hit.file_version_id
        or parent.physical_page_index != hit.physical_page_index
        or parent.display_page_number != hit.display_page_number
        or parent.printed_page_label != hit.printed_page_label
    ):
        raise ValueError("parent chunk context must match semantic span lineage")
    if not parent.start_offset <= hit.start_offset < hit.end_offset <= parent.end_offset:
        raise ValueError("semantic span must be within the parent chunk context")
    relative_start = hit.start_offset - parent.start_offset
    relative_end = hit.end_offset - parent.start_offset
    if parent.chunk_text[relative_start:relative_end] != hit.embedding_span_text:
        raise ValueError("semantic span text must match the parent chunk context range")
