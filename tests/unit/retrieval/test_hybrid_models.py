"""Contract tests for immutable, evidence-preserving hybrid result models."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from academic_chatbot.documents.native_pdf import build_native_pdf_anchor
from academic_chatbot.documents.normalization import canonicalize_extracted_words
from academic_chatbot.retrieval.hybrid_models import (
    ChunkCandidateIdentity,
    ExactRationalScore,
    HybridChannelMembership,
    HybridChannelState,
    HybridLexicalContribution,
    HybridParentChunkContext,
    HybridRankingTrace,
    HybridRetrievalHit,
    HybridRetrievalResults,
    HybridSemanticContribution,
)
from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.service import RetrievalHit


def _anchors() -> tuple[object, object, object]:
    canonical = canonicalize_extracted_words(
        (
            {"text": "alpha", "x0": 1.0, "top": 1.0, "x1": 10.0, "bottom": 5.0},
            {"text": "beta", "x0": 11.0, "top": 1.0, "x1": 20.0, "bottom": 5.0},
            {"text": "gamma", "x0": 21.0, "top": 1.0, "x1": 30.0, "bottom": 5.0},
        ),
        page_width_points=100.0,
        page_height_points=100.0,
    )
    return tuple(
        build_native_pdf_anchor(
            file_version_id="file-version-1",
            source_pdf_sha256="a" * 64,
            physical_page_index=0,
            page_width_points=100.0,
            page_height_points=100.0,
            source_page_rotation_degrees=0,
            canonical_page=canonical,
            word=word,
        )
        for word in canonical.words
    )  # type: ignore[return-value]


def _identity(*, chunk_id: str = "chunk-1") -> ChunkCandidateIdentity:
    return ChunkCandidateIdentity(
        project_id="project-1",
        document_generation_id="generation-1",
        page_id="page-1",
        chunk_id=chunk_id,
    )


def _parent(*, chunk_id: str = "chunk-1") -> HybridParentChunkContext:
    return HybridParentChunkContext(
        identity=_identity(chunk_id=chunk_id),
        paper_id="paper-1",
        file_version_id="file-version-1",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_ordinal=0,
        start_offset=0,
        end_offset=16,
        chunk_text="alpha beta gamma",
    )


def _lexical_hit(*, rank: int = 2, raw_score: float = -3.25) -> RetrievalHit:
    return RetrievalHit(
        project_id="project-1",
        paper_id="paper-1",
        file_version_id="file-version-1",
        document_generation_id="generation-1",
        page_id="page-1",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_id="chunk-1",
        chunk_ordinal=0,
        chunk_text="alpha beta gamma",
        start_offset=0,
        end_offset=16,
        rank=rank,
        raw_bm25_score=raw_score,
        anchors=_anchors(),
    )


def _semantic_hit(*, rank: int = 3, raw_score: float = 0.75) -> SemanticRetrievalHit:
    return SemanticRetrievalHit(
        project_id="project-1",
        paper_id="paper-1",
        file_version_id="file-version-1",
        document_generation_id="generation-1",
        page_id="page-1",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_id="chunk-1",
        embedding_span_id="embedding-span-1",
        embedding_profile_id="embedding-profile-1",
        vector_generation_id="vector-generation-1",
        start_offset=6,
        end_offset=10,
        embedding_span_text="beta",
        rank=rank,
        raw_semantic_score=raw_score,
        anchors=(_anchors()[1],),
    )


def _trace(
    membership: HybridChannelMembership,
    *,
    lexical_rank: int | None = None,
    semantic_rank: int | None = None,
) -> HybridRankingTrace:
    return HybridRankingTrace(
        fusion_profile_id="rrf-v1",
        fusion_score=ExactRationalScore(numerator=1, denominator=41),
        fusion_rank=1,
        channel_membership=membership,
        lexical_rank=lexical_rank,
        semantic_rank=semantic_rank,
    )


def test_candidate_identity_is_hashable_and_occurrence_specific() -> None:
    first = _identity()
    same_occurrence = _identity()
    repeated_text_elsewhere = _identity(chunk_id="chunk-2")

    assert first == same_occurrence
    assert hash(first) == hash(same_occurrence)
    assert first != repeated_text_elsewhere
    assert {first, same_occurrence, repeated_text_elsewhere} == {first, repeated_text_elsewhere}


def test_lexical_only_hit_preserves_parent_chunk_evidence_and_raw_bm25() -> None:
    lexical = HybridLexicalContribution(lexical_hit=_lexical_hit())
    result = HybridRetrievalHit(
        identity=_identity(),
        parent_chunk=_parent(),
        lexical_contribution=lexical,
        semantic_contribution=None,
        trace=_trace(HybridChannelMembership.LEXICAL_ONLY, lexical_rank=2),
    )

    assert result.channel_membership is HybridChannelMembership.LEXICAL_ONLY
    assert result.lexical_contribution is not None
    assert result.lexical_contribution.lexical_hit.raw_bm25_score == -3.25
    assert result.lexical_contribution.lexical_hit.anchors == _anchors()
    assert result.semantic_contribution is None


def test_semantic_only_hit_keeps_span_evidence_separate_from_parent_context() -> None:
    semantic = HybridSemanticContribution(semantic_hit=_semantic_hit())
    result = HybridRetrievalHit(
        identity=_identity(),
        parent_chunk=_parent(),
        lexical_contribution=None,
        semantic_contribution=semantic,
        trace=_trace(HybridChannelMembership.SEMANTIC_ONLY, semantic_rank=3),
    )

    assert result.channel_membership is HybridChannelMembership.SEMANTIC_ONLY
    assert result.parent_chunk.chunk_text == "alpha beta gamma"
    assert not hasattr(result.parent_chunk, "anchors")
    assert result.semantic_contribution is not None
    assert result.semantic_contribution.semantic_hit.embedding_span_text == "beta"
    assert tuple(
        anchor.anchor_text for anchor in result.semantic_contribution.semantic_hit.anchors
    ) == ("beta",)


def test_dual_channel_hit_keeps_channel_packets_and_raw_scores_separate() -> None:
    lexical = HybridLexicalContribution(lexical_hit=_lexical_hit())
    semantic = HybridSemanticContribution(semantic_hit=_semantic_hit())
    result = HybridRetrievalHit(
        identity=_identity(),
        parent_chunk=_parent(),
        lexical_contribution=lexical,
        semantic_contribution=semantic,
        trace=_trace(HybridChannelMembership.BOTH, lexical_rank=2, semantic_rank=3),
    )

    assert result.channel_membership is HybridChannelMembership.BOTH
    assert result.lexical_contribution is not None
    assert result.semantic_contribution is not None
    assert result.lexical_contribution.lexical_hit.raw_bm25_score == -3.25
    assert result.semantic_contribution.semantic_hit.raw_semantic_score == 0.75
    assert (
        result.lexical_contribution.lexical_hit.anchors
        != result.semantic_contribution.semantic_hit.anchors
    )


def test_existing_channel_contracts_embed_without_copying_or_relabeling_evidence() -> None:
    lexical = HybridLexicalContribution(lexical_hit=_lexical_hit())
    semantic = HybridSemanticContribution(semantic_hit=_semantic_hit())

    assert lexical.lexical_hit == _lexical_hit()
    assert semantic.semantic_hit == _semantic_hit()
    assert lexical.identity == semantic.identity == _identity()


def test_trace_serializes_stable_membership_and_exact_score_not_confidence() -> None:
    trace = _trace(HybridChannelMembership.BOTH, lexical_rank=2, semantic_rank=3)

    dumped = trace.model_dump(mode="json")

    assert dumped["channel_membership"] == "both"
    assert dumped["fusion_score"] == {"numerator": 1, "denominator": 41}
    assert "confidence" not in HybridRankingTrace.model_json_schema()["properties"]["fusion_score"]


def test_results_allow_healthy_empty_channel_state_without_availability_errors() -> None:
    semantic = HybridSemanticContribution(semantic_hit=_semantic_hit())
    hit = HybridRetrievalHit(
        identity=_identity(),
        parent_chunk=_parent(),
        lexical_contribution=None,
        semantic_contribution=semantic,
        trace=_trace(HybridChannelMembership.SEMANTIC_ONLY, semantic_rank=3),
    )

    results = HybridRetrievalResults(
        project_id="project-1",
        query="beta",
        fusion_profile_id="rrf-v1",
        lexical_state=HybridChannelState.HEALTHY_EMPTY,
        semantic_state=HybridChannelState.HEALTHY_RESULTS,
        hits=(hit,),
    )

    assert results.model_dump(mode="json")["lexical_state"] == "healthy_empty"


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: HybridRankingTrace(
                fusion_profile_id="rrf-v1",
                fusion_score=ExactRationalScore(numerator=1, denominator=41),
                fusion_rank=1,
                channel_membership=HybridChannelMembership.LEXICAL_ONLY,
                lexical_rank=0,
            ),
            "greater than or equal to 1",
        ),
        (
            lambda: HybridParentChunkContext(
                identity=_identity(),
                paper_id="paper-1",
                file_version_id="file-version-1",
                physical_page_index=0,
                display_page_number=1,
                printed_page_label=None,
                chunk_ordinal=0,
                start_offset=5,
                end_offset=5,
                chunk_text="x",
            ),
            "parent chunk offsets",
        ),
        (
            lambda: HybridLexicalContribution(lexical_hit=_lexical_hit(raw_score=math.inf)),
            "raw BM25 score must be finite",
        ),
    ],
)
def test_structurally_invalid_values_are_rejected(factory: object, message: str) -> None:
    assert callable(factory)
    with pytest.raises(ValidationError, match=message):
        factory()


def test_final_hit_rejects_missing_contributions_and_forbidden_fields() -> None:
    with pytest.raises(ValidationError, match="at least one channel contribution"):
        HybridRetrievalHit(
            identity=_identity(),
            parent_chunk=_parent(),
            lexical_contribution=None,
            semantic_contribution=None,
            trace=_trace(HybridChannelMembership.LEXICAL_ONLY, lexical_rank=2),
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ChunkCandidateIdentity(
            project_id="project-1",
            document_generation_id="generation-1",
            page_id="page-1",
            chunk_id="chunk-1",
            vector_row=7,
        )
