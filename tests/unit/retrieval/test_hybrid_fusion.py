"""Unit tests for the frozen, pure ``rrf-v1`` fusion profile."""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations

import pytest

from academic_chatbot.documents.native_pdf import build_native_pdf_anchor
from academic_chatbot.documents.normalization import canonicalize_extracted_words
from academic_chatbot.retrieval.hybrid_fusion import (
    MISSING_RANK,
    RRF_V1_K,
    RRF_V1_LEXICAL_WEIGHT,
    RRF_V1_PROFILE_ID,
    RRF_V1_SEMANTIC_WEIGHT,
    candidate_limit,
    collapse_semantic_hits,
    fuse_candidates,
    lexical_candidate_key,
    semantic_candidate_key,
)
from academic_chatbot.retrieval.hybrid_models import HybridChannelMembership
from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.service import RetrievalHit

_TEXT = "alpha beta gamma delta"
_WORD_RANGES = {
    "alpha": (0, 5),
    "beta": (6, 10),
    "gamma": (11, 16),
    "delta": (17, 22),
}


def _anchors() -> dict[str, object]:
    canonical = canonicalize_extracted_words(
        tuple(
            {
                "text": word,
                "x0": float(index * 10),
                "top": 1.0,
                "x1": float(index * 10 + 9),
                "bottom": 5.0,
            }
            for index, word in enumerate(_WORD_RANGES)
        ),
        page_width_points=100.0,
        page_height_points=100.0,
    )
    return {
        word.text: build_native_pdf_anchor(
            file_version_id="file-1",
            source_pdf_sha256="a" * 64,
            physical_page_index=0,
            page_width_points=100.0,
            page_height_points=100.0,
            source_page_rotation_degrees=0,
            canonical_page=canonical,
            word=word,
        )
        for word in canonical.words
    }


_ANCHORS = _anchors()


def _lexical_hit(
    *,
    chunk_id: str = "chunk-1",
    generation_id: str = "generation-1",
    rank: int = 1,
    raw_score: float = -1.0,
    paper_id: str = "paper-1",
) -> RetrievalHit:
    return RetrievalHit(
        project_id="project-1",
        paper_id=paper_id,
        file_version_id="file-1",
        document_generation_id=generation_id,
        page_id="page-1",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_id=chunk_id,
        chunk_ordinal=0,
        chunk_text=_TEXT,
        start_offset=0,
        end_offset=len(_TEXT),
        rank=rank,
        raw_bm25_score=raw_score,
        anchors=tuple(_ANCHORS.values()),
    )


def _semantic_hit(
    *,
    chunk_id: str = "chunk-1",
    generation_id: str = "generation-1",
    rank: int = 1,
    raw_score: float = 0.5,
    word: str = "beta",
    end_word: str | None = None,
    span_id: str = "span-1",
    paper_id: str = "paper-1",
) -> SemanticRetrievalHit:
    start_offset, end_offset = _WORD_RANGES[word]
    if end_word is not None:
        end_offset = _WORD_RANGES[end_word][1]
    span_anchors = tuple(
        anchor
        for anchor_word, anchor in _ANCHORS.items()
        if start_offset <= _WORD_RANGES[anchor_word][0]
        and _WORD_RANGES[anchor_word][1] <= end_offset
    )
    return SemanticRetrievalHit(
        project_id="project-1",
        paper_id=paper_id,
        file_version_id="file-1",
        document_generation_id=generation_id,
        page_id="page-1",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_id=chunk_id,
        embedding_span_id=span_id,
        embedding_profile_id="embedding-profile-1",
        vector_generation_id="vector-generation-1",
        start_offset=start_offset,
        end_offset=end_offset,
        embedding_span_text=_TEXT[start_offset:end_offset],
        rank=rank,
        raw_semantic_score=raw_score,
        anchors=span_anchors,
    )


def _identity_tuple(candidate: object) -> tuple[str, str, str, str]:
    identity = candidate.identity  # type: ignore[attr-defined]
    return (
        identity.project_id,
        identity.document_generation_id,
        identity.page_id,
        identity.chunk_id,
    )


def test_rrf_v1_constants_and_candidate_depth_are_frozen() -> None:
    assert RRF_V1_PROFILE_ID == "rrf-v1"
    assert RRF_V1_K == 40
    assert RRF_V1_LEXICAL_WEIGHT == RRF_V1_SEMANTIC_WEIGHT == 1
    assert [candidate_limit(limit) for limit in (1, 5, 10, 11, 20, 50)] == [
        50,
        50,
        50,
        55,
        100,
        100,
    ]


@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_candidate_limit_rejects_non_positive_or_non_integer_values(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        candidate_limit(value)  # type: ignore[arg-type]


def test_channel_key_adapters_converge_on_the_same_parent_occurrence() -> None:
    lexical = _lexical_hit()
    semantic = _semantic_hit()

    assert lexical_candidate_key(lexical) == semantic_candidate_key(semantic)
    assert semantic_candidate_key(_semantic_hit(chunk_id="chunk-2")) != semantic_candidate_key(
        semantic
    )


def test_semantic_collapse_uses_the_frozen_representative_tuple_and_is_order_independent() -> None:
    first_parent = (
        _semantic_hit(rank=2, raw_score=0.99, word="delta", span_id="span-z"),
        _semantic_hit(rank=1, raw_score=0.10, word="gamma", span_id="span-a"),
        _semantic_hit(rank=1, raw_score=0.90, word="delta", span_id="span-b"),
        _semantic_hit(rank=1, raw_score=0.90, word="beta", span_id="span-c"),
        _semantic_hit(rank=1, raw_score=0.90, word="beta", span_id="span-a"),
    )
    second_parent = _semantic_hit(chunk_id="chunk-2", rank=3, word="alpha", span_id="span-other")

    expected = ("span-a", "span-other")
    all_hits = (*first_parent, second_parent)
    for ordered in (all_hits, tuple(reversed(all_hits))):
        collapsed = collapse_semantic_hits(ordered)
        assert tuple(item.semantic_hit.embedding_span_id for item in collapsed) == expected


def test_semantic_rank_precedes_raw_cosine_for_inconsistent_manual_input() -> None:
    collapsed = collapse_semantic_hits(
        (
            _semantic_hit(rank=2, raw_score=0.99, word="delta", span_id="high-cosine"),
            _semantic_hit(rank=1, raw_score=-0.99, word="alpha", span_id="best-rank"),
        )
    )

    assert collapsed[0].semantic_hit.embedding_span_id == "best-rank"


def test_semantic_collapse_uses_lower_end_after_an_equal_start_and_score() -> None:
    collapsed = collapse_semantic_hits(
        (
            _semantic_hit(
                rank=1,
                raw_score=0.9,
                word="beta",
                end_word="gamma",
                span_id="longer-span",
            ),
            _semantic_hit(rank=1, raw_score=0.9, word="beta", span_id="shorter-span"),
        )
    )

    assert collapsed[0].semantic_hit.embedding_span_id == "shorter-span"


@pytest.mark.parametrize(
    ("lexical", "semantic", "expected"),
    [
        ((_lexical_hit(rank=1),), (), Fraction(1, 41)),
        ((), (_semantic_hit(rank=1),), Fraction(1, 41)),
        ((_lexical_hit(rank=1),), (_semantic_hit(rank=1),), Fraction(2, 41)),
        ((_lexical_hit(rank=1),), (_semantic_hit(rank=2),), Fraction(1, 41) + Fraction(1, 42)),
    ],
)
def test_rrf_uses_exact_reduced_rational_scores(
    lexical: tuple[RetrievalHit, ...],
    semantic: tuple[SemanticRetrievalHit, ...],
    expected: Fraction,
) -> None:
    candidate = fuse_candidates(lexical, semantic, final_limit=10)[0]

    assert candidate.trace.fusion_score.numerator == expected.numerator
    assert candidate.trace.fusion_score.denominator == expected.denominator


def test_raw_channel_scores_are_preserved_but_do_not_calibrate_rrf() -> None:
    baseline = fuse_candidates(
        (_lexical_hit(rank=1, raw_score=-0.01),),
        (_semantic_hit(rank=2, raw_score=0.01),),
        final_limit=10,
    )[0]
    altered = fuse_candidates(
        (_lexical_hit(rank=1, raw_score=-99999.0),),
        (_semantic_hit(rank=2, raw_score=99999.0),),
        final_limit=10,
    )[0]

    assert altered.trace.fusion_score == baseline.trace.fusion_score
    assert altered.lexical_contribution is not None
    assert altered.semantic_contribution is not None
    assert altered.lexical_contribution.lexical_hit.raw_bm25_score == -99999.0
    assert altered.semantic_contribution.semantic_hit.raw_semantic_score == 99999.0


def test_duplicate_channels_cast_one_vote_and_keep_the_selected_source_objects() -> None:
    lexical = _lexical_hit(rank=2)
    semantic_best = _semantic_hit(rank=1, word="beta", span_id="best")
    semantic_other = _semantic_hit(rank=2, word="gamma", span_id="other")

    candidates = fuse_candidates((lexical,), (semantic_other, semantic_best), final_limit=10)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.trace.channel_membership is HybridChannelMembership.BOTH
    assert candidate.trace.lexical_rank == 2
    assert candidate.trace.semantic_rank == 1
    assert candidate.semantic_contribution is not None
    assert candidate.semantic_contribution.semantic_hit == semantic_best


def test_duplicate_lexical_parent_uses_best_rank_then_existing_bm25_order() -> None:
    selected = fuse_candidates(
        (
            _lexical_hit(rank=3, raw_score=-10.0),
            _lexical_hit(rank=1, raw_score=10.0),
            _lexical_hit(rank=1, raw_score=-5.0),
        ),
        (),
        final_limit=10,
    )[0]

    assert selected.trace.lexical_rank == 1
    assert selected.lexical_contribution is not None
    assert selected.lexical_contribution.lexical_hit.raw_bm25_score == -5.0


def test_contradictory_duplicate_parent_lineage_fails_closed() -> None:
    with pytest.raises(ValueError, match="contradictory lexical evidence"):
        fuse_candidates(
            (_lexical_hit(), _lexical_hit(paper_id="paper-other")), (), final_limit=10
        )

    with pytest.raises(ValueError, match="contradictory cross-channel lineage"):
        fuse_candidates(
            (_lexical_hit(),),
            (_semantic_hit(paper_id="paper-other"),),
            final_limit=10,
        )


def test_final_order_uses_exact_ties_missing_sentinel_and_identity_after_shuffling() -> None:
    lexical = (
        _lexical_hit(chunk_id="chunk-b", rank=1),
        _lexical_hit(chunk_id="chunk-a", rank=1),
        _lexical_hit(chunk_id="chunk-c", rank=2),
    )
    semantic = (
        _semantic_hit(chunk_id="chunk-d", rank=1, word="alpha", span_id="span-d"),
        _semantic_hit(chunk_id="chunk-e", rank=2, word="beta", span_id="span-e"),
    )

    expected = ("chunk-a", "chunk-b", "chunk-d", "chunk-c", "chunk-e")
    for lexical_order in permutations(lexical):
        candidates = fuse_candidates(lexical_order, tuple(reversed(semantic)), final_limit=10)
        assert tuple(candidate.identity.chunk_id for candidate in candidates) == expected
        assert candidates[0].trace.lexical_rank == 1
        assert candidates[2].trace.lexical_rank is None
        assert candidates[2].trace.semantic_rank == 1
        assert MISSING_RANK > candidate_limit(50)


def test_final_limit_is_applied_only_after_full_sorting_and_rank_assignment() -> None:
    candidates = fuse_candidates(
        (_lexical_hit(chunk_id="chunk-low", rank=3), _lexical_hit(chunk_id="chunk-high", rank=1)),
        (),
        final_limit=1,
    )

    assert len(candidates) == 1
    assert candidates[0].identity.chunk_id == "chunk-high"
    assert candidates[0].trace.fusion_rank == 1


def test_invalid_model_constructed_rank_and_score_fail_closed() -> None:
    bad_rank = _lexical_hit().model_copy(update={"rank": 0})
    bad_score = _semantic_hit().model_copy(update={"raw_semantic_score": float("inf")})

    with pytest.raises(ValueError, match="rank must be a positive integer"):
        fuse_candidates((bad_rank,), (), final_limit=10)
    with pytest.raises(ValueError, match="raw semantic score must be finite"):
        fuse_candidates((), (bad_score,), final_limit=10)


def test_profile_is_not_mutable_or_selectable() -> None:
    with pytest.raises(ValueError, match="only supports the frozen rrf-v1 profile"):
        fuse_candidates((_lexical_hit(),), (), final_limit=10, profile="rrf-v2")


def test_frozen_rrf_v1_fixture_has_explicit_order_scores_and_membership() -> None:
    identities = (
        ("task0-project", "generation-a", "page-a", "chunk-a"),
        ("task0-project", "generation-b", "page-b", "chunk-b"),
        ("task0-project", "generation-c", "page-c", "chunk-c"),
    )
    lexical = tuple(
        _lexical_hit(
            chunk_id=identity[3],
            generation_id=identity[1],
            rank=rank,
            raw_score=score,
        ).model_copy(update={"project_id": identity[0], "page_id": identity[2]})
        for identity, rank, score in zip(identities[:2], (1, 2), (0.9, 0.8), strict=True)
    )
    semantic = tuple(
        _semantic_hit(
            chunk_id=identity[3],
            generation_id=identity[1],
            rank=rank,
            raw_score=score,
            word="beta",
            span_id=span_id,
        ).model_copy(update={"project_id": identity[0], "page_id": identity[2]})
        for identity, rank, score, span_id in zip(
            identities, (2, 1, 1), (0.7, 0.6, 0.5), ("span-a", "span-b", "span-c"), strict=True
        )
    )

    actual = fuse_candidates(lexical, semantic, final_limit=10)

    assert tuple(_identity_tuple(candidate) for candidate in actual) == identities
    expected_scores = (Fraction(83, 1722), Fraction(83, 1722), Fraction(1, 41))
    assert tuple(
        Fraction(candidate.trace.fusion_score.numerator, candidate.trace.fusion_score.denominator)
        for candidate in actual
    ) == expected_scores
    assert [candidate.channel_membership for candidate in actual] == [
        HybridChannelMembership.BOTH,
        HybridChannelMembership.BOTH,
        HybridChannelMembership.SEMANTIC_ONLY,
    ]
    assert [candidate.trace.fusion_rank for candidate in actual] == [1, 2, 3]
