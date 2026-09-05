"""Pure deterministic semantic collapse and exact ``rrf-v1`` fusion.

This module does not retrieve evidence or construct parent chunk context.
Task 3 owns read-only parent-context reconstruction before final
``HybridRetrievalHit`` values are materialized.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from academic_chatbot.retrieval.hybrid_models import (
    ExactRationalScore,
    HybridCandidateKey,
    HybridChannelMembership,
    HybridLexicalContribution,
    HybridRankingTrace,
    HybridSemanticContribution,
)
from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.service import RetrievalHit

RRF_V1_PROFILE_ID = "rrf-v1"
RRF_V1_K = 40
RRF_V1_LEXICAL_WEIGHT = 1
RRF_V1_SEMANTIC_WEIGHT = 1
_MIN_CANDIDATE_DEPTH = 50
_MAX_CANDIDATE_DEPTH = 100
MISSING_RANK = 2**31 - 1


@dataclass(frozen=True, slots=True)
class FusedHybridCandidate:
    """Pure fusion output awaiting Task 3 parent-context reconstruction.

    It deliberately reuses Task 1's identity, labelled contributions, and
    ranking trace rather than duplicating a final hybrid retrieval wire model.
    """

    identity: HybridCandidateKey
    lexical_contribution: HybridLexicalContribution | None
    semantic_contribution: HybridSemanticContribution | None
    trace: HybridRankingTrace

    @property
    def channel_membership(self) -> HybridChannelMembership:
        """Expose the trace's validated channel membership."""

        return self.trace.channel_membership


def candidate_limit(final_limit: int) -> int:
    """Return the frozen per-channel candidate depth for a final result limit."""

    if type(final_limit) is not int or final_limit <= 0:
        raise ValueError("final_limit must be a positive integer")
    return min(_MAX_CANDIDATE_DEPTH, max(_MIN_CANDIDATE_DEPTH, final_limit * 5))


def lexical_candidate_key(hit: RetrievalHit) -> HybridCandidateKey:
    """Derive the frozen parent identity from one lexical hit."""

    _validate_lexical_hit(hit)
    return HybridCandidateKey(
        project_id=hit.project_id,
        document_generation_id=hit.document_generation_id,
        page_id=hit.page_id,
        chunk_id=hit.chunk_id,
    )


def semantic_candidate_key(hit: SemanticRetrievalHit) -> HybridCandidateKey:
    """Derive the frozen parent identity from one semantic span hit."""

    _validate_semantic_hit(hit)
    return HybridCandidateKey(
        project_id=hit.project_id,
        document_generation_id=hit.document_generation_id,
        page_id=hit.page_id,
        chunk_id=hit.chunk_id,
    )


def collapse_semantic_hits(
    hits: Iterable[SemanticRetrievalHit],
) -> tuple[HybridSemanticContribution, ...]:
    """Select exactly one frozen-order semantic representative per parent chunk."""

    groups: dict[HybridCandidateKey, list[SemanticRetrievalHit]] = {}
    for hit in hits:
        key = semantic_candidate_key(hit)
        groups.setdefault(key, []).append(hit)

    collapsed: list[HybridSemanticContribution] = []
    for key in sorted(groups, key=_identity_tuple):
        group = groups[key]
        _require_consistent_semantic_parent(group)
        representative = min(group, key=_semantic_representative_key)
        collapsed.append(HybridSemanticContribution(semantic_hit=representative))
    return tuple(collapsed)


def fuse_candidates(
    lexical_hits: Iterable[RetrievalHit],
    semantic_hits: Iterable[SemanticRetrievalHit],
    *,
    final_limit: int,
    profile: str = RRF_V1_PROFILE_ID,
) -> tuple[FusedHybridCandidate, ...]:
    """Fuse supplied channel results under the immutable exact ``rrf-v1`` profile.

    Inputs are already bounded by the caller's channel requests. This function
    only collapses duplicate parent candidates, ranks the fused candidates, and
    applies ``final_limit`` after assigning deterministic fusion ranks.
    """

    if profile != RRF_V1_PROFILE_ID:
        raise ValueError("fusion only supports the frozen rrf-v1 profile")
    candidate_limit(final_limit)

    lexical = _collapse_lexical_hits(lexical_hits)
    semantic = collapse_semantic_hits(semantic_hits)
    candidates: dict[
        HybridCandidateKey,
        tuple[HybridLexicalContribution | None, HybridSemanticContribution | None],
    ] = {}
    for lexical_contribution in lexical:
        candidates[lexical_contribution.identity] = (lexical_contribution, None)
    for semantic_contribution in semantic:
        existing = candidates.get(semantic_contribution.identity)
        if existing is None:
            candidates[semantic_contribution.identity] = (None, semantic_contribution)
            continue
        existing_lexical, _ = existing
        assert existing_lexical is not None
        _require_consistent_cross_channel_lineage(
            existing_lexical.lexical_hit, semantic_contribution.semantic_hit
        )
        candidates[semantic_contribution.identity] = (existing_lexical, semantic_contribution)

    pending = tuple(
        (
            identity,
            lexical_contribution,
            semantic_contribution,
            _exact_rrf_score(lexical_contribution, semantic_contribution),
        )
        for identity, (lexical_contribution, semantic_contribution) in candidates.items()
    )
    ordered = sorted(pending, key=_final_order_key)
    fused = tuple(
        _materialize_fused_candidate(
            identity=identity,
            lexical_contribution=lexical_contribution,
            semantic_contribution=semantic_contribution,
            score=score,
            fusion_rank=index,
        )
        for index, (identity, lexical_contribution, semantic_contribution, score) in enumerate(
            ordered, start=1
        )
    )
    return fused[:final_limit]


def _collapse_lexical_hits(
    hits: Iterable[RetrievalHit],
) -> tuple[HybridLexicalContribution, ...]:
    groups: dict[HybridCandidateKey, list[RetrievalHit]] = {}
    for hit in hits:
        key = lexical_candidate_key(hit)
        groups.setdefault(key, []).append(hit)

    collapsed: list[HybridLexicalContribution] = []
    for key in sorted(groups, key=_identity_tuple):
        group = groups[key]
        _require_consistent_lexical_evidence(group)
        # FTS5 orders BM25 ascending, then paper/file/page/chunk facts. The
        # parent key fixes generation/page/chunk, while contradictory remaining
        # evidence has already failed closed above.
        representative = min(group, key=_lexical_representative_key)
        collapsed.append(HybridLexicalContribution(lexical_hit=representative))
    return tuple(collapsed)


def _validate_lexical_hit(hit: RetrievalHit) -> None:
    if not isinstance(hit, RetrievalHit):
        raise ValueError("lexical input must be a RetrievalHit")
    _validate_rank(hit.rank)
    if not math.isfinite(hit.raw_bm25_score):
        raise ValueError("raw BM25 score must be finite")


def _validate_semantic_hit(hit: SemanticRetrievalHit) -> None:
    if not isinstance(hit, SemanticRetrievalHit):
        raise ValueError("semantic input must be a SemanticRetrievalHit")
    _validate_rank(hit.rank)
    if not math.isfinite(hit.raw_semantic_score):
        raise ValueError("raw semantic score must be finite")


def _validate_rank(rank: int) -> None:
    if type(rank) is not int or not 1 <= rank <= _MAX_CANDIDATE_DEPTH:
        raise ValueError("rank must be a positive integer within the frozen candidate depth")


def _require_consistent_lexical_evidence(hits: list[RetrievalHit]) -> None:
    first = hits[0]
    expected = _lexical_evidence_facts(first)
    if any(_lexical_evidence_facts(hit) != expected for hit in hits[1:]):
        raise ValueError("contradictory lexical evidence for one parent candidate")


def _require_consistent_semantic_parent(hits: list[SemanticRetrievalHit]) -> None:
    first = hits[0]
    expected = _shared_lineage_facts(first)
    if any(_shared_lineage_facts(hit) != expected for hit in hits[1:]):
        raise ValueError("contradictory semantic parent lineage for one parent candidate")


def _require_consistent_cross_channel_lineage(
    lexical: RetrievalHit, semantic: SemanticRetrievalHit
) -> None:
    if _shared_lineage_facts(lexical) != _shared_lineage_facts(semantic):
        raise ValueError("contradictory cross-channel lineage for one parent candidate")


def _lexical_evidence_facts(hit: RetrievalHit) -> tuple[object, ...]:
    return (
        *_shared_lineage_facts(hit),
        hit.chunk_ordinal,
        hit.start_offset,
        hit.end_offset,
        hit.chunk_text,
        hit.anchors,
    )


def _shared_lineage_facts(hit: RetrievalHit | SemanticRetrievalHit) -> tuple[object, ...]:
    return (
        hit.project_id,
        hit.paper_id,
        hit.file_version_id,
        hit.document_generation_id,
        hit.page_id,
        hit.physical_page_index,
        hit.display_page_number,
        hit.printed_page_label,
        hit.chunk_id,
    )


def _lexical_representative_key(hit: RetrievalHit) -> tuple[object, ...]:
    return (
        hit.rank,
        hit.raw_bm25_score,
        hit.paper_id,
        hit.file_version_id,
        hit.physical_page_index,
        hit.chunk_ordinal,
        hit.chunk_id,
    )


def _semantic_representative_key(hit: SemanticRetrievalHit) -> tuple[object, ...]:
    return (
        hit.rank,
        -hit.raw_semantic_score,
        hit.start_offset,
        hit.end_offset,
        hit.embedding_span_id,
    )


def _exact_rrf_score(
    lexical: HybridLexicalContribution | None,
    semantic: HybridSemanticContribution | None,
) -> ExactRationalScore:
    score = Fraction(0, 1)
    if lexical is not None:
        score += Fraction(RRF_V1_LEXICAL_WEIGHT, RRF_V1_K + lexical.lexical_hit.rank)
    if semantic is not None:
        score += Fraction(RRF_V1_SEMANTIC_WEIGHT, RRF_V1_K + semantic.semantic_hit.rank)
    if score <= 0:
        raise ValueError("a fused candidate requires at least one channel contribution")
    return ExactRationalScore(numerator=score.numerator, denominator=score.denominator)


def _final_order_key(
    item: tuple[
        HybridCandidateKey,
        HybridLexicalContribution | None,
        HybridSemanticContribution | None,
        ExactRationalScore,
    ],
) -> tuple[Fraction, int, int, int, int, tuple[str, str, str, str]]:
    identity, lexical, semantic, score = item
    lexical_rank = lexical.lexical_hit.rank if lexical is not None else MISSING_RANK
    semantic_rank = semantic.semantic_hit.rank if semantic is not None else MISSING_RANK
    exact_score = Fraction(score.numerator, score.denominator)
    return (
        -exact_score,
        min(lexical_rank, semantic_rank),
        -int(lexical is not None) - int(semantic is not None),
        lexical_rank,
        semantic_rank,
        _identity_tuple(identity),
    )


def _materialize_fused_candidate(
    *,
    identity: HybridCandidateKey,
    lexical_contribution: HybridLexicalContribution | None,
    semantic_contribution: HybridSemanticContribution | None,
    score: ExactRationalScore,
    fusion_rank: int,
) -> FusedHybridCandidate:
    lexical_rank = lexical_contribution.lexical_hit.rank if lexical_contribution else None
    semantic_rank = semantic_contribution.semantic_hit.rank if semantic_contribution else None
    return FusedHybridCandidate(
        identity=identity,
        lexical_contribution=lexical_contribution,
        semantic_contribution=semantic_contribution,
        trace=HybridRankingTrace(
            fusion_profile_id=RRF_V1_PROFILE_ID,
            fusion_score=score,
            fusion_rank=fusion_rank,
            channel_membership=_membership_for(lexical_contribution, semantic_contribution),
            lexical_rank=lexical_rank,
            semantic_rank=semantic_rank,
        ),
    )


def _membership_for(
    lexical: HybridLexicalContribution | None, semantic: HybridSemanticContribution | None
) -> HybridChannelMembership:
    if lexical is not None and semantic is not None:
        return HybridChannelMembership.BOTH
    if lexical is not None:
        return HybridChannelMembership.LEXICAL_ONLY
    if semantic is not None:
        return HybridChannelMembership.SEMANTIC_ONLY
    raise ValueError("a fused candidate requires at least one channel contribution")


def _identity_tuple(identity: HybridCandidateKey) -> tuple[str, str, str, str]:
    return (
        identity.project_id,
        identity.document_generation_id,
        identity.page_id,
        identity.chunk_id,
    )
