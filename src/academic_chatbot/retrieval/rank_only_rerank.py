"""Explicit rank-only ordering for already acquired baseline occurrences."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.semantic_position import (
    MAX_POSITIONAL_HITS_EXAMINED,
    SEMANTIC_POSITIONAL_ACQUISITION_POLICY_ID,
    SUPPORTED_POSITIONAL_CONCERN_ID,
    SemanticPositionalAcquisition,
)

RANK_ONLY_PARENT_RERANK_POLICY_ID = "rank-only-parent-rerank-v1"
MAX_RERANK_CANDIDATES = MAX_POSITIONAL_HITS_EXAMINED

type OccurrenceIdentity = tuple[object, ...]
type ParentOccurrenceKey = tuple[str, str, str, str, str]


class RankOnlyParentRerankError(ValueError):
    """Raised when the explicit rank-only experiment input is unsafe."""


@dataclass(frozen=True, slots=True)
class RankedParentOccurrence:
    """One existing baseline occurrence with experiment-only order metadata."""

    hit: SemanticRetrievalHit
    original_rank: int
    post_rank: int
    frontier_depth: int


@dataclass(frozen=True, slots=True)
class _RankedCandidate:
    hit: SemanticRetrievalHit
    original_index: int
    semantic_rank: int
    lexical_rank: int | None
    occurrence_identity: OccurrenceIdentity


def apply_rank_only_parent_rerank(
    *,
    acquisition: SemanticPositionalAcquisition,
    lexical_ranks: Mapping[ParentOccurrenceKey, int],
    policy_id: str | None = None,
) -> tuple[RankedParentOccurrence, ...]:
    """Apply the frozen Pareto policy to the acquired baseline lane only.

    The lexical mapping is an annotation supplied by the experiment caller.
    This function performs no retrieval, embedding, model, or storage work.
    """

    baseline = _validate_acquisition(acquisition)
    if policy_id != RANK_ONLY_PARENT_RERANK_POLICY_ID:
        return _unchanged(baseline)

    ranks = _validate_lexical_ranks(
        lexical_ranks, project_id=acquisition.project_id
    )
    candidates = tuple(
        _RankedCandidate(
            hit=hit,
            original_index=index,
            semantic_rank=hit.rank,
            lexical_rank=ranks.get(_parent_key(hit)),
            occurrence_identity=_occurrence_identity(hit),
        )
        for index, hit in enumerate(baseline)
    )
    layered = _build_frontier_layers(candidates)
    ordered = sorted(
        layered,
        key=lambda candidate: (
            candidate[1],
            candidate[0].original_index,
            candidate[0].occurrence_identity,
        ),
    )
    return tuple(
        RankedParentOccurrence(
            hit=candidate[0].hit,
            original_rank=candidate[0].semantic_rank,
            post_rank=post_rank,
            frontier_depth=candidate[1],
        )
        for post_rank, candidate in enumerate(ordered, start=1)
    )


def _validate_acquisition(
    acquisition: SemanticPositionalAcquisition,
) -> tuple[SemanticRetrievalHit, ...]:
    if not isinstance(acquisition, SemanticPositionalAcquisition):
        raise RankOnlyParentRerankError("positional acquisition is invalid")
    if acquisition.policy_id != SEMANTIC_POSITIONAL_ACQUISITION_POLICY_ID:
        raise RankOnlyParentRerankError("positional acquisition policy is unsupported")
    if acquisition.concern_id != SUPPORTED_POSITIONAL_CONCERN_ID:
        raise RankOnlyParentRerankError("positional acquisition concern is unsupported")
    baseline = acquisition.baseline_hits
    if type(baseline) is not tuple or len(baseline) > MAX_RERANK_CANDIDATES:
        raise RankOnlyParentRerankError("baseline candidate count exceeds the bound")
    if acquisition.ordered_hits != acquisition.baseline_hits + acquisition.positional_hits:
        raise RankOnlyParentRerankError("positional acquisition ordering is inconsistent")

    seen: set[OccurrenceIdentity] = set()
    for hit in baseline:
        if not isinstance(hit, SemanticRetrievalHit):
            raise RankOnlyParentRerankError("baseline occurrence is invalid")
        identity = _occurrence_identity(hit)
        if identity in seen:
            raise RankOnlyParentRerankError("baseline occurrence identity is duplicated")
        seen.add(identity)
        if (
            hit.project_id != acquisition.project_id
            or hit.embedding_profile_id != acquisition.embedding_profile_id
            or hit.vector_generation_id != acquisition.vector_generation_id
        ):
            raise RankOnlyParentRerankError("baseline occurrence lineage is inconsistent")
        if type(hit.rank) is not int or hit.rank < 1:
            raise RankOnlyParentRerankError("semantic acquisition rank is invalid")
    return baseline


def _validate_lexical_ranks(
    lexical_ranks: Mapping[ParentOccurrenceKey, int], *, project_id: str
) -> dict[ParentOccurrenceKey, int]:
    if not isinstance(lexical_ranks, Mapping):
        raise RankOnlyParentRerankError("lexical rank mapping is invalid")
    validated: dict[ParentOccurrenceKey, int] = {}
    for key, rank in lexical_ranks.items():
        if (
            type(key) is not tuple
            or len(key) != 5
            or any(type(value) is not str or not value for value in key)
        ):
            raise RankOnlyParentRerankError("lexical parent identity is invalid")
        if key[0] != project_id:
            raise RankOnlyParentRerankError("lexical rank project scope is inconsistent")
        if type(rank) is not int or rank < 1:
            raise RankOnlyParentRerankError("lexical rank is invalid")
        validated[key] = rank
    return validated


def _build_frontier_layers(
    candidates: tuple[_RankedCandidate, ...],
) -> tuple[tuple[_RankedCandidate, int], ...]:
    remaining = list(candidates)
    layered: list[tuple[_RankedCandidate, int]] = []
    frontier_depth = 1
    while remaining:
        frontier = [
            candidate
            for candidate in remaining
            if not any(
                _dominates(other, candidate)
                for other in remaining
                if other is not candidate
            )
        ]
        if not frontier:
            raise RankOnlyParentRerankError("pareto frontier construction failed")
        frontier_ids = {id(candidate) for candidate in frontier}
        layered.extend((candidate, frontier_depth) for candidate in frontier)
        remaining = [candidate for candidate in remaining if id(candidate) not in frontier_ids]
        frontier_depth += 1
    return tuple(layered)


def _dominates(left: _RankedCandidate, right: _RankedCandidate) -> bool:
    semantic_no_worse = left.semantic_rank <= right.semantic_rank
    lexical_no_worse = _lexical_no_worse(left.lexical_rank, right.lexical_rank)
    semantic_better = left.semantic_rank < right.semantic_rank
    lexical_better = _lexical_better(left.lexical_rank, right.lexical_rank)
    return (
        semantic_no_worse
        and lexical_no_worse
        and (semantic_better or lexical_better)
    )


def _lexical_no_worse(left: int | None, right: int | None) -> bool:
    if left is None:
        return right is None
    return right is None or left <= right


def _lexical_better(left: int | None, right: int | None) -> bool:
    if left is None:
        return False
    return right is None or left < right


def _unchanged(
    baseline: tuple[SemanticRetrievalHit, ...],
) -> tuple[RankedParentOccurrence, ...]:
    return tuple(
        RankedParentOccurrence(
            hit=hit,
            original_rank=hit.rank,
            post_rank=rank,
            frontier_depth=0,
        )
        for rank, hit in enumerate(baseline, start=1)
    )


def _parent_key(hit: SemanticRetrievalHit) -> ParentOccurrenceKey:
    return (
        hit.project_id,
        hit.file_version_id,
        hit.document_generation_id,
        hit.page_id,
        hit.chunk_id,
    )


def _occurrence_identity(hit: SemanticRetrievalHit) -> OccurrenceIdentity:
    values = (
        hit.project_id,
        hit.file_version_id,
        hit.document_generation_id,
        hit.page_id,
        hit.chunk_id,
        hit.embedding_span_id,
        hit.embedding_profile_id,
        hit.vector_generation_id,
        hit.start_offset,
        hit.end_offset,
    )
    if any(type(value) is not str or not value for value in values[:8]):
        raise RankOnlyParentRerankError("occurrence identity is invalid")
    if (
        type(values[8]) is not int
        or type(values[9]) is not int
        or values[8] < 0
        or values[9] <= values[8]
    ):
        raise RankOnlyParentRerankError("occurrence range is invalid")
    return values
