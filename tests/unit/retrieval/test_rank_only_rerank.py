from __future__ import annotations

from dataclasses import replace

import pytest

from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.semantic_position import SemanticPositionalAcquisition


def _hit(
    *,
    rank: int,
    span_id: str | None = None,
    project_id: str = "project-1",
    file_version_id: str = "file-1",
    document_generation_id: str = "generation-1",
    page_id: str = "page-1",
    chunk_id: str | None = None,
) -> SemanticRetrievalHit:
    span = span_id or f"span-{rank}"
    return SemanticRetrievalHit.model_construct(
        project_id=project_id,
        paper_id="paper-1",
        file_version_id=file_version_id,
        document_generation_id=document_generation_id,
        page_id=page_id,
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_id=chunk_id or f"chunk-{span}",
        embedding_span_id=span,
        embedding_profile_id="profile-1",
        vector_generation_id="vector-1",
        start_offset=rank * 10,
        end_offset=rank * 10 + 5,
        embedding_span_text=f"candidate-{span}",
        rank=rank,
        raw_semantic_score=1.0,
        anchors=(),
    )


def _acquisition(
    baseline: tuple[SemanticRetrievalHit, ...],
    positional: tuple[SemanticRetrievalHit, ...] = (),
) -> SemanticPositionalAcquisition:
    return SemanticPositionalAcquisition(
        project_id="project-1",
        query="ordinary query",
        embedding_profile_id="profile-1",
        vector_generation_id="vector-1",
        policy_id="semantic-positional-acquisition-v1",
        concern_id="stated-study-objective-v1",
        requested_limit=max(len(baseline), 1),
        examined_hits=baseline + positional,
        baseline_hits=baseline,
        positional_hits=positional,
        ordered_hits=baseline + positional,
    )


def _parent_key(hit: SemanticRetrievalHit) -> tuple[str, str, str, str, str]:
    return (
        hit.project_id,
        hit.file_version_id,
        hit.document_generation_id,
        hit.page_id,
        hit.chunk_id,
    )


def _apply(
    acquisition: SemanticPositionalAcquisition,
    lexical_ranks: dict[tuple[str, str, str, str, str], int],
    *,
    policy_id: str | None = "rank-only-parent-rerank-v1",
):
    from academic_chatbot.retrieval.rank_only_rerank import (
        apply_rank_only_parent_rerank,
    )

    return apply_rank_only_parent_rerank(
        acquisition=acquisition,
        lexical_ranks=lexical_ranks,
        policy_id=policy_id,
    )


def test_policy_identity_is_explicit() -> None:
    from academic_chatbot.retrieval.rank_only_rerank import (
        RANK_ONLY_PARENT_RERANK_POLICY_ID,
    )

    assert RANK_ONLY_PARENT_RERANK_POLICY_ID == "rank-only-parent-rerank-v1"


def test_candidate_universe_is_preserved() -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 4))
    positional = (_hit(rank=4, span_id="positional"),)
    acquisition = _acquisition(baseline, positional)
    lexical_only = _hit(rank=5, span_id="lexical-only")

    result = _apply(
        acquisition,
        {
            _parent_key(baseline[0]): 3,
            _parent_key(baseline[1]): 1,
            _parent_key(lexical_only): 1,
        },
    )

    assert {item.hit.embedding_span_id for item in result} == {
        hit.embedding_span_id for hit in baseline
    }
    assert positional[0].embedding_span_id not in {
        item.hit.embedding_span_id for item in result
    }
    assert lexical_only.embedding_span_id not in {
        item.hit.embedding_span_id for item in result
    }


def test_pareto_layers_use_two_ordinal_dimensions() -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 5))
    lexical_ranks = {
        _parent_key(baseline[0]): 4,
        _parent_key(baseline[1]): 5,
        _parent_key(baseline[2]): 1,
    }

    result = _apply(_acquisition(baseline), lexical_ranks)

    assert [item.hit for item in result] == [baseline[0], baseline[2], baseline[1], baseline[3]]
    assert [item.frontier_depth for item in result] == [1, 1, 2, 3]
    assert [item.post_rank for item in result] == [1, 2, 3, 4]


def test_missing_lexical_rank_is_infinite_absence() -> None:
    first = _hit(rank=1, span_id="absent")
    second = _hit(rank=2, span_id="finite")

    result = _apply(
        _acquisition((first, second)),
        {_parent_key(second): 3},
    )

    assert [item.hit for item in result] == [first, second]
    assert [item.frontier_depth for item in result] == [1, 1]


def test_equal_frontier_preserves_input_order() -> None:
    first = _hit(rank=2, span_id="first")
    second = _hit(rank=1, span_id="second")
    acquisition = _acquisition((first, second))

    result_one = _apply(
        acquisition,
        {_parent_key(first): 1, _parent_key(second): 2},
    )
    result_two = _apply(
        acquisition,
        {_parent_key(first): 1, _parent_key(second): 2},
    )

    assert [item.hit for item in result_one] == [first, second]
    assert result_one == result_two


def test_original_semantic_rank_is_not_mutated() -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 4))
    acquisition = _acquisition(baseline)
    before = (
        acquisition.examined_hits,
        acquisition.baseline_hits,
        acquisition.positional_hits,
        acquisition.ordered_hits,
    )

    result = _apply(
        acquisition,
        {
            _parent_key(baseline[0]): 4,
            _parent_key(baseline[1]): 5,
            _parent_key(baseline[2]): 1,
        },
    )

    assert [item.hit.rank for item in result] == [1, 3, 2]
    assert (
        acquisition.examined_hits,
        acquisition.baseline_hits,
        acquisition.positional_hits,
        acquisition.ordered_hits,
    ) == before
    assert tuple(item.hit for item in result) != baseline


def test_unsupported_policy_is_a_no_op() -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 4))

    result = _apply(
        _acquisition(baseline),
        {_parent_key(baseline[0]): 3, _parent_key(baseline[1]): 1},
        policy_id="different-policy-v1",
    )

    assert tuple(item.hit for item in result) == baseline
    assert [item.post_rank for item in result] == [1, 2, 3]


@pytest.mark.parametrize(
    "mutated",
    (
        lambda hits: replace(_acquisition(hits), project_id="other-project"),
        lambda hits: _acquisition((hits[0].model_copy(update={"rank": 0}), *hits[1:])),
    ),
)
def test_invalid_scope_or_rank_fails_closed(mutated) -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 4))

    with pytest.raises(ValueError):
        _apply(mutated(baseline), {_parent_key(baseline[0]): 1})


def test_candidate_cap_is_enforced() -> None:
    baseline = tuple(_hit(rank=rank, span_id=f"span-{rank}") for rank in range(1, 102))

    with pytest.raises(ValueError, match="candidate"):
        _apply(_acquisition(baseline), {})


def test_no_derived_hybrid_or_multi_query_signal_is_accepted() -> None:
    baseline = (_hit(rank=1),)

    with pytest.raises(ValueError):
        _apply(_acquisition(baseline), {_parent_key(baseline[0]): 1.0})
