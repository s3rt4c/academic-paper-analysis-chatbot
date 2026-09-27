from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from academic_chatbot.domain.library import Project
from academic_chatbot.retrieval.rank_only_service import (
    MAX_PRODUCTION_LEXICAL_RANK_DEPTH,
    PRODUCTION_RANK_ONLY_PARENT_RERANK_POLICY_ID,
    ProductionRankOnlyParentRerankService,
    ProductionRankOnlyUnsupportedError,
)
from academic_chatbot.retrieval.selection import (
    CanonicalPosition,
    occurrence_identity,
)
from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.semantic_position import SemanticPositionalAcquisition
from academic_chatbot.retrieval.service import RetrievalHit, RetrievalResults

_PROJECT = Project(project_id="project-1", display_name="Synthetic")
_QUERY = "Which result is selected?"


def _semantic_hit(
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
        embedding_span_text=f"candidate {span}",
        rank=rank,
        raw_semantic_score=1.0,
        anchors=(),
    )


def _lexical_hit(
    source: SemanticRetrievalHit,
    *,
    rank: int,
    project_id: str | None = None,
    file_version_id: str | None = None,
    document_generation_id: str | None = None,
    page_id: str | None = None,
    chunk_id: str | None = None,
) -> RetrievalHit:
    return RetrievalHit.model_construct(
        project_id=project_id or source.project_id,
        paper_id=source.paper_id,
        file_version_id=file_version_id or source.file_version_id,
        document_generation_id=document_generation_id or source.document_generation_id,
        page_id=page_id or source.page_id,
        physical_page_index=source.physical_page_index,
        display_page_number=source.display_page_number,
        printed_page_label=source.printed_page_label,
        chunk_id=chunk_id if chunk_id is not None else source.chunk_id,
        chunk_ordinal=0,
        chunk_text="lexical candidate",
        start_offset=0,
        end_offset=17,
        rank=rank,
        raw_bm25_score=-1.0,
        anchors=(),
    )


def _acquisition(
    baseline: tuple[SemanticRetrievalHit, ...],
    positional: tuple[SemanticRetrievalHit, ...] = (),
) -> SemanticPositionalAcquisition:
    return SemanticPositionalAcquisition(
        project_id="project-1",
        query=_QUERY,
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


def _positions(
    hits: tuple[SemanticRetrievalHit, ...],
) -> dict[tuple[object, ...], CanonicalPosition]:
    return {
        occurrence_identity(hit): CanonicalPosition(
            occurrence_identity=occurrence_identity(hit),
            document_generation_id=hit.document_generation_id,
            page_id=hit.page_id,
            absolute_start=index * 10,
            source_policy_id="semantic-positional-acquisition-v1",
        )
        for index, hit in enumerate(hits)
    }


def _lexical_results(
    hits: tuple[RetrievalHit, ...],
    *,
    project_id: str = "project-1",
) -> RetrievalResults:
    return RetrievalResults(project_id=project_id, query=_QUERY, hits=hits)


class _LexicalSpy:
    def __init__(self, results: RetrievalResults) -> None:
        self.results = results
        self.calls: list[tuple[Project, str, int]] = []

    def search(self, project: Project, query: str, limit: int) -> RetrievalResults:
        self.calls.append((project, query, limit))
        return self.results


def _run(
    acquisition: SemanticPositionalAcquisition,
    lexical: _LexicalSpy,
    *,
    positions: dict[tuple[object, ...], CanonicalPosition] | None = None,
    policy_id: str | None = PRODUCTION_RANK_ONLY_PARENT_RERANK_POLICY_ID,
):
    return ProductionRankOnlyParentRerankService(
        lexical_service=lexical
    ).rerank_and_select(
        project=_PROJECT,
        query=_QUERY,
        acquisition=acquisition,
        positions=positions if positions is not None else _positions(acquisition.ordered_hits),
        policy_id=policy_id,
    )


def test_production_policy_identity_is_frozen() -> None:
    assert PRODUCTION_RANK_ONLY_PARENT_RERANK_POLICY_ID == "rank-only-parent-rerank-v1"
    assert MAX_PRODUCTION_LEXICAL_RANK_DEPTH == 100


def test_supported_activation_uses_one_explicit_lexical_call() -> None:
    first, second, third = (_semantic_hit(rank=rank) for rank in range(1, 4))
    lexical = _LexicalSpy(
        _lexical_results(
            (
                _lexical_hit(third, rank=1),
                _lexical_hit(first, rank=2),
                _lexical_hit(second, rank=3),
            )
        )
    )

    result = _run(_acquisition((first, second, third)), lexical)

    assert lexical.calls == [(_PROJECT, _QUERY, 100)]
    assert result.activated is True
    assert result.lexical_retrieval_count == 1
    assert [item.hit for item in result.ranked_baseline] == [first, third, second]
    assert result.lexical_result_count == 3


def test_policy_absent_is_exact_no_op_without_lexical_call() -> None:
    baseline = tuple(_semantic_hit(rank=rank) for rank in range(1, 4))
    lexical = _LexicalSpy(_lexical_results(()))

    result = _run(_acquisition(baseline), lexical, policy_id=None)

    assert lexical.calls == []
    assert result.activated is False
    assert result.lexical_retrieval_count == 0
    assert tuple(item.hit for item in result.ranked_baseline) == baseline
    assert result.selected_hits == baseline


def test_unsupported_policy_is_exact_no_op_without_lexical_call() -> None:
    baseline = tuple(_semantic_hit(rank=rank) for rank in range(1, 4))
    lexical = _LexicalSpy(_lexical_results(()))

    result = _run(_acquisition(baseline), lexical, policy_id="other-policy-v1")

    assert lexical.calls == []
    assert result.activated is False
    assert tuple(item.hit for item in result.ranked_baseline) == baseline


def test_unsupported_concern_rejects_before_lexical_call() -> None:
    baseline = tuple(_semantic_hit(rank=rank) for rank in range(1, 3))
    lexical = _LexicalSpy(_lexical_results(()))
    acquisition = replace(_acquisition(baseline), concern_id="other-concern-v1")

    with pytest.raises(ProductionRankOnlyUnsupportedError):
        _run(acquisition, lexical)

    assert lexical.calls == []


@pytest.mark.parametrize(
    "mutate",
    (
        lambda acquisition, hits: replace(
            acquisition,
            baseline_hits=(hits[0].model_copy(update={"rank": 2}), hits[1]),
            examined_hits=(hits[0].model_copy(update={"rank": 2}), hits[1]),
            ordered_hits=(hits[0].model_copy(update={"rank": 2}), hits[1]),
        ),
        lambda acquisition, hits: replace(
            acquisition,
            baseline_hits=(hits[0], hits[0]),
            examined_hits=(hits[0], hits[0]),
            ordered_hits=(hits[0], hits[0]),
        ),
        lambda acquisition, hits: replace(acquisition, project_id="other-project"),
        lambda acquisition, hits: replace(acquisition, query="different-query"),
    ),
)
def test_selected_policy_malformed_state_fails_closed(mutate) -> None:
    hits = (_semantic_hit(rank=1), _semantic_hit(rank=2, span_id="second"))
    lexical = _LexicalSpy(_lexical_results(()))
    acquisition = mutate(_acquisition(hits), hits)
    positions = _positions(acquisition.ordered_hits)
    if len(positions) == 2 and acquisition.project_id != "project-1":
        positions = {}

    with pytest.raises(ValueError):
        _run(acquisition, lexical, positions=positions)

    assert lexical.calls == []


def test_missing_canonical_position_fails_closed_before_lexical_call() -> None:
    baseline = tuple(_semantic_hit(rank=rank) for rank in range(1, 3))
    lexical = _LexicalSpy(_lexical_results(()))
    positions = _positions(baseline)
    positions.pop(occurrence_identity(baseline[1]))

    with pytest.raises(ValueError):
        _run(_acquisition(baseline), lexical, positions=positions)

    assert lexical.calls == []


def test_candidate_cap_is_enforced_before_lexical_work() -> None:
    baseline = tuple(_semantic_hit(rank=rank, span_id=f"span-{rank}") for rank in range(1, 102))
    lexical = _LexicalSpy(_lexical_results(()))

    with pytest.raises(ValueError, match="bound"):
        _run(_acquisition(baseline), lexical, positions={})

    assert lexical.calls == []


def test_lexical_parent_mapping_uses_full_scope_key() -> None:
    baseline = (_semantic_hit(rank=1),)
    wrong_project = _lexical_hit(baseline[0], rank=1, project_id="other-project")
    lexical = _LexicalSpy(_lexical_results((wrong_project,)))

    with pytest.raises(ValueError, match="project"):
        _run(_acquisition(baseline), lexical)


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"file_version_id": "file-2"}, "scope"),
        ({"document_generation_id": "generation-2"}, "scope"),
        ({"page_id": "page-2"}, "scope"),
    ),
)
def test_near_matching_lexical_scope_fails_closed(kwargs, message: str) -> None:
    baseline = (_semantic_hit(rank=1),)
    lexical_hit = _lexical_hit(baseline[0], rank=1, **kwargs)
    lexical = _LexicalSpy(_lexical_results((lexical_hit,)))

    with pytest.raises(ValueError, match=message):
        _run(_acquisition(baseline), lexical)


def test_lexical_only_parent_is_not_added() -> None:
    baseline = (_semantic_hit(rank=1),)
    lexical_only = _semantic_hit(rank=2, span_id="lexical-only", chunk_id="chunk-other")
    lexical = _LexicalSpy(
        _lexical_results(
            (
                _lexical_hit(lexical_only, rank=1),
                _lexical_hit(baseline[0], rank=2),
            )
        )
    )

    result = _run(_acquisition(baseline), lexical)

    assert tuple(item.hit for item in result.ranked_baseline) == baseline
    assert all(item.hit.chunk_id != lexical_only.chunk_id for item in result.ranked_baseline)


def test_lexical_absence_is_explicit_infinity() -> None:
    first = _semantic_hit(rank=1, span_id="first")
    second = _semantic_hit(rank=2, span_id="second")
    lexical = _LexicalSpy(_lexical_results((_lexical_hit(second, rank=1),)))

    result = _run(_acquisition((first, second)), lexical)

    assert [item.hit for item in result.ranked_baseline] == [first, second]
    assert [item.frontier_depth for item in result.ranked_baseline] == [1, 1]


def test_service_has_no_semantic_hybrid_or_model_dependency() -> None:
    import academic_chatbot.retrieval.rank_only_service as module

    assert not {
        "SemanticRetrievalService",
        "HybridRetrievalService",
        "ExactVectorStore",
        "EvidenceBundleService",
        "OfflineEmbedder",
    } & set(module.__dict__)


def test_supported_result_is_repeatable() -> None:
    baseline = tuple(_semantic_hit(rank=rank) for rank in range(1, 4))
    lexical = _LexicalSpy(
        _lexical_results(
            tuple(_lexical_hit(hit, rank=index) for index, hit in enumerate(baseline, 1))
        )
    )

    first = _run(_acquisition(baseline), lexical)
    second = _run(_acquisition(baseline), lexical)

    assert first.ranked_baseline == second.ranked_baseline
    assert first.selected_hits == second.selected_hits
    assert first.lexical_retrieval_count == second.lexical_retrieval_count == 1
    assert first.lexical_result_count == second.lexical_result_count == 3


@pytest.mark.parametrize(
    "lexical_hits",
    (
        lambda hit: (_lexical_hit(hit, rank=1), _lexical_hit(hit, rank=2)),
        lambda hit: (_lexical_hit(hit, rank=2),),
        lambda hit: (RetrievalHit.model_construct(
            project_id=hit.project_id,
            paper_id=hit.paper_id,
            file_version_id=hit.file_version_id,
            document_generation_id=hit.document_generation_id,
            page_id=hit.page_id,
            physical_page_index=0,
            display_page_number=1,
            printed_page_label=None,
            chunk_id=hit.chunk_id,
            chunk_ordinal=0,
            chunk_text="lexical candidate",
            start_offset=0,
            end_offset=17,
            rank=0,
            raw_bm25_score=-1.0,
            anchors=(),
        ),),
        lambda hit: (_lexical_hit(hit, rank=1, chunk_id=""),),
    ),
)
def test_lexical_duplicates_and_invalid_ranks_fail_closed(lexical_hits) -> None:
    baseline = (_semantic_hit(rank=1),)
    lexical = _LexicalSpy(_lexical_results(lexical_hits(baseline[0])))

    with pytest.raises(ValueError):
        _run(_acquisition(baseline), lexical)


def test_only_baseline_occurrences_enter_pareto_reranker() -> None:
    baseline = tuple(_semantic_hit(rank=rank) for rank in range(1, 3))
    auxiliary = _semantic_hit(rank=3, span_id="auxiliary")
    lexical_only = _semantic_hit(rank=4, span_id="lexical-only")
    lexical = _LexicalSpy(
        _lexical_results(
            (
                _lexical_hit(auxiliary, rank=1),
                _lexical_hit(lexical_only, rank=2),
                _lexical_hit(baseline[0], rank=3),
                _lexical_hit(baseline[1], rank=4),
            )
        )
    )

    result = _run(_acquisition(baseline, (auxiliary,)), lexical)

    assert {occurrence_identity(item.hit) for item in result.ranked_baseline} == {
        occurrence_identity(hit) for hit in baseline
    }
    assert auxiliary not in tuple(item.hit for item in result.ranked_baseline)
    assert auxiliary in result.selected_hits


def test_multiple_frontiers_and_all_non_dominated_ties() -> None:
    first = _semantic_hit(rank=1, span_id="first")
    second = _semantic_hit(rank=2, span_id="second")
    third = _semantic_hit(rank=3, span_id="third")
    lexical = _LexicalSpy(
        _lexical_results(
            (
                _lexical_hit(third, rank=1),
                _lexical_hit(first, rank=2),
                _lexical_hit(second, rank=3),
            )
        )
    )

    result = _run(_acquisition((first, second, third)), lexical)

    assert [item.frontier_depth for item in result.ranked_baseline] == [1, 1, 2]
    assert [item.hit for item in result.ranked_baseline] == [first, third, second]

    tied_first = _semantic_hit(rank=1, span_id="tied-first")
    tied_second = _semantic_hit(rank=2, span_id="tied-second")
    tied_lexical = _LexicalSpy(
        _lexical_results(
            (
                _lexical_hit(tied_first, rank=1),
                _lexical_hit(tied_second, rank=2),
            )
        )
    )
    tied_result = _run(_acquisition((tied_first, tied_second)), tied_lexical)

    assert [item.hit for item in tied_result.ranked_baseline] == [tied_first, tied_second]


def test_selector_receives_reranked_prefix_and_preserves_auxiliary() -> None:
    baseline = tuple(_semantic_hit(rank=rank) for rank in range(1, 9))
    auxiliary = _semantic_hit(rank=9, span_id="auxiliary")
    lexical = _LexicalSpy(
        _lexical_results(
            (
                *(_lexical_hit(hit, rank=index) for index, hit in enumerate(reversed(baseline), 1)),
                _lexical_hit(auxiliary, rank=9),
            )
        )
    )

    result = _run(_acquisition(baseline, (auxiliary,)), lexical)

    assert result.selected_hits[:7] == tuple(item.hit for item in result.ranked_baseline[:7])
    assert result.selected_hits[-1] == auxiliary


def test_result_is_immutable_and_keeps_acquisition_rank() -> None:
    baseline = tuple(_semantic_hit(rank=rank) for rank in range(1, 3))
    before = tuple(baseline)
    lexical = _LexicalSpy(
        _lexical_results(
            tuple(
                _lexical_hit(hit, rank=index)
                for index, hit in enumerate(baseline, 1)
            )
        )
    )

    result = _run(_acquisition(baseline), lexical)

    assert tuple(item.hit for item in result.ranked_baseline) == before
    assert [item.original_rank for item in result.ranked_baseline] == [1, 2]
    assert [item.hit.rank for item in result.ranked_baseline] == [1, 2]
    with pytest.raises(FrozenInstanceError):
        result.activated = False  # type: ignore[misc]
