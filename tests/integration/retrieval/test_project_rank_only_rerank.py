from __future__ import annotations

import pytest

from academic_chatbot.evidence.resolver import EvidenceReadResolver
from academic_chatbot.evidence.service import EvidenceBundleService
from academic_chatbot.retrieval.rank_only_rerank import (
    apply_rank_only_parent_rerank,
)
from academic_chatbot.retrieval.selection import (
    CanonicalPosition,
    occurrence_identity,
    select_guarded_earliest_auxiliary,
)
from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.semantic_position import SemanticPositionalAcquisition
from tests.fixtures.evidence_bundle.database import database


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


def _positions(
    hits: tuple[SemanticRetrievalHit, ...],
) -> dict[tuple[object, ...], CanonicalPosition]:
    return {
        occurrence_identity(hit): CanonicalPosition(
            occurrence_identity=occurrence_identity(hit),
            document_generation_id=hit.document_generation_id,
            page_id=hit.page_id,
            absolute_start=hit.start_offset,
            source_policy_id="semantic-positional-acquisition-v1",
        )
        for hit in hits
    }


def test_explicit_lexical_annotation_does_not_add_candidates() -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 4))
    lexical_only = _hit(rank=4, span_id="lexical-only")

    result = apply_rank_only_parent_rerank(
        acquisition=_acquisition(baseline),
        lexical_ranks={
            _parent_key(baseline[0]): 2,
            _parent_key(lexical_only): 1,
        },
        policy_id="rank-only-parent-rerank-v1",
    )

    assert {occurrence_identity(item.hit) for item in result} == {
        occurrence_identity(hit) for hit in baseline
    }
    assert lexical_only.embedding_span_id not in {
        item.hit.embedding_span_id for item in result
    }


def test_reranked_baseline_prefix_is_valid_selector_input() -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 9))
    auxiliary = _hit(rank=9, span_id="auxiliary")
    acquisition = _acquisition(baseline, (auxiliary,))
    ranks = {
        _parent_key(hit): lexical_rank
        for hit, lexical_rank in zip(baseline, (8, 7, 1, 6, 5, 4, 3, 2), strict=True)
    }

    reranked = apply_rank_only_parent_rerank(
        acquisition=acquisition,
        lexical_ranks=ranks,
        policy_id="rank-only-parent-rerank-v1",
    )
    current = tuple(item.hit for item in reranked[:8])
    selected = select_guarded_earliest_auxiliary(
        acquisition=acquisition,
        current=current,
        positions=_positions(acquisition.ordered_hits),
        selector_policy_id="guarded-earliest-aux-selector-v1",
    )

    assert selected[:7] == current[:7]
    assert selected[-1] is auxiliary
    assert auxiliary not in tuple(item.hit for item in reranked)


def test_selector_resolver_packing_and_citations_keep_identity(tmp_path) -> None:
    db = database(tmp_path)
    request = db.retrieved(mode="semantic")
    reordered = tuple(
        candidate.model_copy(update={"reported_rank": position})
        for position, candidate in enumerate(reversed(request.candidates), start=1)
    )

    bundle = EvidenceBundleService(
        resolver=EvidenceReadResolver(data_root=db.paths.data_root)
    ).build(request.model_copy(update={"candidates": reordered}))

    assert {candidate.parent for candidate in reordered} == {
        candidate.parent for candidate in request.candidates
    }
    assert tuple(entry.citation_label for entry in bundle.entries) == tuple(
        f"E{position}" for position in range(1, len(bundle.entries) + 1)
    )
    assert tuple(disposition.reference for disposition in bundle.dispositions) == reordered
    assert all(disposition.omission is None for disposition in bundle.dispositions)


def test_default_runtime_order_is_unchanged_without_activation() -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 4))
    acquisition = _acquisition(baseline)

    result = apply_rank_only_parent_rerank(
        acquisition=acquisition,
        lexical_ranks={_parent_key(baseline[0]): 1},
    )

    assert tuple(item.hit for item in result) == baseline
    assert [item.post_rank for item in result] == [1, 2, 3]


def test_rank_only_module_has_no_retrieval_or_embedding_service() -> None:
    import academic_chatbot.retrieval.rank_only_rerank as module

    assert not {
        "RetrievalService",
        "SemanticRetrievalService",
        "ExactVectorStore",
    } & set(module.__dict__)


def test_project_file_version_and_generation_are_isolated() -> None:
    baseline = (_hit(rank=1),)
    acquisition = _acquisition(baseline)
    wrong_scope = (
        "other-project",
        "file-1",
        "generation-1",
        "page-1",
        "chunk-span-1",
    )

    with pytest.raises(ValueError, match="project scope"):
        apply_rank_only_parent_rerank(
            acquisition=acquisition,
            lexical_ranks={wrong_scope: 1},
            policy_id="rank-only-parent-rerank-v1",
        )


def test_acquisition_tuples_are_not_mutated() -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 4))
    positional = (_hit(rank=4, span_id="auxiliary"),)
    acquisition = _acquisition(baseline, positional)
    before = (
        acquisition.examined_hits,
        acquisition.baseline_hits,
        acquisition.positional_hits,
        acquisition.ordered_hits,
    )

    apply_rank_only_parent_rerank(
        acquisition=acquisition,
        lexical_ranks={_parent_key(baseline[0]): 3, _parent_key(baseline[1]): 1},
        policy_id="rank-only-parent-rerank-v1",
    )

    assert (
        acquisition.examined_hits,
        acquisition.baseline_hits,
        acquisition.positional_hits,
        acquisition.ordered_hits,
    ) == before
