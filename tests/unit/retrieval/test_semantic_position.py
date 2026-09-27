from __future__ import annotations

import pytest

from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.semantic_position import (
    MAX_POSITIONAL_AUXILIARY_HITS,
    MAX_POSITIONAL_HITS_EXAMINED,
    PositionalAcquisitionSelection,
    PositionedPage,
    SemanticPositionalAcquisitionError,
    SemanticPositionalPolicyError,
    build_positional_acquisition,
)


def _selection() -> PositionalAcquisitionSelection:
    return PositionalAcquisitionSelection(concern_id="stated-study-objective-v1")


def _page(
    *,
    document_generation_id: str = "generation-1",
    page_id: str = "page-1",
    physical_page_index: int = 0,
    canonical_text_length: int = 100,
) -> PositionedPage:
    return PositionedPage(
        document_generation_id=document_generation_id,
        page_id=page_id,
        physical_page_index=physical_page_index,
        canonical_text_length=canonical_text_length,
    )


def _hit(
    *,
    rank: int,
    start_offset: int = 0,
    end_offset: int = 10,
    page_id: str = "page-1",
    physical_page_index: int = 0,
    document_generation_id: str = "generation-1",
    embedding_span_id: str | None = None,
    raw_semantic_score: float = 1.0,
) -> SemanticRetrievalHit:
    return SemanticRetrievalHit.model_construct(
        project_id="project-1",
        paper_id="paper-1",
        file_version_id="file-1",
        document_generation_id=document_generation_id,
        page_id=page_id,
        physical_page_index=physical_page_index,
        display_page_number=physical_page_index + 1,
        printed_page_label=None,
        chunk_id=f"chunk-{embedding_span_id or rank}",
        embedding_span_id=embedding_span_id or f"span-{rank}",
        embedding_profile_id="profile-1",
        vector_generation_id="vector-1",
        start_offset=start_offset,
        end_offset=end_offset,
        embedding_span_text="candidate",
        rank=rank,
        raw_semantic_score=raw_semantic_score,
        anchors=(),
    )


def _build(
    hits: tuple[SemanticRetrievalHit, ...],
    pages: tuple[PositionedPage, ...] = (_page(),),
    *,
    requested_limit: int = 1,
):
    return build_positional_acquisition(
        project_id="project-1",
        query="user query",
        embedding_profile_id="profile-1",
        vector_generation_id="vector-1",
        examined_hits=hits,
        pages=pages,
        requested_limit=requested_limit,
        selection=_selection(),
    )


def test_supported_selection_has_frozen_policy_identity() -> None:
    result = _build((_hit(rank=1),))

    assert result.policy_id == "semantic-positional-acquisition-v1"
    assert result.concern_id == "stated-study-objective-v1"


@pytest.mark.parametrize(
    "concern_id",
    ("", "unsupported-concern-v1"),
)
def test_missing_or_unsupported_concern_fails_closed(concern_id: str) -> None:
    with pytest.raises(SemanticPositionalPolicyError):
        build_positional_acquisition(
            project_id="project-1",
            query="user query",
            embedding_profile_id="profile-1",
            vector_generation_id="vector-1",
            examined_hits=(_hit(rank=1),),
            pages=(_page(),),
            requested_limit=1,
            selection=PositionalAcquisitionSelection(concern_id=concern_id),
        )


def test_no_positional_selection_is_not_constructed() -> None:
    with pytest.raises(TypeError):
        build_positional_acquisition(  # type: ignore[call-arg]
            project_id="project-1",
            query="user query",
            embedding_profile_id="profile-1",
            vector_generation_id="vector-1",
            examined_hits=(_hit(rank=1),),
            pages=(_page(),),
            requested_limit=1,
        )


def test_single_page_position_uses_code_points() -> None:
    result = _build(
        (
            _hit(rank=1, start_offset=3, end_offset=4),
            _hit(rank=2, start_offset=0, end_offset=1),
        ),
        pages=(_page(canonical_text_length=4),),
    )

    assert [hit.rank for hit in result.positional_hits] == [2]


def test_multi_page_position_adds_one_conceptual_separator() -> None:
    result = _build(
        (
            _hit(
                rank=1,
                page_id="page-2",
                physical_page_index=1,
                start_offset=3,
                end_offset=4,
            ),
            _hit(
                rank=2,
                page_id="page-2",
                physical_page_index=1,
                start_offset=0,
                end_offset=1,
            ),
        ),
        pages=(
            _page(page_id="page-1", canonical_text_length=1),
            _page(page_id="page-2", physical_page_index=1, canonical_text_length=4),
        ),
    )

    assert result.positional_hits == ()


def test_documents_are_positioned_independently() -> None:
    result = _build(
        (
            _hit(
                rank=1,
                document_generation_id="generation-1",
                page_id="page-1",
                start_offset=90,
                end_offset=99,
            ),
            _hit(
                rank=2,
                document_generation_id="generation-2",
                page_id="page-2",
                start_offset=0,
                end_offset=10,
            ),
        ),
        pages=(
            _page(document_generation_id="generation-1", canonical_text_length=100),
            _page(
                document_generation_id="generation-2",
                page_id="page-2",
                canonical_text_length=100,
            ),
        ),
        requested_limit=1,
    )

    assert [hit.document_generation_id for hit in result.positional_hits] == [
        "generation-2"
    ]


@pytest.mark.parametrize(
    ("start_offset", "expected"),
    ((0, True), (24, True), (25, False), (26, False)),
)
def test_position_boundary_uses_exact_integer_comparison(
    start_offset: int, expected: bool
) -> None:
    result = _build(
        (
            _hit(rank=1, start_offset=90, end_offset=99),
            _hit(rank=2, start_offset=start_offset, end_offset=start_offset + 1),
        ),
        pages=(_page(canonical_text_length=100),),
    )

    assert (result.positional_hits == (result.examined_hits[1],)) is expected


def test_baseline_hits_are_preserved_before_auxiliary_hits() -> None:
    baseline = _hit(rank=1, start_offset=90, end_offset=99)
    auxiliary = _hit(rank=2, start_offset=0, end_offset=10)

    result = _build((baseline, auxiliary))

    assert result.baseline_hits == (baseline,)
    assert result.positional_hits == (auxiliary,)
    assert result.ordered_hits == (baseline, auxiliary)


def test_auxiliary_hits_are_rank_ordered_and_capped_at_32() -> None:
    hits = (
        _hit(rank=1, start_offset=90, end_offset=99),
        *(_hit(rank=rank, start_offset=rank, end_offset=rank + 1) for rank in range(2, 42)),
    )

    result = _build(hits, pages=(_page(canonical_text_length=1000),))

    assert len(result.examined_hits) == 41
    assert len(result.positional_hits) == MAX_POSITIONAL_AUXILIARY_HITS
    assert [hit.rank for hit in result.positional_hits] == list(range(2, 34))


def test_baseline_duplicate_is_not_repeated_in_auxiliary() -> None:
    baseline = _hit(
        rank=1,
        start_offset=0,
        end_offset=10,
        embedding_span_id="same-span",
    )
    duplicate = _hit(
        rank=2,
        start_offset=0,
        end_offset=10,
        embedding_span_id="same-span",
    )

    result = _build((baseline, duplicate))

    assert result.baseline_hits == (baseline,)
    assert result.positional_hits == ()


def test_baseline_duplicate_is_not_repeated_inside_baseline() -> None:
    baseline = _hit(
        rank=1,
        start_offset=0,
        end_offset=10,
        embedding_span_id="same-span",
    )
    duplicate = _hit(
        rank=2,
        start_offset=0,
        end_offset=10,
        embedding_span_id="same-span",
    )

    result = _build((baseline, duplicate), requested_limit=2)

    assert result.baseline_hits == (baseline,)
    assert result.ordered_hits == (baseline,)


def test_contradictory_duplicate_lineage_fails_closed() -> None:
    first = _hit(
        rank=1,
        start_offset=0,
        end_offset=10,
        embedding_span_id="same-span",
        raw_semantic_score=1.0,
    )
    contradictory = _hit(
        rank=2,
        start_offset=0,
        end_offset=10,
        embedding_span_id="same-span",
        raw_semantic_score=0.5,
    )

    with pytest.raises(SemanticPositionalAcquisitionError):
        _build((first, contradictory))


def test_limit_above_100_fails_closed() -> None:
    with pytest.raises(SemanticPositionalAcquisitionError):
        _build((_hit(rank=1),), requested_limit=MAX_POSITIONAL_HITS_EXAMINED + 1)


def test_empty_examined_hits_return_empty_tuples() -> None:
    result = _build((), pages=())

    assert result.examined_hits == ()
    assert result.baseline_hits == ()
    assert result.positional_hits == ()
    assert result.ordered_hits == ()


@pytest.mark.parametrize(
    ("start_offset", "end_offset", "page_length"),
    ((-1, 1, 100), (10, 10, 100), (10, 101, 100), (0, 1, 0)),
)
def test_invalid_page_metadata_or_offsets_fail_closed(
    start_offset: int, end_offset: int, page_length: int
) -> None:
    with pytest.raises(SemanticPositionalAcquisitionError):
        _build(
            (_hit(rank=1, start_offset=start_offset, end_offset=end_offset),),
            pages=(_page(canonical_text_length=page_length),),
        )


def test_hit_physical_page_must_match_position_metadata() -> None:
    with pytest.raises(SemanticPositionalAcquisitionError):
        _build(
            (_hit(rank=1, physical_page_index=1),),
            pages=(_page(),),
        )
