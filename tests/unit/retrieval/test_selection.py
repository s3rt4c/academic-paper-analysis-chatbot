from __future__ import annotations

from dataclasses import replace

import pytest

from academic_chatbot.retrieval.selection import (
    GUARDED_EARLIEST_AUX_SELECTOR_POLICY_ID,
    CanonicalPosition,
    GuardedAuxSelectorError,
    occurrence_identity,
    select_guarded_earliest_auxiliary,
)
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
    start_offset: int = 0,
    end_offset: int = 10,
    raw_semantic_score: float = 1.0,
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
        start_offset=start_offset,
        end_offset=end_offset,
        embedding_span_text=f"candidate-{span}",
        rank=rank,
        raw_semantic_score=raw_semantic_score,
        anchors=(),
    )


def _acquisition(
    baseline: tuple[SemanticRetrievalHit, ...],
    positional: tuple[SemanticRetrievalHit, ...] = (),
    *,
    concern_id: str = "stated-study-objective-v1",
    policy_id: str = "semantic-positional-acquisition-v1",
) -> SemanticPositionalAcquisition:
    return SemanticPositionalAcquisition(
        project_id="project-1",
        query="user query",
        embedding_profile_id="profile-1",
        vector_generation_id="vector-1",
        policy_id=policy_id,
        concern_id=concern_id,
        requested_limit=max(len(baseline), 1),
        examined_hits=baseline + positional,
        baseline_hits=baseline,
        positional_hits=positional,
        ordered_hits=baseline + positional,
    )


def _positions(
    hits: tuple[SemanticRetrievalHit, ...],
    starts: dict[str, int] | None = None,
) -> dict[tuple[object, ...], CanonicalPosition]:
    starts = starts or {}
    return {
        occurrence_identity(hit): CanonicalPosition(
            occurrence_identity=occurrence_identity(hit),
            document_generation_id=hit.document_generation_id,
            page_id=hit.page_id,
            absolute_start=starts.get(hit.embedding_span_id, hit.rank * 10),
            source_policy_id="semantic-positional-acquisition-v1",
        )
        for hit in hits
    }


def _select(
    acquisition: SemanticPositionalAcquisition,
    current: tuple[SemanticRetrievalHit, ...],
    *,
    positions: dict[tuple[object, ...], CanonicalPosition] | None = None,
    selector_policy_id: str = GUARDED_EARLIEST_AUX_SELECTOR_POLICY_ID,
) -> tuple[SemanticRetrievalHit, ...]:
    return select_guarded_earliest_auxiliary(
        acquisition=acquisition,
        current=current,
        positions=positions
        if positions is not None
        else _positions(acquisition.ordered_hits),
        selector_policy_id=selector_policy_id,
    )


def test_policy_identity_is_explicit() -> None:
    assert GUARDED_EARLIEST_AUX_SELECTOR_POLICY_ID == "guarded-earliest-aux-selector-v1"


def test_earliest_positional_only_candidate_replaces_only_slot_eight() -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 9))
    positional = (
        _hit(rank=9, span_id="late", start_offset=1),
        _hit(rank=10, span_id="early", start_offset=2),
        _hit(rank=11, span_id="early-tie", start_offset=3),
    )
    acquisition = _acquisition(baseline, positional)
    positions = _positions(
        acquisition.ordered_hits,
        {"late": 90, "early": 20, "early-tie": 20},
    )

    result = _select(acquisition, baseline, positions=positions)

    assert result[:7] == baseline[:7]
    assert result[7] is positional[1]
    assert len(result) == 8


def test_no_positional_only_candidate_preserves_current_first_eight() -> None:
    current = tuple(_hit(rank=rank) for rank in range(1, 9))
    acquisition = _acquisition(current)

    assert _select(acquisition, current) == current


def test_base_and_both_candidates_are_skipped() -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 9))
    both = baseline[7]
    positional = (
        both,
        _hit(rank=9, span_id="true-aux", start_offset=2),
    )
    acquisition = _acquisition(baseline, positional)
    positions = _positions(acquisition.ordered_hits, {"true-aux": 30})

    result = _select(acquisition, baseline, positions=positions)

    assert result == (*baseline[:7], positional[1])


def test_equal_absolute_start_uses_semantic_rank_then_identity() -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 9))
    lower_rank = _hit(rank=10, span_id="rank-first")
    higher_rank = _hit(rank=11, span_id="identity-second")
    same_rank_a = _hit(rank=12, span_id="a")
    same_rank_b = _hit(rank=12, span_id="b")
    positional = (higher_rank, lower_rank, same_rank_b, same_rank_a)
    acquisition = _acquisition(baseline, positional)
    positions = _positions(
        acquisition.ordered_hits,
        {
            "rank-first": 40,
            "identity-second": 80,
            "a": 90,
            "b": 90,
        },
    )

    result = _select(acquisition, baseline, positions=positions)

    assert result[-1] is lower_rank

    identity_tie_acquisition = _acquisition(baseline, (same_rank_b, same_rank_a))
    identity_tie_positions = _positions(
        identity_tie_acquisition.ordered_hits,
        {"a": 90, "b": 90},
    )

    identity_tie_result = _select(
        identity_tie_acquisition,
        baseline,
        positions=identity_tie_positions,
    )

    assert identity_tie_result[-1] is same_rank_a


def test_current_slot_eight_is_replaceable_without_content_classification() -> None:
    current = tuple(
        _hit(rank=rank, span_id="important" if rank == 8 else None)
        for rank in range(1, 9)
    )
    auxiliary = _hit(rank=9, span_id="auxiliary")
    acquisition = _acquisition(current[:7], (current[7], auxiliary))
    positions = _positions(acquisition.ordered_hits, {"auxiliary": 1})

    result = _select(acquisition, current, positions=positions)

    assert result == (*current[:7], auxiliary)


@pytest.mark.parametrize("current_size", range(8))
def test_underfilled_current_preserves_all_then_appends_one_auxiliary(current_size: int) -> None:
    current = tuple(_hit(rank=rank) for rank in range(1, current_size + 1))
    auxiliary = _hit(rank=20, span_id=f"aux-{current_size}")
    acquisition = _acquisition(current, (auxiliary,))

    result = _select(acquisition, current)

    assert result == (*current, auxiliary)
    assert len(result) <= 8


@pytest.mark.parametrize(
    "positions",
    (
        {},
        {
            ("project-1", "file-1", "generation-1", "page-1", "chunk-span-9",
             "span-9", "profile-1", "vector-1", 0, 10): CanonicalPosition(
                occurrence_identity=(
                    "project-1",
                    "file-1",
                    "generation-1",
                    "page-1",
                    "chunk-span-9",
                    "span-9",
                    "profile-1",
                    "vector-1",
                    0,
                    10,
                ),
                document_generation_id="generation-1",
                page_id="page-1",
                absolute_start=-1,
                source_policy_id="semantic-positional-acquisition-v1",
            )
        },
    ),
)
def test_missing_or_invalid_authoritative_coordinate_fails_closed(
    positions: dict[tuple[object, ...], CanonicalPosition],
) -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 9))
    auxiliary = _hit(rank=9)
    acquisition = _acquisition(baseline, (auxiliary,))

    with pytest.raises(GuardedAuxSelectorError):
        _select(acquisition, baseline, positions=positions)


@pytest.mark.parametrize(
    ("selector_policy_id", "concern_id", "acquisition_policy_id"),
    (
        (
            "ordinary-selection-v1",
            "stated-study-objective-v1",
            "semantic-positional-acquisition-v1",
        ),
        (
            GUARDED_EARLIEST_AUX_SELECTOR_POLICY_ID,
            "other-concern-v1",
            "semantic-positional-acquisition-v1",
        ),
        (
            GUARDED_EARLIEST_AUX_SELECTOR_POLICY_ID,
            "stated-study-objective-v1",
            "other-acquisition-v1",
        ),
    ),
)
def test_guarded_selector_requires_explicit_supported_activation(
    selector_policy_id: str,
    concern_id: str,
    acquisition_policy_id: str,
) -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 9))
    acquisition = _acquisition(
        baseline,
        (_hit(rank=9),),
        concern_id=concern_id,
        policy_id=acquisition_policy_id,
    )

    with pytest.raises(GuardedAuxSelectorError):
        _select(
            acquisition,
            baseline,
            selector_policy_id=selector_policy_id,
        )


def test_lineage_mismatch_fails_closed() -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 9))
    auxiliary = _hit(rank=9, project_id="other-project")
    acquisition = _acquisition(baseline, (auxiliary,))

    with pytest.raises(GuardedAuxSelectorError):
        _select(acquisition, baseline)


def test_duplicate_current_candidate_fails_closed() -> None:
    hit = _hit(rank=1)
    current = (hit, hit)
    acquisition = _acquisition((hit,))

    with pytest.raises(GuardedAuxSelectorError):
        _select(acquisition, current)


def test_output_has_at_most_eight_unique_occurrence_identities() -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 9))
    positional = tuple(_hit(rank=rank, span_id=f"aux-{rank}") for rank in range(9, 20))
    acquisition = _acquisition(baseline, positional)

    result = _select(acquisition, baseline)

    identities = [occurrence_identity(hit) for hit in result]
    assert len(result) <= 8
    assert len(identities) == len(set(identities))


def test_current_values_are_not_mutated() -> None:
    current = tuple(_hit(rank=rank) for rank in range(1, 9))
    acquisition = _acquisition(current, (_hit(rank=9, span_id="aux"),))
    before = tuple(current)

    _select(acquisition, current)

    assert current == before
    assert acquisition.ordered_hits == acquisition.baseline_hits + acquisition.positional_hits


def test_malformed_position_identity_fails_closed() -> None:
    baseline = tuple(_hit(rank=rank) for rank in range(1, 9))
    auxiliary = _hit(rank=9, span_id="auxiliary")
    acquisition = _acquisition(baseline, (auxiliary,))
    positions = _positions(acquisition.ordered_hits)
    positions[occurrence_identity(auxiliary)] = replace(
        positions[occurrence_identity(auxiliary)],
        occurrence_identity=("wrong",),
    )

    with pytest.raises(GuardedAuxSelectorError):
        _select(acquisition, baseline, positions=positions)
