from __future__ import annotations

from dataclasses import replace

import pytest

from academic_chatbot.retrieval.selection import (
    CanonicalPosition,
    GuardedAuxSelectorError,
    occurrence_identity,
    select_guarded_earliest_auxiliary,
)
from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.semantic_position import PositionalAcquisitionSelection
from tests.integration.retrieval.test_project_semantic_search import (
    _active_service,
    _project_value,
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


def test_project_acquisition_can_feed_guarded_selector_without_mutation(tmp_path) -> None:
    service, _, _, _ = _active_service(tmp_path)
    acquisition = service.acquire_positional(
        _project_value(),
        "meaningful query",
        limit=1,
        positional_selection=PositionalAcquisitionSelection(
            concern_id="stated-study-objective-v1"
        ),
    )
    baseline = acquisition.baseline_hits[0]
    auxiliary = baseline.model_copy(
        update={
            "chunk_id": "chunk-auxiliary",
            "embedding_span_id": "span-auxiliary",
            "embedding_span_text": "auxiliary",
            "rank": 2,
            "start_offset": 1,
            "end_offset": 5,
        }
    )
    additive = replace(
        acquisition,
        examined_hits=(*acquisition.examined_hits, auxiliary),
        positional_hits=(auxiliary,),
        ordered_hits=(*acquisition.baseline_hits, auxiliary),
    )
    before = (
        acquisition.examined_hits,
        acquisition.baseline_hits,
        acquisition.positional_hits,
        acquisition.ordered_hits,
        acquisition.policy_id,
        acquisition.concern_id,
    )

    result = select_guarded_earliest_auxiliary(
        acquisition=additive,
        current=(baseline,),
        positions=_positions(additive.ordered_hits),
        selector_policy_id="guarded-earliest-aux-selector-v1",
    )

    assert result == (baseline, auxiliary)
    assert result[0] == baseline
    assert result[1].project_id == acquisition.project_id
    assert result[1].file_version_id == baseline.file_version_id
    assert result[1].document_generation_id == baseline.document_generation_id
    assert result[1].embedding_profile_id == acquisition.embedding_profile_id
    assert result[1].vector_generation_id == acquisition.vector_generation_id
    assert (result[1].start_offset, result[1].end_offset) == (1, 5)
    assert (
        acquisition.examined_hits,
        acquisition.baseline_hits,
        acquisition.positional_hits,
        acquisition.ordered_hits,
        acquisition.policy_id,
        acquisition.concern_id,
    ) == before


def test_project_selection_requires_every_authoritative_coordinate(tmp_path) -> None:
    service, _, _, _ = _active_service(tmp_path)
    acquisition = service.acquire_positional(
        _project_value(),
        "meaningful query",
        limit=1,
        positional_selection=PositionalAcquisitionSelection(
            concern_id="stated-study-objective-v1"
        ),
    )

    with pytest.raises(GuardedAuxSelectorError):
        select_guarded_earliest_auxiliary(
            acquisition=acquisition,
            current=acquisition.baseline_hits,
            positions={},
            selector_policy_id="guarded-earliest-aux-selector-v1",
        )
