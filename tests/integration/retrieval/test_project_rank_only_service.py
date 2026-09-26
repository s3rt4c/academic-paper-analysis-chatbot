from __future__ import annotations

import pytest

from academic_chatbot.domain.library import Project
from academic_chatbot.evidence.models import (
    CandidateOrigin,
    EvidenceBundleRequest,
    PreviewBudget,
)
from academic_chatbot.evidence.references import candidate_ref_from_semantic
from academic_chatbot.evidence.resolver import EvidenceReadResolver
from academic_chatbot.evidence.service import EvidenceBundleService
from academic_chatbot.retrieval.rank_only_service import (
    ProductionRankOnlyError,
    ProductionRankOnlyParentRerankService,
)
from academic_chatbot.retrieval.selection import CanonicalPosition, occurrence_identity
from academic_chatbot.retrieval.semantic import (
    SemanticRetrievalHit,
    SemanticRetrievalService,
)
from academic_chatbot.retrieval.semantic_position import (
    PositionalAcquisitionSelection,
    SemanticPositionalAcquisition,
)
from academic_chatbot.retrieval.service import RetrievalHit, RetrievalResults, RetrievalService
from tests.fixtures.evidence_bundle.database import _SyntheticQueryEmbedder, database
from tests.integration.embeddings.test_vector_publication import _profile

_PROJECT = Project(project_id="project-1", display_name="Synthetic")
_QUERY = "Which result is selected?"


def _semantic_hit(*, rank: int, span_id: str) -> SemanticRetrievalHit:
    return SemanticRetrievalHit.model_construct(
        project_id="project-1",
        paper_id="paper-1",
        file_version_id="file-1",
        document_generation_id="generation-1",
        page_id="page-1",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_id=f"chunk-{span_id}",
        embedding_span_id=span_id,
        embedding_profile_id="profile-1",
        vector_generation_id="vector-1",
        start_offset=rank * 10,
        end_offset=rank * 10 + 5,
        embedding_span_text=f"candidate {span_id}",
        rank=rank,
        raw_semantic_score=1.0,
        anchors=(),
    )


def _lexical_hit(source: SemanticRetrievalHit, *, rank: int) -> RetrievalHit:
    return RetrievalHit.model_construct(
        project_id=source.project_id,
        paper_id=source.paper_id,
        file_version_id=source.file_version_id,
        document_generation_id=source.document_generation_id,
        page_id=source.page_id,
        physical_page_index=source.physical_page_index,
        display_page_number=source.display_page_number,
        printed_page_label=source.printed_page_label,
        chunk_id=source.chunk_id,
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


def _results(hits: tuple[RetrievalHit, ...]) -> RetrievalResults:
    return RetrievalResults(project_id=_PROJECT.project_id, query=_QUERY, hits=hits)


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
    policy_id: str | None = "rank-only-parent-rerank-v1",
):
    return ProductionRankOnlyParentRerankService(
        lexical_service=lexical
    ).rerank_and_select(
        project=_PROJECT,
        query=_QUERY,
        acquisition=acquisition,
        positions=_positions(acquisition.ordered_hits),
        policy_id=policy_id,
    )


def test_production_boundary_reorders_existing_baseline_and_keeps_auxiliary_lane() -> None:
    baseline = tuple(_semantic_hit(rank=rank, span_id=f"span-{rank}") for rank in range(1, 4))
    auxiliary = _semantic_hit(rank=4, span_id="auxiliary")
    lexical = _LexicalSpy(
        _results(
            (
                _lexical_hit(baseline[2], rank=1),
                _lexical_hit(baseline[0], rank=2),
                _lexical_hit(baseline[1], rank=3),
            )
        )
    )

    result = _run(_acquisition(baseline, (auxiliary,)), lexical)

    assert lexical.calls == [(_PROJECT, _QUERY, 100)]
    assert tuple(item.hit for item in result.ranked_baseline) == (
        baseline[0],
        baseline[2],
        baseline[1],
    )
    assert result.selected_hits[:2] == (baseline[0], baseline[2])
    assert result.selected_hits[-1] is auxiliary


def test_default_boundary_is_unchanged_and_does_not_call_lexical() -> None:
    baseline = tuple(_semantic_hit(rank=rank, span_id=f"span-{rank}") for rank in range(1, 4))
    lexical = _LexicalSpy(_results(()))

    result = _run(_acquisition(baseline), lexical, policy_id=None)

    assert lexical.calls == []
    assert tuple(item.hit for item in result.ranked_baseline) == baseline
    assert result.selected_hits == baseline


def test_lexical_scope_mismatch_fails_before_selector_output() -> None:
    baseline = (_semantic_hit(rank=1, span_id="span-1"),)
    wrong_scope = _lexical_hit(baseline[0], rank=1).model_copy(
        update={"document_generation_id": "generation-2"}
    )
    lexical = _LexicalSpy(_results((wrong_scope,)))

    with pytest.raises(ProductionRankOnlyError, match="scope"):
        _run(_acquisition(baseline), lexical)


def test_lexical_results_cannot_expand_candidate_identity_set() -> None:
    baseline = (_semantic_hit(rank=1, span_id="span-1"),)
    lexical_only = _semantic_hit(rank=2, span_id="lexical-only")
    lexical = _LexicalSpy(
        _results(
            (
                _lexical_hit(lexical_only, rank=1),
                _lexical_hit(baseline[0], rank=2),
            )
        )
    )

    result = _run(_acquisition(baseline), lexical)

    assert tuple(item.hit for item in result.ranked_baseline) == baseline
    assert all(item.chunk_id != lexical_only.chunk_id for item in result.selected_hits)


def test_project_production_order_flows_through_evidence_without_identity_change(
    tmp_path,
) -> None:
    db = database(tmp_path)
    profile = _profile()
    db.semantic()
    project = Project(project_id=db.request.scope.project_id, display_name="Synthetic")
    semantic = SemanticRetrievalService(
        data_root=db.paths.data_root,
        profile=profile,
        embedder=_SyntheticQueryEmbedder(profile),
    )
    acquisition = semantic.acquire_positional(
        project,
        "Alpha",
        limit=1,
        positional_selection=PositionalAcquisitionSelection(
            concern_id="stated-study-objective-v1"
        ),
    )
    lexical = RetrievalService(data_root=db.paths.data_root)
    result = ProductionRankOnlyParentRerankService(
        lexical_service=lexical
    ).rerank_and_select(
        project=project,
        query="Alpha",
        acquisition=acquisition,
        positions=_positions(acquisition.ordered_hits),
        policy_id="rank-only-parent-rerank-v1",
    )
    references = tuple(
        candidate_ref_from_semantic(hit).model_copy(update={"reported_rank": position})
        for position, hit in enumerate(result.selected_hits, start=1)
    )
    request = EvidenceBundleRequest(
        scope=db.request.scope,
        origin=CandidateOrigin(
            mode="semantic",
            embedding_profile_id=acquisition.embedding_profile_id,
            vector_generation_id=acquisition.vector_generation_id,
        ),
        candidates=references,
        preview_budget=PreviewBudget(),
    )
    bundle = EvidenceBundleService(
        resolver=EvidenceReadResolver(data_root=db.paths.data_root)
    ).build(request)

    assert result.activated is True
    assert result.lexical_retrieval_count == 1
    assert tuple(disposition.reference for disposition in bundle.dispositions) == references
    assert tuple(entry.citation_label for entry in bundle.entries) == tuple(
        f"E{position}" for position in range(1, len(bundle.entries) + 1)
    )
    assert all(disposition.omission is None for disposition in bundle.dispositions)
