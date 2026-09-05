"""Unit contracts for read-only hybrid retrieval orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from academic_chatbot.domain.library import Project
from academic_chatbot.retrieval.hybrid_models import HybridParentChunkContext
from academic_chatbot.retrieval.hybrid_service import HybridRetrievalService
from academic_chatbot.retrieval.semantic import (
    SemanticArtifactIntegrityError,
    SemanticIndexStaleError,
    SemanticIndexUnavailableError,
    SemanticRetrievalHit,
    SemanticRetrievalResults,
)
from academic_chatbot.retrieval.service import RetrievalHit, RetrievalResults, RetrievalStorageError


def _project() -> Project:
    return Project(project_id="project-1", display_name="Research")


def _lexical(*, chunk_id: str = "chunk-1", rank: int = 1) -> RetrievalHit:
    return RetrievalHit.model_construct(
        project_id="project-1",
        paper_id="paper-1",
        file_version_id="file-1",
        document_generation_id="generation-1",
        page_id="page-1",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_id=chunk_id,
        chunk_ordinal=0,
        chunk_text="alpha beta",
        start_offset=0,
        end_offset=10,
        rank=rank,
        raw_bm25_score=-1.0,
        anchors=(),
    )


def _semantic(*, chunk_id: str = "chunk-1", rank: int = 1) -> SemanticRetrievalHit:
    return SemanticRetrievalHit.model_construct(
        project_id="project-1",
        paper_id="paper-1",
        file_version_id="file-1",
        document_generation_id="generation-1",
        page_id="page-1",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_id=chunk_id,
        embedding_span_id=f"span-{chunk_id}",
        embedding_profile_id="profile-1",
        vector_generation_id="vector-1",
        start_offset=0,
        end_offset=5,
        embedding_span_text="alpha",
        rank=rank,
        raw_semantic_score=0.9,
        anchors=(),
    )


def _parent(chunk_id: str) -> HybridParentChunkContext:
    return HybridParentChunkContext(
        identity={
            "project_id": "project-1",
            "document_generation_id": "generation-1",
            "page_id": "page-1",
            "chunk_id": chunk_id,
        },
        paper_id="paper-1",
        file_version_id="file-1",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_ordinal=0,
        start_offset=0,
        end_offset=10,
        chunk_text="alpha beta",
    )


@dataclass
class _LexicalService:
    hits: tuple[RetrievalHit, ...] = ()
    error: Exception | None = None
    calls: list[tuple[Project, str, int]] = field(default_factory=list)

    def search(self, project: Project, query: str, limit: int) -> RetrievalResults:
        self.calls.append((project, query, limit))
        if self.error is not None:
            raise self.error
        return RetrievalResults(project_id=project.project_id, query=query, hits=self.hits)


@dataclass
class _SemanticService:
    hits: tuple[SemanticRetrievalHit, ...] = ()
    error: Exception | None = None
    calls: list[tuple[Project, str, int]] = field(default_factory=list)

    def search(self, project: Project, query: str, limit: int) -> SemanticRetrievalResults:
        self.calls.append((project, query, limit))
        if self.error is not None:
            raise self.error
        return SemanticRetrievalResults(
            project_id=project.project_id,
            query=query,
            embedding_profile_id="profile-1",
            vector_generation_id="vector-1",
            hits=self.hits,
        )


@dataclass
class _Resolver:
    calls: list[str] = field(default_factory=list)

    def resolve(self, project: Project, candidate: object) -> HybridParentChunkContext:
        assert project == _project()
        chunk_id = candidate.identity.chunk_id  # type: ignore[attr-defined]
        self.calls.append(chunk_id)
        return _parent(chunk_id)


def _service(
    lexical: _LexicalService, semantic: _SemanticService, resolver: _Resolver
) -> HybridRetrievalService:
    return HybridRetrievalService(
        data_root="unused",
        lexical_service=lexical,
        semantic_service=semantic,
        parent_resolver=resolver,
    )


def test_search_passes_the_original_query_and_frozen_depth_to_both_channels() -> None:
    lexical, semantic, resolver = (
        _LexicalService(hits=(_lexical(),)),
        _SemanticService(hits=(_semantic(),)),
        _Resolver(),
    )

    result = _service(lexical, semantic, resolver).search(_project(), "original query", limit=11)

    assert lexical.calls == [(_project(), "original query", 55)]
    assert semantic.calls == [(_project(), "original query", 55)]
    assert len(result.hits) == 1
    assert result.hits[0].trace.fusion_rank == 1


def test_healthy_empty_channels_are_successful_and_stateful() -> None:
    lexical, semantic, resolver = (
        _LexicalService(hits=(_lexical(),)),
        _SemanticService(),
        _Resolver(),
    )

    lexical_only = _service(lexical, semantic, resolver).search(_project(), "query", limit=10)

    assert lexical_only.lexical_state.value == "healthy_results"
    assert lexical_only.semantic_state.value == "healthy_empty"
    assert lexical_only.hits[0].semantic_contribution is None

    semantic_only = _service(
        _LexicalService(), _SemanticService(hits=(_semantic(),)), _Resolver()
    ).search(_project(), "query", limit=10)
    assert semantic_only.lexical_state.value == "healthy_empty"
    assert semantic_only.semantic_state.value == "healthy_results"
    assert semantic_only.hits[0].lexical_contribution is None


@pytest.mark.parametrize(
    "error",
    (
        SemanticIndexUnavailableError("missing"),
        SemanticIndexStaleError("stale"),
        SemanticArtifactIntegrityError("corrupt"),
    ),
)
def test_unhealthy_semantic_channel_never_falls_back_to_lexical(error: Exception) -> None:
    lexical, semantic, resolver = (
        _LexicalService(hits=(_lexical(),)),
        _SemanticService(error=error),
        _Resolver(),
    )

    with pytest.raises(type(error)):
        _service(lexical, semantic, resolver).search(_project(), "query", limit=10)

    assert resolver.calls == []


def test_lexical_backend_failure_never_falls_back_to_semantic() -> None:
    lexical = _LexicalService(error=RetrievalStorageError("database unavailable"))

    with pytest.raises(RetrievalStorageError):
        _service(lexical, _SemanticService(hits=(_semantic(),)), _Resolver()).search(
            _project(), "query", limit=10
        )


def test_service_preserves_task_two_order_and_does_not_rerank_parent_context() -> None:
    lexical = _LexicalService(
        hits=(_lexical(chunk_id="chunk-late", rank=2), _lexical(chunk_id="chunk-early", rank=1))
    )
    semantic, resolver = _SemanticService(), _Resolver()

    result = _service(lexical, semantic, resolver).search(_project(), "query", limit=10)

    assert [hit.identity.chunk_id for hit in result.hits] == ["chunk-early", "chunk-late"]
    assert [hit.trace.fusion_rank for hit in result.hits] == [1, 2]
    assert resolver.calls == ["chunk-early", "chunk-late"]
