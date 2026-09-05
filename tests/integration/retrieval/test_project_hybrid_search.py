"""Integration contracts for read-only hybrid evidence resolution."""

from __future__ import annotations

# ruff: noqa: E501
import pytest

from academic_chatbot.db.connection import connect_project_database
from academic_chatbot.retrieval.hybrid_fusion import fuse_candidates
from academic_chatbot.retrieval.hybrid_models import HybridChannelMembership
from academic_chatbot.retrieval.hybrid_service import (
    HybridRetrievalIntegrityError,
    HybridRetrievalService,
    _ParentChunkEvidenceResolver,
)
from academic_chatbot.retrieval.service import RetrievalService
from tests.integration.retrieval.test_project_semantic_search import (
    _active_service,
    _project_value,
)


def test_hybrid_search_resolves_one_current_parent_with_separate_channel_evidence(
    tmp_path,
) -> None:
    semantic, _, repository, _ = _active_service(tmp_path)
    paths = repository._paths  # type: ignore[attr-defined]
    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        connection.execute("INSERT INTO chunk_fts VALUES ('chunk-one', 'alpha beta gamma')")
    finally:
        connection.close()
    service = HybridRetrievalService(
        data_root=paths.data_root,
        lexical_service=RetrievalService(data_root=paths.data_root),
        semantic_service=semantic,
    )

    results = service.search(_project_value(), "alpha", limit=10)

    assert len(results.hits) == 1
    hit = results.hits[0]
    assert hit.channel_membership is HybridChannelMembership.BOTH
    assert hit.parent_chunk.chunk_text == "alpha beta gamma"
    assert hit.lexical_contribution is not None
    assert hit.semantic_contribution is not None
    assert hit.lexical_contribution.lexical_hit.chunk_text == "alpha beta gamma"
    assert hit.semantic_contribution.semantic_hit.embedding_span_text == "alpha beta gamma"


def test_hybrid_search_keeps_semantic_only_evidence_when_lexical_is_healthy_empty(tmp_path) -> None:
    semantic, _, repository, _ = _active_service(tmp_path)
    paths = repository._paths  # type: ignore[attr-defined]
    service = HybridRetrievalService(
        data_root=paths.data_root,
        lexical_service=RetrievalService(data_root=paths.data_root),
        semantic_service=semantic,
    )

    results = service.search(_project_value(), "no-lexical-match", limit=10)

    assert results.lexical_state.value == "healthy_empty"
    assert results.semantic_state.value == "healthy_results"
    assert len(results.hits) == 1
    assert results.hits[0].lexical_contribution is None
    assert results.hits[0].semantic_contribution is not None


def test_parent_resolver_refuses_a_candidate_generation_after_active_generation_changes(tmp_path) -> None:
    semantic, _, repository, _ = _active_service(tmp_path)
    paths = repository._paths  # type: ignore[attr-defined]
    candidate = fuse_candidates((), semantic.search(_project_value(), "query").hits, final_limit=10)[0]
    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        connection.execute(
            "INSERT INTO document_generations VALUES ('generation-next', 'file-one', 'native-v2', '2026-09-03T00:00:00Z')"
        )
        connection.execute(
            "INSERT INTO pages (page_id, document_generation_id, page_number, canonical_text, canonical_text_sha256, needs_ocr) VALUES ('page-next', 'generation-next', 1, '', ?, 1)",
            ("f" * 64,),
        )
        connection.execute(
            "UPDATE generation_publications SET document_generation_id = 'generation-next' WHERE file_version_id = 'file-one'"
        )
    finally:
        connection.close()

    with pytest.raises(HybridRetrievalIntegrityError, match="no longer"):
        _ParentChunkEvidenceResolver(data_root=paths.data_root).resolve(_project_value(), candidate)
