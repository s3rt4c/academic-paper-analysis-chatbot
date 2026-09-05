"""Security contracts for hybrid parent evidence resolution."""

from __future__ import annotations

from dataclasses import replace

import pytest

from academic_chatbot.retrieval.hybrid_fusion import fuse_candidates
from academic_chatbot.retrieval.hybrid_models import HybridCandidateKey
from academic_chatbot.retrieval.hybrid_service import (
    HybridRetrievalIntegrityError,
    _ParentChunkEvidenceResolver,
)
from tests.integration.retrieval.test_project_semantic_search import (
    _active_service,
    _project_value,
)


def test_parent_resolver_rejects_a_crafted_cross_project_candidate_key(tmp_path) -> None:
    semantic, _, repository, _ = _active_service(tmp_path)
    semantic_hits = semantic.search(_project_value(), "query").hits
    candidate = fuse_candidates((), semantic_hits, final_limit=10)[0]
    forged = replace(
        candidate,
        identity=HybridCandidateKey(
            project_id="project-other",
            document_generation_id=candidate.identity.document_generation_id,
            page_id=candidate.identity.page_id,
            chunk_id=candidate.identity.chunk_id,
        ),
    )
    paths = repository._paths  # type: ignore[attr-defined]

    with pytest.raises(HybridRetrievalIntegrityError, match="contribution"):
        _ParentChunkEvidenceResolver(data_root=paths.data_root).resolve(_project_value(), forged)
