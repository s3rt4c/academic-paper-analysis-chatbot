"""End-to-end offline scoped acquisition over real synthetic published authority."""

import json

import pytest

from academic_chatbot.library.repository import ProjectRepository
from academic_chatbot.retrieval.scoped_models import ScopedRetrievalError
from academic_chatbot.retrieval.scoped_serialization import canonical_scoped_result_bytes
from academic_chatbot.retrieval.scoped_service import SourceScopedRetrievalService
from academic_chatbot.retrieval.service import RetrievalService
from tests.fixtures.retrieval.scoped import SparseEmbedder, crowded


def service(fixture):
    return SourceScopedRetrievalService(
        data_root=fixture.db.paths.data_root,
        repository=ProjectRepository(fixture.db.paths),
        lexical_service=RetrievalService(data_root=fixture.db.paths.data_root),
        semantic_service=fixture.semantic,
    )


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_scoped_search_real_published_identity_channels_and_repeatability(tmp_path, mode):
    fixture = crowded(tmp_path, vectors=mode != "lexical")
    reader = service(fixture)
    request = reader.prepare_request(
        project_id="project-one",
        paper_id="selected",
        query="Alpha",
        mode=mode,
        embedding_profile_id=fixture.semantic.profile.embedding_profile_id
        if mode != "lexical"
        else None,
    )
    result = reader.search_scoped(request)
    assert result.source == fixture.scope and result.request_fingerprint == request.fingerprint
    assert len(result.hits) == 2 and result.retrieval_completeness == "not_assessed"
    assert canonical_scoped_result_bytes(result) == canonical_scoped_result_bytes(
        reader.search_scoped(request)
    )
    depth = 50 if mode == "hybrid" else 10
    for channel in ("lexical", "semantic"):
        state = getattr(result.channels, channel)
        if mode == channel or mode == "hybrid":
            assert (state.state, state.requested_limit, state.returned_count) == (
                "nonempty",
                depth,
                2,
            )
        else:
            assert (state.state, state.requested_limit, state.returned_count) == (
                "not_run",
                None,
                0,
            )
    payload = json.loads(canonical_scoped_result_bytes(result))
    assert str(fixture.db.paths.data_root) not in json.dumps(payload)
    assert "inspection" not in payload and "bundle" not in payload
    reader.recheck_current(request)


def test_fresh_vector_release_observes_commit_while_acquisition_snapshot_held(tmp_path):
    fixture = crowded(tmp_path, vectors=True)
    reader = service(fixture)
    request = fixture.request("semantic")

    class CommittingEmbedder(SparseEmbedder):
        def embed_queries(self, texts):
            fixture.db.execute(
                "DELETE FROM generation_publications WHERE file_version_id!=?",
                (fixture.scope.file_version_id,),
            )
            return super().embed_queries(texts)

    fixture.semantic._embedder = CommittingEmbedder(fixture.semantic.profile)
    with pytest.raises(ScopedRetrievalError) as caught:
        reader.search_scoped(request)
    assert caught.value.code == "VECTOR_NOT_CURRENT"


def test_honest_semantic_empty_differs_from_unavailable(tmp_path):
    from academic_chatbot.retrieval.semantic import SemanticRetrievalService
    from tests.fixtures.evidence_bundle.database import _SyntheticQueryEmbedder, database
    from tests.integration.embeddings.test_vector_publication import _profile

    db = database(tmp_path, empty=True)
    db.semantic()
    profile = _profile()
    semantic = SemanticRetrievalService(
        data_root=db.paths.data_root, profile=profile, embedder=_SyntheticQueryEmbedder(profile)
    )
    reader = SourceScopedRetrievalService(
        data_root=db.paths.data_root,
        repository=ProjectRepository(db.paths),
        lexical_service=RetrievalService(data_root=db.paths.data_root),
        semantic_service=semantic,
    )
    request = reader.prepare_request(
        project_id="project-one",
        paper_id="paper-one",
        query="Alpha",
        mode="hybrid",
        embedding_profile_id=profile.embedding_profile_id,
    )
    result = reader.search_scoped(request)
    assert not result.hits
    assert result.channels.lexical.state == result.channels.semantic.state == "empty"
    db.execute("DELETE FROM vector_generation_publications")
    with pytest.raises(ScopedRetrievalError) as caught:
        reader.search_scoped(request)
    assert caught.value.code == "VECTOR_UNAVAILABLE"


@pytest.mark.parametrize("bound", ["mapping", "snapshot", "profile", "artifact"])
def test_semantic_authority_bounds_before_expensive_acquisition(tmp_path, monkeypatch, bound):
    from academic_chatbot.evidence import models

    fixture = crowded(tmp_path, vectors=True)
    request = fixture.request("semantic")

    def forbidden(*args, **kwargs):
        raise AssertionError("embedding/hit materialized before resource admission")

    monkeypatch.setattr(fixture.semantic, "_embed_query", forbidden)
    if bound == "mapping":
        monkeypatch.setattr(models, "MAX_VECTOR_ROWS", 1)
    elif bound == "snapshot":
        monkeypatch.setattr(models, "MAX_SOURCE_SNAPSHOT_ROWS", 1)
    elif bound == "profile":
        monkeypatch.setattr(models, "MAX_PROFILE_BYTES", 1)
    else:
        monkeypatch.setattr(models, "MAX_VECTOR_METADATA_BYTES", 1)
    with pytest.raises(ScopedRetrievalError) as caught:
        fixture.semantic.search_scoped(request)
    assert caught.value.code == "RESOURCE_LIMIT"


def test_selected_semantic_page_preflight_precedes_full_text(tmp_path, monkeypatch):
    fixture = crowded(tmp_path, vectors=True)
    request = fixture.request("semantic")
    page = fixture.db.rows(
        "SELECT page_id FROM pages WHERE document_generation_id=?",
        (fixture.scope.document_generation_id,),
    )[0][0]
    fixture.db.corrupt(
        "UPDATE pages SET canonical_text=? WHERE page_id=?", ("Alpha " + "x" * 1048576, page)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized selected text materialized")

    monkeypatch.setattr("academic_chatbot.retrieval.semantic._semantic_hit", forbidden)
    with pytest.raises(ScopedRetrievalError) as caught:
        fixture.semantic.search_scoped(request)
    assert caught.value.code == "RESOURCE_LIMIT"


@pytest.mark.parametrize("digest", ["f" * 64, "z" * 64])
def test_fresh_registration_remains_bound_to_admitted_embedder(tmp_path, digest):
    fixture = crowded(tmp_path, vectors=True)
    reader = SourceScopedRetrievalService(
        data_root=fixture.db.paths.data_root,
        repository=ProjectRepository(fixture.db.paths),
        lexical_service=RetrievalService(data_root=fixture.db.paths.data_root),
        semantic_service=fixture.semantic,
    )
    request = reader.prepare_request(
        project_id="project-one",
        paper_id="selected",
        query="Alpha",
        mode="semantic",
        embedding_profile_id=fixture.semantic.profile.embedding_profile_id,
    )
    fixture.db.corrupt("UPDATE embedding_profiles SET artifact_manifest_sha256=?", (digest,))
    for acquire in (lambda: reader.recheck_current(request), lambda: reader.search_scoped(request)):
        with pytest.raises(ScopedRetrievalError) as caught:
            acquire()
        assert caught.value.code == "PROFILE_MISMATCH"


def test_scoped_model_factory_preflights_profile_before_legacy_loader(tmp_path, monkeypatch):
    from academic_chatbot.evidence import models
    from academic_chatbot.retrieval.semantic import SemanticRetrievalService

    fixture = crowded(tmp_path, vectors=True)
    monkeypatch.setattr(models, "MAX_PROFILE_BYTES", 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("registered profile loaded before bounded admission")

    monkeypatch.setattr("academic_chatbot.retrieval.semantic.json.loads", forbidden)
    with pytest.raises(ScopedRetrievalError) as caught:
        SemanticRetrievalService.open_from_model_root(
            data_root=fixture.db.paths.data_root,
            project_id="project-one",
            profile_id=fixture.semantic.profile.embedding_profile_id,
            model_root=tmp_path,
            scoped=True,
        )
    assert caught.value.code == "RESOURCE_LIMIT"
