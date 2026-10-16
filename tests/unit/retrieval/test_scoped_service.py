"""Real source-scoped acquisition and independent starvation regressions."""

import pytest

from academic_chatbot.retrieval.scoped_models import ScopedRetrievalError
from academic_chatbot.retrieval.service import RetrievalService
from tests.fixtures.retrieval.scoped import crowded, lexical_oracle


def test_lexical_pre_cap_starvation_and_fixed_storage_ties(tmp_path):
    fixture = crowded(tmp_path)
    lexical = RetrievalService(data_root=fixture.db.paths.data_root)
    before = lexical.search(fixture.project, "Alpha", limit=50)
    assert len(before.hits) == 50
    assert all(h.paper_id == "distractor" for h in before.hits)
    assert all(h.paper_id == "distractor" for h in before.hits[:10])
    expected = lexical_oracle(fixture)
    assert len(expected) == 2
    assert hasattr(lexical, "search_scoped"), "scoped lexical acquisition absent"
    result = lexical.search_scoped(fixture.request())
    assert [h.chunk_id for h in result.hits] == [r[0] for r in expected[:10]]
    assert [h.raw_bm25_score for h in result.hits] == [r[1] for r in expected[:10]]
    assert [h.rank for h in result.hits] == [1, 2]
    assert result == lexical.search_scoped(fixture.request())
    assert all(h.paper_id == "selected" and h.anchors for h in result.hits)
    assert lexical.search(fixture.project, "Alpha", limit=50) == before


@pytest.mark.parametrize(
    "corruption", ["fts", "range", "oversized", "anchors", "indexed_oversized"]
)
def test_lexical_scoped_corruption_and_resource_preflight(tmp_path, monkeypatch, corruption):
    fixture = crowded(tmp_path)
    lexical = RetrievalService(data_root=fixture.db.paths.data_root)
    assert hasattr(lexical, "search_scoped"), "scoped lexical acquisition absent"
    chunk = lexical_oracle(fixture)[0][0]
    page = fixture.db.rows("SELECT page_id FROM chunks WHERE chunk_id=?", (chunk,))[0][0]
    if corruption == "fts":
        fixture.db.corrupt(
            "UPDATE chunk_fts SET chunk_text='Alpha wrong' WHERE chunk_id=?", (chunk,)
        )
    elif corruption == "indexed_oversized":
        fixture.db.corrupt(
            "UPDATE chunk_fts SET chunk_text=? WHERE chunk_id=?", ("Alpha " + "x" * 1048576, chunk)
        )
    elif corruption == "range":
        fixture.db.corrupt("UPDATE chunks SET start_offset=1 WHERE chunk_id=?", (chunk,))
    elif corruption == "oversized":
        fixture.db.corrupt(
            "UPDATE pages SET canonical_text=? WHERE page_id=?", ("Alpha " + "x" * 1048576, page)
        )
    else:
        fixture.db.corrupt(
            """WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<130)
               INSERT INTO page_anchors SELECT a.page_anchor_id||'-extra-'||n.x,
               a.evidence_id||'-extra-'||n.x, a.page_id,0,n.x,a.anchor_text,
               a.anchor_text_sha256,a.boxes_sha256,a.x0,a.top,a.x1,a.bottom
               FROM (SELECT * FROM page_anchors WHERE page_id=? LIMIT 1) a CROSS JOIN n
               WHERE NOT EXISTS (SELECT 1 FROM page_anchors b
                 WHERE b.page_id=a.page_id AND b.char_start=0 AND b.char_end=n.x)""",
            (page,),
        )
    if corruption in {"oversized", "anchors", "indexed_oversized"}:

        def forbidden(*args, **kwargs):
            raise AssertionError("full text reconstructed before resource preflight")

        monkeypatch.setattr("academic_chatbot.retrieval.service._hit_from_row", forbidden)
    with pytest.raises(ScopedRetrievalError) as caught:
        lexical.search_scoped(fixture.request())
    assert caught.value.code == (
        "RESOURCE_LIMIT"
        if corruption in {"oversized", "anchors", "indexed_oversized"}
        else "STORAGE_INTEGRITY"
    )


def coordinator(fixture, **kwargs):
    import importlib

    assert importlib.util.find_spec("academic_chatbot.retrieval.scoped_service"), (
        "scoped authority coordinator absent"
    )
    from academic_chatbot.library.repository import ProjectRepository

    module = importlib.import_module("academic_chatbot.retrieval.scoped_service")
    return module.SourceScopedRetrievalService(
        data_root=fixture.db.paths.data_root,
        repository=ProjectRepository(fixture.db.paths),
        lexical_service=kwargs.pop(
            "lexical_service", RetrievalService(data_root=fixture.db.paths.data_root)
        ),
        semantic_service=fixture.semantic,
        **kwargs,
    )


def test_coordinator_binds_authority_and_rejects_forged_and_changed_source(tmp_path):
    fixture = crowded(tmp_path)
    service = coordinator(fixture)
    request = service.prepare_request(
        project_id="project-one", paper_id="selected", query="Alpha", mode="lexical"
    )
    assert request.scope == fixture.scope
    result = service.search_scoped(request)
    assert result.request_fingerprint == request.fingerprint and len(result.hits) == 2
    forged = request.model_copy(
        update={"scope": request.scope.model_copy(update={"document_generation_id": "stale"})}
    )
    with pytest.raises(ScopedRetrievalError) as caught:
        service.search_scoped(forged)
    assert caught.value.code == "SOURCE_NOT_CURRENT"
    fixture.db.execute(
        "DELETE FROM generation_publications WHERE file_version_id=?",
        (fixture.scope.file_version_id,),
    )
    with pytest.raises(ScopedRetrievalError) as caught:
        service.recheck_current(request)
    assert caught.value.code == "SOURCE_NOT_CURRENT"


def test_coordinator_release_check_discards_acquired_hits(tmp_path):
    fixture = crowded(tmp_path)

    class ChangingLexical:
        def search_scoped(self, request):
            hits = RetrievalService(data_root=fixture.db.paths.data_root).search_scoped(request)
            fixture.db.execute(
                "DELETE FROM generation_publications WHERE file_version_id=?",
                (fixture.scope.file_version_id,),
            )
            return hits

    service = coordinator(fixture, lexical_service=ChangingLexical())
    request = fixture.request()
    with pytest.raises(ScopedRetrievalError) as caught:
        service.search_scoped(request)
    assert caught.value.code == "SOURCE_NOT_CURRENT"


def test_coordinator_safe_unexpected_exception_and_lazy_source_first(tmp_path):
    fixture = crowded(tmp_path)

    class BrokenLexical:
        def search_scoped(self, request):
            raise RuntimeError("SECRET CANARY C:/private/path")

    service = coordinator(fixture, lexical_service=BrokenLexical())
    with pytest.raises(ScopedRetrievalError) as caught:
        service.search_scoped(fixture.request())
    assert caught.value.code == "INTERNAL_INVARIANT_FAILURE"
    assert "SECRET" not in str(caught.value)
    import importlib

    from academic_chatbot.library.repository import ProjectRepository

    module = importlib.import_module("academic_chatbot.retrieval.scoped_service")

    def forbidden():
        raise AssertionError("BGE opened before source admission")

    service = module.SourceScopedRetrievalService(
        data_root=fixture.db.paths.data_root,
        repository=ProjectRepository(fixture.db.paths),
        lexical_service=RetrievalService(data_root=fixture.db.paths.data_root),
        semantic_service=forbidden,
    )
    with pytest.raises(ScopedRetrievalError) as caught:
        service.prepare_request(
            project_id="project-one",
            paper_id="absent",
            query="Alpha",
            mode="semantic",
            embedding_profile_id="synthetic",
        )
    assert caught.value.code == "SOURCE_NOT_FOUND"


def test_scoped_canonical_output_all_or_nothing_and_terminal_lf_bound(tmp_path, monkeypatch):
    import importlib

    fixture = crowded(tmp_path)
    result = coordinator(fixture).search_scoped(fixture.request())
    assert importlib.util.find_spec("academic_chatbot.retrieval.scoped_serialization")
    module = importlib.import_module("academic_chatbot.retrieval.scoped_serialization")
    payload = module.canonical_scoped_result_bytes(result)
    assert payload.startswith(b'{"') and not payload.endswith(b"\n")
    assert payload == module.canonical_scoped_result_bytes(result)
    assert b'"semantic_identity":null' in payload
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", len(payload) + 1)
    assert module.canonical_scoped_result_bytes(result) == payload
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", len(payload))
    with pytest.raises(ScopedRetrievalError) as caught:
        module.canonical_scoped_result_bytes(result)
    assert caught.value.code == "RESOURCE_LIMIT"


def test_direct_malformed_scoped_request_has_invalid_request_code(tmp_path):
    fixture = crowded(tmp_path)
    reader = coordinator(fixture)
    invalid = fixture.request().model_copy(update={"limit": True})
    with pytest.raises(ScopedRetrievalError) as caught:
        reader.search_scoped(invalid)
    assert caught.value.code == "INVALID_REQUEST"
