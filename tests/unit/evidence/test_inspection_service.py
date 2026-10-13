"""Two authoritative builds and fresh release checks protect every inspection outcome."""

import hashlib
import importlib

import pytest

from academic_chatbot.evidence.models import PreviewBudget
from academic_chatbot.evidence.resolver import EvidenceReadResolver
from academic_chatbot.evidence.service import EvidenceBundleService
from academic_chatbot.library.repository import ProjectRepository
from academic_chatbot.retrieval.scoped_service import SourceScopedRetrievalService
from academic_chatbot.retrieval.service import RetrievalService
from tests.fixtures.retrieval.scoped import crowded, lexical_oracle, semantic_oracle
from tests.unit.evidence.test_inspection_models import models


def inspection_class():
    name = "academic_chatbot.evidence.inspection_service"
    assert importlib.util.find_spec(name), "inspection composition is absent"
    return importlib.import_module(name).EvidenceInspectionService


def components(tmp_path, *, mode="lexical", budget=None):
    cls = inspection_class()
    fixture = crowded(tmp_path, vectors=mode != "lexical")
    scoped = SourceScopedRetrievalService(
        data_root=fixture.db.paths.data_root,
        repository=ProjectRepository(fixture.db.paths),
        lexical_service=RetrievalService(data_root=fixture.db.paths.data_root),
        semantic_service=fixture.semantic,
    )
    bundle = EvidenceBundleService(
        resolver=EvidenceReadResolver(data_root=fixture.db.paths.data_root)
    )
    request = models().EvidenceInspectionRequest(
        retrieval_request=fixture.request(mode), preview_budget=budget or PreviewBudget()
    )
    return fixture, scoped, bundle, request, cls


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_adapter_preserves_ordered_storage_references_and_builds_twice(tmp_path, mode):
    fixture, scoped, bundles, request, cls = components(tmp_path, mode=mode)
    events, captured = [], []

    class Retrieval:
        def search_scoped(self, request):
            events.append("search")
            return scoped.search_scoped(request)

        def recheck_current(self, request):
            events.append("check")
            scoped.recheck_current(request)

    class Bundles:
        def build(self, request):
            events.append("build")
            captured.append(request)
            return bundles.build(request)

    result = cls(retrieval_service=Retrieval(), bundle_service=Bundles()).inspect(request)
    assert result.outcome == "ready"
    assert events.count("search") == 1 and events.count("build") == 2 and events[-1] == "check"
    assert captured[0] == captured[1]
    expected_chunks = (
        [r[0] for r in lexical_oracle(fixture)]
        if mode != "semantic"
        else [r[2] for r in semantic_oracle(fixture)]
    )
    assert [c.parent.chunk_id for c in captured[0].candidates] == expected_chunks
    assert [c.reported_rank for c in captured[0].candidates] == [1, 2]
    assert (
        captured[0].scope == fixture.scope
        and captured[0].concern_profile_id == "stated-study-objective-v1"
    )
    origin = captured[0].origin
    assert origin.mode == mode and origin.fusion_profile_id == (
        "rrf-v1" if mode == "hybrid" else None
    )
    if mode != "lexical":
        assert (
            origin.embedding_profile_id
            == request.retrieval_request.semantic_identity.embedding_profile_id
        )
        assert (
            origin.vector_generation_id
            == request.retrieval_request.semantic_identity.vector_generation_id
        )
    for candidate in captured[0].candidates:
        chunk = fixture.db.rows(
            "SELECT * FROM chunks WHERE chunk_id=?", (candidate.parent.chunk_id,)
        )[0]
        for channel in ("lexical", "semantic"):
            ref = getattr(candidate, channel)
            if mode not in (channel, "hybrid"):
                assert ref is None
                continue
            assert (
                ref.start_offset == chunk["start_offset"] and ref.end_offset == chunk["end_offset"]
            )
            assert (
                ref.expected_text_sha256 == hashlib.sha256(chunk["chunk_text"].encode()).hexdigest()
            )
            anchors = fixture.db.rows(
                "SELECT page_anchor_id FROM page_anchors WHERE page_id=? "
                "AND char_start>=? AND char_end<=? ORDER BY char_start,char_end,page_anchor_id",
                (chunk["page_id"], ref.start_offset, ref.end_offset),
            )
            assert ref.expected_anchor_ids == tuple(a[0] for a in anchors)
            if channel == "semantic":
                assert (
                    ref.embedding_span_id
                    == fixture.db.rows(
                        "SELECT embedding_span_id FROM embedding_spans WHERE chunk_id=?",
                        (chunk["chunk_id"],),
                    )[0][0]
                )


@pytest.mark.parametrize(
    "entries,size,query,outcome",
    [
        (8, 16384, "Alpha", "ready"),
        (1, 16384, "Alpha", "partial"),
        (8, 77, "Alpha", "insufficient"),
        (8, 16384, "absentterm", "insufficient"),
    ],
)
def test_preview_outcomes_come_only_from_existing_bundle_authority(
    tmp_path, entries, size, query, outcome
):
    _fixture, scoped, bundle, request, cls = components(
        tmp_path, budget=PreviewBudget(max_entries=entries, max_content_bytes=size)
    )
    request = request.model_copy(
        update={"retrieval_request": request.retrieval_request.model_copy(update={"query": query})}
    )
    result = cls(retrieval_service=scoped, bundle_service=bundle).inspect(request)
    assert result.outcome == outcome
    assert result.bundle_fingerprint == result.bundle.fingerprint
    assert result.bundle.coverage.document_completeness == "not_assessed"
    assert result.bundle.support_assessment == "not_performed"
    assert (
        result.bundle.budget.used_entries <= entries
        and result.bundle.budget.used_content_bytes <= size
    )
    if outcome == "partial":
        assert result.bundle.entries and any(d.omission for d in result.bundle.dispositions)
    if outcome == "insufficient":
        assert not result.bundle.entries


@pytest.mark.parametrize("point", ["after_retrieval", "after_second_build"])
@pytest.mark.parametrize(
    "entries,size,query",
    [(8, 16384, "Alpha"), (1, 16384, "Alpha"), (8, 77, "Alpha"), (8, 16384, "absentterm")],
)
def test_source_publication_race_suppresses_evidence(tmp_path, point, entries, size, query):
    fixture, scoped, bundle, request, cls = components(
        tmp_path, budget=PreviewBudget(max_entries=entries, max_content_bytes=size)
    )
    request = request.model_copy(
        update={"retrieval_request": request.retrieval_request.model_copy(update={"query": query})}
    )

    class Retrieval:
        def search_scoped(self, request):
            acquired = scoped.search_scoped(request)
            if point == "after_retrieval":
                fixture.db.execute(
                    "DELETE FROM generation_publications WHERE file_version_id=?",
                    (fixture.scope.file_version_id,),
                )
            return acquired

        def recheck_current(self, request):
            scoped.recheck_current(request)

    class Bundles:
        calls = 0

        def build(self, request):
            result = bundle.build(request)
            self.calls += 1
            if self.calls == 2 and point == "after_second_build":
                fixture.db.execute(
                    "DELETE FROM generation_publications WHERE file_version_id=?",
                    (fixture.scope.file_version_id,),
                )
            return result

    result = cls(retrieval_service=Retrieval(), bundle_service=Bundles()).inspect(request)
    assert result.outcome == "rejected" and result.error.code == "SOURCE_NOT_CURRENT"
    assert "bundle" not in result.model_dump() and result.request_fingerprint == request.fingerprint


def test_source_change_after_serialization_is_suppressed_before_return(tmp_path, monkeypatch):
    fixture, scoped, bundle, request, cls = components(tmp_path)
    from academic_chatbot.evidence import inspection_service

    original = inspection_service.canonical_inspection_result_bytes

    def change_after_bytes(result):
        raw = original(result)
        fixture.db.execute(
            "DELETE FROM generation_publications WHERE file_version_id=?",
            (fixture.scope.file_version_id,),
        )
        return raw

    monkeypatch.setattr(inspection_service, "canonical_inspection_result_bytes", change_after_bytes)
    result = cls(retrieval_service=scoped, bundle_service=bundle).inspect(request)
    assert result.outcome == "rejected" and result.error.code == "SOURCE_NOT_CURRENT"
    assert "bundle" not in result.model_dump()


def test_changed_coverage_between_builds_is_not_released(tmp_path):
    fixture, scoped, bundle, request, cls = components(tmp_path)

    class ChangingBundles:
        calls = 0

        def build(self, request):
            result = bundle.build(request)
            self.calls += 1
            if self.calls == 1:
                fixture.db.corrupt(
                    "UPDATE pages SET extraction_quality='low_native_text', needs_ocr=1 "
                    "WHERE document_generation_id=?",
                    (fixture.scope.document_generation_id,),
                )
            return result

    result = cls(retrieval_service=scoped, bundle_service=ChangingBundles()).inspect(request)
    assert result.outcome == "failed" and result.error.code == "INTERNAL_INVARIANT_FAILURE"
    assert "bundle" not in result.model_dump()


def test_invalid_request_and_unexpected_collaborator_failures_are_sanitized(tmp_path):
    _fixture, _scoped, bundle, request, cls = components(tmp_path)

    class Broken:
        def search_scoped(self, request):
            raise RuntimeError("SECRET C:/private/file.sqlite")

        def recheck_current(self, request):
            raise AssertionError("unexpected release")

    service = cls(retrieval_service=Broken(), bundle_service=bundle)
    result = service.inspect(request)
    assert result.outcome == "failed" and result.error.code == "INTERNAL_INVARIANT_FAILURE"
    assert result.request_fingerprint is None and "SECRET" not in result.model_dump_json()
    forged = request.model_copy(
        update={"retrieval_request": request.retrieval_request.model_copy(update={"limit": True})}
    )
    invalid = service.inspect(forged)
    assert invalid.outcome == "rejected" and invalid.error.code == "INVALID_REQUEST"
    assert invalid.request_fingerprint is None


def test_request_fingerprint_memory_failure_returns_safe_resource_result(tmp_path, monkeypatch):
    _fixture, scoped, bundle, request, cls = components(tmp_path)

    def exhausted(*args, **kwargs):
        raise MemoryError("PRIVATE-CANARY C:/private/request")

    monkeypatch.setattr(
        "academic_chatbot.evidence.inspection_serialization._bounded_bytes", exhausted
    )
    result = cls(retrieval_service=scoped, bundle_service=bundle).inspect(request)
    assert result.outcome == "failed" and result.error.code == "RESOURCE_LIMIT"
    assert result.request_fingerprint is None and "bundle" not in result.model_dump()
    assert "PRIVATE" not in result.model_dump_json()


@pytest.mark.parametrize("mutation", ["foreign", "unsupported", "query"])
def test_untrusted_retrieval_packets_cannot_bypass_candidate_authority(tmp_path, mutation):
    _fixture, scoped, bundle, request, cls = components(tmp_path)
    acquired = scoped.search_scoped(request.retrieval_request)
    if mutation == "foreign":
        acquired = acquired.model_copy(
            update={"hits": (acquired.hits[0].model_copy(update={"paper_id": "foreign"}),)}
        )
    elif mutation == "unsupported":
        acquired = acquired.model_copy(update={"hits": (object(),)})
    else:
        acquired = acquired.model_copy(update={"query": "changed"})

    class Retrieval:
        def search_scoped(self, request):
            return acquired

        def recheck_current(self, request):
            scoped.recheck_current(request)

    result = cls(retrieval_service=Retrieval(), bundle_service=bundle).inspect(request)
    assert result.outcome == "failed" and result.error.code == "INTERNAL_INVARIANT_FAILURE"
    assert "bundle" not in result.model_dump()


def test_hybrid_channel_binding_allows_acquisition_above_final_cap(tmp_path):
    from pydantic import TypeAdapter, ValidationError

    from academic_chatbot.evidence.inspection_models import EvidenceInspectionResult

    _fixture, scoped, bundle, request, cls = components(tmp_path, mode="hybrid")
    request = request.model_copy(
        update={"retrieval_request": request.retrieval_request.model_copy(update={"limit": 1})}
    )
    result = cls(retrieval_service=scoped, bundle_service=bundle).inspect(request)
    assert result.outcome == "ready" and len(result.bundle.dispositions) == 1
    identity = result.retrieval_identity
    assert (
        identity.channels.lexical.returned_count == identity.channels.semantic.returned_count == 2
    )
    assert TypeAdapter(EvidenceInspectionResult).validate_python(result) == result
    for name in ("lexical", "semantic"):
        forged_channels = identity.channels.model_copy(
            update={
                name: getattr(identity.channels, name).model_copy(
                    update={"state": "empty", "returned_count": 0}
                )
            }
        )
        forged = result.model_copy(
            update={"retrieval_identity": identity.model_copy(update={"channels": forged_channels})}
        )
        with pytest.raises(ValidationError):
            TypeAdapter(EvidenceInspectionResult).validate_python(forged)
