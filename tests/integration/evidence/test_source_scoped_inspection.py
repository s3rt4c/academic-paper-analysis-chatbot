"""Real source/retrieval/evidence authority with synthetic PDFs and offline fake embeddings."""

import json

import pytest

from academic_chatbot.documents.import_service import (
    DocumentImportService,
    document_generation_id_for,
)
from academic_chatbot.documents.native_pdf import NativePdfParser
from academic_chatbot.embeddings.repository import EmbeddingRepository
from academic_chatbot.evidence.inspection_models import EvidenceInspectionRequest
from academic_chatbot.evidence.inspection_serialization import canonical_inspection_result_bytes
from academic_chatbot.evidence.inspection_service import EvidenceInspectionService
from academic_chatbot.evidence.models import PreviewBudget
from academic_chatbot.evidence.resolver import EvidenceReadResolver
from academic_chatbot.evidence.service import EvidenceBundleService
from academic_chatbot.library.repository import ProjectRepository
from academic_chatbot.library.service import LibraryService
from academic_chatbot.retrieval.scoped_service import SourceScopedRetrievalService
from academic_chatbot.retrieval.semantic import SemanticRetrievalService
from academic_chatbot.retrieval.service import RetrievalService
from tests.fixtures.evidence_bundle.database import _SyntheticQueryEmbedder, database
from tests.fixtures.evidence_bundle.pdfs import write_pdf
from tests.fixtures.retrieval.scoped import SparseEmbedder
from tests.integration.embeddings.test_vector_publication import _builder, _profile
from tests.unit.evidence.test_inspection_service import components


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_inspection_real_bundle_authority_and_repeated_canonical_bytes(tmp_path, mode):
    fixture, scoped, bundles, request, cls = components(tmp_path, mode=mode)
    service = cls(retrieval_service=scoped, bundle_service=bundles)
    result = service.inspect(request)
    assert result.outcome == "ready"
    assert (
        result.source.source_pdf_sha256
        == fixture.db.rows(
            "SELECT sha256 FROM file_versions WHERE file_version_id=?",
            (fixture.scope.file_version_id,),
        )[0][0]
    )
    assert [e.citation_label for e in result.bundle.entries] == ["E1", "E2"]
    for entry in result.bundle.entries:
        assert entry.source.scope == result.source and entry.trust == "untrusted_source_data"
        assert entry.source.parent.project_id == fixture.scope.project_id
        assert entry.source.parent.document_generation_id == fixture.scope.document_generation_id
        assert entry.source.display_page_number == entry.source.physical_page_index + 1
        assert entry.anchors and "foreignmarker" not in entry.text
        assert {p.channel for p in entry.contributions} == (
            {"lexical", "semantic"} if mode == "hybrid" else {mode}
        )
    assert all(not c.citable and c.trust == "untrusted_source_data" for c in result.bundle.context)
    assert result.bundle.budget.max_entries == 8 and result.bundle.budget.max_content_bytes == 16384
    raw = canonical_inspection_result_bytes(result)
    assert raw == canonical_inspection_result_bytes(service.inspect(request))
    payload = json.loads(raw)
    assert "hits" not in payload and "coverage" not in payload and "citations" not in payload
    assert payload["bundle"]["coverage"]["document_completeness"] == "not_assessed"
    assert payload["bundle"]["selected_source_retrieval_completeness"] == "not_assessed"
    assert payload["bundle"]["support_assessment"] == "not_performed"


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_healthy_empty_source_remains_insufficient_with_honest_native_coverage(tmp_path, mode):
    db = database(tmp_path, empty=True)
    semantic = None
    if mode != "lexical":
        db.semantic()
        profile = _profile()
        semantic = SemanticRetrievalService(
            data_root=db.paths.data_root, profile=profile, embedder=_SyntheticQueryEmbedder(profile)
        )
    scoped = SourceScopedRetrievalService(
        data_root=db.paths.data_root,
        repository=ProjectRepository(db.paths),
        lexical_service=RetrievalService(data_root=db.paths.data_root),
        semantic_service=semantic,
    )
    retrieval = scoped.prepare_request(
        project_id="project-one",
        paper_id="paper-one",
        query="Alpha",
        mode=mode,
        embedding_profile_id=semantic.profile.embedding_profile_id if semantic else None,
    )
    result = EvidenceInspectionService(
        retrieval_service=scoped,
        bundle_service=EvidenceBundleService(
            resolver=EvidenceReadResolver(data_root=db.paths.data_root)
        ),
    ).inspect(EvidenceInspectionRequest(retrieval_request=retrieval))
    assert result.outcome == "insufficient" and not result.bundle.entries
    assert result.bundle.insufficient_reason == "no_candidates"
    assert (
        result.bundle.coverage.flagged_page_ids
        and result.bundle.coverage.document_completeness == "not_assessed"
    )


def test_known_native_gap_and_budget_omissions_remain_visible_partial_evidence(tmp_path):
    fixture, scoped, bundles, request, cls = components(
        tmp_path, budget=PreviewBudget(max_entries=1)
    )
    fixture.db.corrupt(
        "UPDATE pages SET needs_ocr=1 WHERE document_generation_id=?",
        (fixture.scope.document_generation_id,),
    )
    result = cls(retrieval_service=scoped, bundle_service=bundles).inspect(request)
    assert result.outcome == "partial" and len(result.bundle.entries) == 1
    assert (
        result.bundle.coverage.status == "known_native_gaps"
        and result.bundle.coverage.flagged_page_ids
    )
    assert any(
        d.omission and d.omission.reason == "entry_limit_exceeded"
        for d in result.bundle.dispositions
    )


@pytest.mark.parametrize("replacement", ["generation", "file_version"])
def test_replaced_authoritative_source_after_second_build_never_releases_old_bundle(
    tmp_path, replacement
):
    fixture, scoped, bundles, request, cls = components(tmp_path)
    if replacement == "generation":
        new_id = document_generation_id_for(
            file_version_id=fixture.scope.file_version_id,
            processing_profile_id="synthetic-inspection-pipeline",
        )
        fixture.db.execute(
            "INSERT INTO document_generations VALUES (?,?,?,?)",
            (
                new_id,
                fixture.scope.file_version_id,
                "synthetic-inspection-pipeline",
                "synthetic-time",
            ),
        )
    else:
        library = LibraryService(data_root=fixture.db.paths.data_root, max_pdf_bytes=1_000_000)
        new_file = library.admit_pdf(
            project_id="project-one",
            paper_id="selected",
            source_path=write_pdf(tmp_path / "replacement.pdf", suffix="replacement"),
        )
        parsed = NativePdfParser(fixture.db.paths).parse(new_file)

    class WriterAfterBuild:
        count = 0

        def build(self, request):
            result = bundles.build(request)
            self.count += 1
            if self.count == 2:
                if replacement == "generation":
                    fixture.db.execute(
                        "UPDATE generation_publications SET document_generation_id=? "
                        "WHERE file_version_id=?",
                        (new_id, fixture.scope.file_version_id),
                    )
                else:
                    DocumentImportService(library.repository_for_project_id("project-one")).publish(
                        parsed
                    )
                    fixture.db.execute(
                        "DELETE FROM generation_publications WHERE file_version_id=?",
                        (fixture.scope.file_version_id,),
                    )
            return result

    result = cls(retrieval_service=scoped, bundle_service=WriterAfterBuild()).inspect(request)
    assert result.outcome == "rejected" and result.error.code == "SOURCE_NOT_CURRENT"
    assert "bundle" not in result.model_dump() and result.request_fingerprint == request.fingerprint


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_historical_vector_publication_switch_after_build_is_failed_without_evidence(
    tmp_path, mode
):
    fixture, scoped, bundles, request, cls = components(tmp_path, mode=mode)
    old_vector = request.retrieval_request.semantic_identity.vector_generation_id
    distractor = tuple(
        fixture.db.rows(
            "SELECT file_version_id,document_generation_id FROM generation_publications "
            "WHERE file_version_id!=?",
            (fixture.scope.file_version_id,),
        )[0]
    )
    fixture.db.execute(
        "DELETE FROM generation_publications WHERE file_version_id=?", (distractor[0],)
    )
    historical = (
        _builder(
            EmbeddingRepository(fixture.db.paths),
            fixture.db.paths,
            SparseEmbedder(fixture.semantic.profile),
        )
        .build(project_id="project-one")
        .generation.vector_generation_id
    )
    fixture.db.execute("INSERT INTO generation_publications VALUES (?,?)", distractor)
    fixture.db.execute(
        "UPDATE vector_generation_publications SET vector_generation_id=?", (old_vector,)
    )

    class WriterAfterBuild:
        count = 0

        def build(self, request):
            result = bundles.build(request)
            self.count += 1
            if self.count == 2:
                fixture.db.execute(
                    "UPDATE vector_generation_publications SET vector_generation_id=?",
                    (historical,),
                )
            return result

    result = cls(retrieval_service=scoped, bundle_service=WriterAfterBuild()).inspect(request)
    assert result.outcome == "failed" and result.error.code == "VECTOR_NOT_CURRENT"
    assert "bundle" not in result.model_dump() and result.request_fingerprint == request.fingerprint


def test_corrupt_text_between_builds_preserves_existing_safe_error(tmp_path):
    fixture, scoped, bundles, request, cls = components(tmp_path)

    class CorruptionAfterBuild:
        count = 0

        def build(self, request):
            result = bundles.build(request)
            self.count += 1
            if self.count == 1:
                fixture.db.corrupt(
                    "UPDATE chunks SET chunk_text='Alpha damaged' WHERE document_generation_id=?",
                    (fixture.scope.document_generation_id,),
                )
            return result

    result = cls(retrieval_service=scoped, bundle_service=CorruptionAfterBuild()).inspect(request)
    assert result.outcome == "failed" and result.error.code in {
        "STORAGE_INTEGRITY",
        "TEXT_DIGEST_MISMATCH",
    }
    assert "bundle" not in result.model_dump()
