from __future__ import annotations

from pathlib import Path

import pytest

from academic_chatbot.documents.import_service import DocumentImportService
from academic_chatbot.documents.native_pdf import NativePdfParser
from academic_chatbot.evidence.resolver import EvidenceReadResolver
from academic_chatbot.evidence.service import EvidenceBundleService
from academic_chatbot.generation.models import (
    AnswerCitation,
    LocalGenerationAnswered,
)
from academic_chatbot.generation.orchestrator import SinglePaperAskService
from academic_chatbot.library.repository import (
    PaperSourceAmbiguousError,
    PaperSourceNotCurrentError,
    ProjectRepository,
    UnknownPaperError,
)
from academic_chatbot.library.service import LibraryService
from academic_chatbot.retrieval.hybrid_service import HybridRetrievalService
from academic_chatbot.retrieval.semantic import SemanticRetrievalService
from tests.fixtures.evidence_bundle.database import _SyntheticQueryEmbedder, database
from tests.fixtures.evidence_bundle.pdfs import write_pdf
from tests.integration.embeddings.test_vector_publication import _profile


class _Cancel:
    def is_set(self) -> bool:
        return False


class _CaptureGeneration:
    def __init__(self, during=None) -> None:
        self.calls = []
        self.during = during

    def generate(self, *, question, bundle, cancel):
        self.calls.append((question, bundle, cancel))
        if self.during is not None:
            self.during()
        entry = bundle.entries[0]
        source = entry.source
        return LocalGenerationAnswered(
            answer="The study objective is stated in the cited source.",
            citation_labels=(entry.citation_label,),
            citations=(
                AnswerCitation(
                    citation_label=entry.citation_label,
                    paper_id=source.scope.paper_id,
                    file_version_id=source.scope.file_version_id,
                    document_generation_id=source.scope.document_generation_id,
                    physical_page_index=source.physical_page_index,
                    display_page_number=source.display_page_number,
                    printed_page_label=source.printed_page_label,
                    page_id=source.parent.page_id,
                    start_offset=source.start_offset,
                    end_offset=source.end_offset,
                    text_sha256=source.text_sha256,
                    anchor_ids=source.anchor_ids,
                ),
            ),
            bundle_fingerprint=bundle.fingerprint,
            rendered_prompt_tokens=20,
            completion_tokens=8,
        )


def _actual_service(db, generation):
    prepared = db.semantic()
    profile = _profile()
    semantic = SemanticRetrievalService(
        data_root=db.paths.data_root,
        profile=profile,
        embedder=_SyntheticQueryEmbedder(profile),
    )
    return SinglePaperAskService(
        repository=ProjectRepository(db.paths),
        retrieval=HybridRetrievalService(
            data_root=db.paths.data_root,
            semantic_service=semantic,
        ),
        evidence=EvidenceBundleService(
            resolver=EvidenceReadResolver(data_root=db.paths.data_root)
        ),
        generation=generation,
        embedding_profile_id=prepared.origin.embedding_profile_id,
        vector_generation_id=prepared.origin.vector_generation_id,
    )


def test_repository_resolves_only_one_published_source(tmp_path: Path) -> None:
    db = database(tmp_path)

    source = ProjectRepository(db.paths).sole_current_paper_source(
        project_id="project-one", paper_id="paper-one"
    )

    assert source.file_version_id == db.request.scope.file_version_id
    assert source.document_generation_id == db.request.scope.document_generation_id


def test_repository_distinguishes_missing_stale_and_ambiguous(tmp_path: Path) -> None:
    missing = database(tmp_path / "missing")
    repository = ProjectRepository(missing.paths)
    with pytest.raises(UnknownPaperError):
        repository.sole_current_paper_source(
            project_id="project-one", paper_id="paper-missing"
        )

    missing.execute("DELETE FROM generation_publications")
    with pytest.raises(PaperSourceNotCurrentError):
        repository.sole_current_paper_source(
            project_id="project-one", paper_id="paper-one"
        )

    ambiguous = database(tmp_path / "ambiguous", older=True)
    with pytest.raises(PaperSourceAmbiguousError):
        ProjectRepository(ambiguous.paths).sole_current_paper_source(
            project_id="project-one", paper_id="paper-one"
        )


def test_actual_hybrid_retrieval_builds_one_paper_authoritative_bundle(
    tmp_path: Path,
) -> None:
    db = database(tmp_path, text="Alpha objective evaluates a synthetic method safely.")
    generation = _CaptureGeneration()
    service = _actual_service(db, generation)

    result = service.ask(
        project_id="project-one",
        paper_id="paper-one",
        question="What is the Alpha objective?",
        cancel=_Cancel(),
    )

    assert result.outcome == "answered"
    assert len(generation.calls) == 1
    bundle = generation.calls[0][1]
    assert bundle.state == "preview_ready"
    assert bundle.entries
    assert all(entry.source.scope.paper_id == "paper-one" for entry in bundle.entries)
    assert all(
        entry.source.scope.file_version_id == db.request.scope.file_version_id
        for entry in bundle.entries
    )
    assert bundle.origin.mode == "hybrid"


def test_second_publication_during_generation_discards_answer(tmp_path: Path) -> None:
    db = database(tmp_path, text="Alpha objective evaluates a synthetic method safely.")

    def publish_second() -> None:
        library = LibraryService(data_root=db.paths.data_root, max_pdf_bytes=1_000_000)
        version = library.admit_pdf(
            project_id="project-one",
            paper_id="paper-one",
            source_path=write_pdf(
                tmp_path / "second.pdf",
                text="Alpha objective changed in another synthetic source.",
            ),
        )
        DocumentImportService(library.repository_for_project_id("project-one")).publish(
            NativePdfParser(db.paths).parse(version)
        )

    service = _actual_service(db, _CaptureGeneration(during=publish_second))

    result = service.ask(
        project_id="project-one",
        paper_id="paper-one",
        question="What is the Alpha objective?",
        cancel=_Cancel(),
    )

    assert result.outcome == "rejected"
    assert result.code == "SOURCE_SCOPE_AMBIGUOUS"
    assert "The study objective" not in result.model_dump_json()
