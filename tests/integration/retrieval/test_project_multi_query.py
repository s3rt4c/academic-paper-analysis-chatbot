from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from academic_chatbot.documents import import_service
from academic_chatbot.documents.import_service import DocumentImportService
from academic_chatbot.documents.native_pdf import NativePdfParser
from academic_chatbot.domain.library import FileVersion, Project
from academic_chatbot.library.service import LibraryService
from academic_chatbot.retrieval.hybrid_service import HybridRetrievalService
from academic_chatbot.retrieval.multi_query import LexicalMultiQueryService
from academic_chatbot.retrieval.semantic import SemanticRetrievalResults
from academic_chatbot.retrieval.service import RetrievalService
from academic_chatbot.storage.paths import ProjectPaths


def _publish_text_pdf(
    *, data_root: Path, project_id: str, text: str
) -> tuple[LibraryService, Project, str, FileVersion]:
    source = data_root.parent / f"{project_id}.pdf"
    source.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(source), invariant=1)
    pdf.drawString(36, 780, text)
    pdf.save()

    library = LibraryService(data_root=data_root, max_pdf_bytes=1_000_000)
    project = library.create_project(display_name="Synthetic", project_id=project_id)
    paper = library.create_paper(project_id=project_id, paper_id=f"paper-{project_id}")
    file_version = library.admit_pdf(
        project_id=project_id, paper_id=paper.paper_id, source_path=source
    )
    parsed = NativePdfParser(
        ProjectPaths.create(data_root, project_id=project_id)
    ).parse(file_version)
    published = DocumentImportService(library.repository_for(project)).publish(parsed)
    return library, project, published.document_generation_id, file_version


def _republish_same_file_version(
    *,
    library: LibraryService,
    project: Project,
    data_root: Path,
    file_version: FileVersion,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    monkeypatch.setattr(import_service, "DOCUMENT_GENERATION_PROFILE_ID", "native-lexical-fts-v2")
    parsed = NativePdfParser(
        ProjectPaths.create(data_root, project_id=project.project_id)
    ).parse(file_version)
    published = DocumentImportService(library.repository_for(project)).publish(parsed)
    return published.document_generation_id


@dataclass
class _EmptySemanticService:
    calls: list[tuple[Project, str, int]] = field(default_factory=list)

    def search(self, project: Project, query: str, limit: int) -> SemanticRetrievalResults:
        self.calls.append((project, query, limit))
        return SemanticRetrievalResults(
            project_id=project.project_id,
            query=query,
            embedding_profile_id="synthetic-profile",
            vector_generation_id="synthetic-generation",
            hits=(),
        )


def test_morphology_branch_recovers_a_plural_only_chunk(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _, project, _, _ = _publish_text_pdf(
        data_root=data_root,
        project_id="project-morphology",
        text="The investigator studies the sample carefully.",
    )

    assert not RetrievalService(data_root=data_root).search(project, "study").hits

    results = LexicalMultiQueryService(data_root=data_root).search(project, "study")

    assert results.hits
    assert "studies" in results.hits[0].chunk_text
    assert len({hit.chunk_id for hit in results.hits}) == len(results.hits)


def test_intent_branch_recovers_a_closed_family_variant(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _, project, _, _ = _publish_text_pdf(
        data_root=data_root,
        project_id="project-intent",
        text="The research aim is clearly stated.",
    )

    assert not RetrievalService(data_root=data_root).search(project, "objective").hits

    results = LexicalMultiQueryService(data_root=data_root).search(project, "objective")

    assert results.hits
    assert "aim" in results.hits[0].chunk_text


def test_duplicate_branch_hits_materialize_one_parent_result(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _, project, _, _ = _publish_text_pdf(
        data_root=data_root,
        project_id="project-duplicate",
        text="The study studies repeated samples.",
    )

    results = LexicalMultiQueryService(data_root=data_root).search(project, "study studies")

    identities = {
        (hit.project_id, hit.document_generation_id, hit.page_id, hit.chunk_id)
        for hit in results.hits
    }
    assert results.hits
    assert len(identities) == len(results.hits)


def test_multi_query_keeps_project_scope_isolated(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _, first, _, _ = _publish_text_pdf(
        data_root=data_root,
        project_id="project-first",
        text="The target belongs to the first project.",
    )
    _, second, _, _ = _publish_text_pdf(
        data_root=data_root,
        project_id="project-second",
        text="The target belongs to the second project.",
    )

    first_hits = LexicalMultiQueryService(data_root=data_root).search(first, "target").hits
    second_hits = LexicalMultiQueryService(data_root=data_root).search(second, "target").hits

    assert first_hits and second_hits
    assert {hit.project_id for hit in first_hits} == {"project-first"}
    assert {hit.project_id for hit in second_hits} == {"project-second"}


def test_multi_query_keeps_only_the_current_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    library, project, first_generation, file_version = _publish_text_pdf(
        data_root=data_root,
        project_id="project-generation",
        text="The study belongs to the first version.",
    )
    second_generation = _republish_same_file_version(
        library=library,
        project=project,
        data_root=data_root,
        file_version=file_version,
        monkeypatch=monkeypatch,
    )

    results = LexicalMultiQueryService(data_root=data_root).search(project, "study")

    assert first_generation != second_generation
    assert {hit.document_generation_id for hit in results.hits} == {second_generation}


def test_hybrid_receives_one_merged_lexical_channel_and_preserves_semantic_query(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _, project, _, _ = _publish_text_pdf(
        data_root=data_root,
        project_id="project-hybrid",
        text="The investigator studies the sample.",
    )
    semantic = _EmptySemanticService()
    service = HybridRetrievalService(
        data_root=data_root,
        lexical_service=LexicalMultiQueryService(data_root=data_root),
        semantic_service=semantic,
    )

    first = service.search(project, "study", limit=10)
    repeated = service.search(project, "study", limit=10)

    assert first == repeated
    assert first.fusion_profile_id == "rrf-v1"
    assert first.hits
    assert first.hits[0].lexical_contribution is not None
    assert first.hits[0].semantic_contribution is None
    assert semantic.calls == [(project, "study", 50), (project, "study", 50)]
