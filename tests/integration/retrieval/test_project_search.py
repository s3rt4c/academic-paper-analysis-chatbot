from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from academic_chatbot.documents import import_service
from academic_chatbot.documents.import_service import DocumentImportService
from academic_chatbot.documents.native_pdf import NativePdfParser
from academic_chatbot.domain.library import Project
from academic_chatbot.library.service import LibraryService
from academic_chatbot.retrieval.fts import RetrievalQueryError
from academic_chatbot.retrieval.service import (
    RetrievalIntegrityError,
    RetrievalService,
    RetrievalStorageError,
)
from academic_chatbot.storage.paths import ProjectPaths

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "pdfs" / "native_anchor.pdf"


def _publish_fixture(
    *, data_root: Path, project_id: str = "project-1", paper_id: str = "paper-1"
):
    library = LibraryService(data_root=data_root, max_pdf_bytes=1_000_000)
    project = library.create_project(display_name="Research", project_id=project_id)
    paper = library.create_paper(project_id=project.project_id, paper_id=paper_id)
    file_version = library.admit_pdf(
        project_id=project.project_id, paper_id=paper.paper_id, source_path=_FIXTURE
    )
    paths = ProjectPaths.create(data_root, project_id=project.project_id)
    parsed = NativePdfParser(paths).parse(file_version)
    published = DocumentImportService(library.repository_for(project)).publish(parsed)
    return library, project, paper, file_version, published


def _persisted_rows(
    library: LibraryService, project_id: str
) -> dict[str, tuple[tuple[object, ...], ...]]:
    tables = (
        "projects",
        "papers",
        "file_versions",
        "document_generations",
        "generation_publications",
        "pages",
        "page_anchors",
        "chunks",
        "chunk_fts",
    )
    with library.repository_for_project_id(project_id)._connection() as connection:
        return {
            table: tuple(tuple(row) for row in connection.execute(f"SELECT rowid, * FROM {table}"))
            for table in tables
        }


def test_project_search_returns_exact_active_evidence_without_reparsing(tmp_path: Path) -> None:
    """Would fail if Task 5 returned copied text without active persisted evidence lineage."""
    data_root = tmp_path / "data"
    _, project, _, _, published = _publish_fixture(data_root=data_root)

    results = RetrievalService(data_root=data_root).search(project, "control", limit=10)

    assert results.project_id == project.project_id
    assert results.hits
    hit = results.hits[0]
    assert hit.document_generation_id == published.document_generation_id
    assert hit.chunk_text == hit.anchors[0].canonical_page_text[
        hit.start_offset : hit.end_offset
    ]
    assert all(
        anchor.char_start < hit.end_offset and anchor.char_end > hit.start_offset
        for anchor in hit.anchors
    )


def test_search_exposes_only_the_currently_active_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if a superseded generation remained retrievable through FTS."""
    data_root = tmp_path / "data"
    library, project, _, file_version, first = _publish_fixture(data_root=data_root)
    paths = ProjectPaths.create(data_root, project_id=project.project_id)
    parsed = NativePdfParser(paths).parse(file_version)
    monkeypatch.setattr(import_service, "DOCUMENT_GENERATION_PROFILE_ID", "native-lexical-fts-v2")
    replacement = DocumentImportService(library.repository_for(project)).publish(parsed)

    results = RetrievalService(data_root=data_root).search(project, "control")

    assert results.hits
    assert {hit.document_generation_id for hit in results.hits} == {
        replacement.document_generation_id
    }
    assert first.document_generation_id != replacement.document_generation_id


def test_search_keeps_duplicate_pdf_results_owned_by_their_papers(tmp_path: Path) -> None:
    """Would fail if shared bytes collapsed distinct paper-owned FileVersions."""
    data_root = tmp_path / "data"
    library = LibraryService(data_root=data_root, max_pdf_bytes=1_000_000)
    project = library.create_project(display_name="Research", project_id="project-1")
    for paper_id in ("paper-a", "paper-b"):
        paper = library.create_paper(project_id=project.project_id, paper_id=paper_id)
        file_version = library.admit_pdf(
            project_id=project.project_id, paper_id=paper.paper_id, source_path=_FIXTURE
        )
        parsed = NativePdfParser(
            ProjectPaths.create(data_root, project_id=project.project_id)
        ).parse(file_version)
        DocumentImportService(library.repository_for(project)).publish(parsed)

    hits = RetrievalService(data_root=data_root).search(project, "control").hits

    assert {hit.paper_id for hit in hits} == {"paper-a", "paper-b"}
    assert len({hit.file_version_id for hit in hits}) == 2


def test_search_orders_equal_content_deterministically_before_applying_limit(
    tmp_path: Path,
) -> None:
    """Would fail if SQL applied LIMIT before its complete deterministic ranking order."""
    data_root = tmp_path / "data"
    library = LibraryService(data_root=data_root, max_pdf_bytes=1_000_000)
    project = library.create_project(display_name="Research", project_id="project-1")
    for paper_id in ("paper-a", "paper-b"):
        paper = library.create_paper(project_id=project.project_id, paper_id=paper_id)
        file_version = library.admit_pdf(
            project_id=project.project_id, paper_id=paper.paper_id, source_path=_FIXTURE
        )
        parsed = NativePdfParser(
            ProjectPaths.create(data_root, project_id=project.project_id)
        ).parse(file_version)
        DocumentImportService(library.repository_for(project)).publish(parsed)

    first = RetrievalService(data_root=data_root).search(project, "control", limit=10).hits
    repeated = RetrievalService(data_root=data_root).search(project, "control", limit=10).hits
    limited = RetrievalService(data_root=data_root).search(project, "control", limit=1).hits

    order = [
        (
            hit.raw_bm25_score,
            hit.paper_id,
            hit.file_version_id,
            hit.physical_page_index,
            hit.chunk_ordinal,
            hit.chunk_id,
        )
        for hit in first
    ]
    assert first == repeated
    assert order == sorted(order)
    assert limited == first[:1]


def test_search_does_not_reparse_or_mutate_the_project_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if search reopened PDFs or performed writes while retrieving evidence."""
    data_root = tmp_path / "data"
    library, project, _, _, _ = _publish_fixture(data_root=data_root)
    before = _persisted_rows(library, project.project_id)

    def fail_reparse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("search must not parse PDF files")

    monkeypatch.setattr(NativePdfParser, "parse", fail_reparse)
    assert RetrievalService(data_root=data_root).search(project, "control").hits

    assert _persisted_rows(library, project.project_id) == before


def test_search_rejects_tampered_chunk_evidence(tmp_path: Path) -> None:
    """Would fail if evidence text were returned despite no exact canonical range match."""
    data_root = tmp_path / "data"
    library, project, _, _, _ = _publish_fixture(data_root=data_root)
    repository = library.repository_for(project)
    with repository._connection() as connection:
        chunk_id = connection.execute(
            "SELECT chunk_id FROM chunks WHERE ordinal = 0"
        ).fetchone()[0]
        connection.execute(
            "UPDATE chunks SET chunk_text = 'control altered' WHERE chunk_id = ?", (chunk_id,)
        )
        connection.execute(
            "UPDATE chunk_fts SET chunk_text = 'control altered' WHERE chunk_id = ?", (chunk_id,)
        )

    with pytest.raises(RetrievalIntegrityError, match="canonical page range"):
        RetrievalService(data_root=data_root).search(project, "control")


def test_search_rejects_duplicate_fts_rows_for_one_chunk(tmp_path: Path) -> None:
    """Would fail if duplicate index rows caused repeated unverified evidence hits."""
    data_root = tmp_path / "data"
    library, project, _, _, _ = _publish_fixture(data_root=data_root)
    repository = library.repository_for(project)
    with repository._connection() as connection:
        chunk_id, chunk_text = connection.execute(
            "SELECT chunk_id, chunk_text FROM chunks WHERE ordinal = 0"
        ).fetchone()
        connection.execute(
            "INSERT INTO chunk_fts (chunk_id, chunk_text) VALUES (?, ?)", (chunk_id, chunk_text)
        )

    with pytest.raises(RetrievalIntegrityError, match="FTS"):
        RetrievalService(data_root=data_root).search(project, "control")


def test_search_rejects_fts_text_that_differs_from_its_chunk(tmp_path: Path) -> None:
    """Would fail if indexed text could return a different persisted chunk as evidence."""
    data_root = tmp_path / "data"
    library, project, _, _, _ = _publish_fixture(data_root=data_root)
    repository = library.repository_for(project)
    with repository._connection() as connection:
        chunk_id = connection.execute(
            "SELECT chunk_id FROM chunks WHERE ordinal = 0"
        ).fetchone()[0]
        connection.execute(
            "UPDATE chunk_fts SET chunk_text = 'control altered' WHERE chunk_id = ?", (chunk_id,)
        )

    with pytest.raises(RetrievalIntegrityError, match="FTS text"):
        RetrievalService(data_root=data_root).search(project, "control")


def test_search_rejects_an_anchor_that_extends_past_the_chunk_range(tmp_path: Path) -> None:
    """Would fail if an overlap-selected anchor were returned outside the exact chunk range."""
    data_root = tmp_path / "data"
    library, project, _, _, _ = _publish_fixture(data_root=data_root)
    repository = library.repository_for(project)
    with repository._connection() as connection:
        chunk = connection.execute(
            """
            SELECT chunks.chunk_id, chunks.page_id, chunks.start_offset, chunks.end_offset,
                   pages.canonical_text
            FROM chunks JOIN pages ON pages.page_id = chunks.page_id
            WHERE chunks.chunk_text LIKE '%control%' AND chunks.ordinal = 0
            """
        ).fetchone()
        assert chunk is not None
        trailing_anchor = connection.execute(
            """
            SELECT char_start, char_end FROM page_anchors
            WHERE page_id = ? AND char_end = ?
            """,
            (chunk["page_id"], chunk["end_offset"]),
        ).fetchone()
        assert trailing_anchor is not None
        new_end = trailing_anchor["char_end"] - 1
        new_text = chunk["canonical_text"][chunk["start_offset"] : new_end]
        assert "control" in new_text
        connection.execute(
            "UPDATE chunks SET end_offset = ?, chunk_text = ? WHERE chunk_id = ?",
            (new_end, new_text, chunk["chunk_id"]),
        )
        connection.execute(
            "UPDATE chunk_fts SET chunk_text = ? WHERE chunk_id = ?",
            (new_text, chunk["chunk_id"]),
        )

    with pytest.raises(RetrievalIntegrityError, match="outside its chunk range"):
        RetrievalService(data_root=data_root).search(project, "control")


def test_search_maps_literal_fts_parser_rejection_to_query_error(tmp_path: Path) -> None:
    """Would fail if a generated literal MATCH failure looked like storage corruption."""
    data_root = tmp_path / "data"
    _, project, _, _, _ = _publish_fixture(data_root=data_root)

    with pytest.raises(RetrievalQueryError, match="plain lexical query"):
        RetrievalService(data_root=data_root).search(project, "control\x00")


def test_needs_ocr_pages_remain_unsearchable(tmp_path: Path) -> None:
    """Would fail if search fabricated lexical evidence for insufficient native text."""
    source = tmp_path / "empty-native.pdf"
    pdf = canvas.Canvas(str(source), invariant=1)
    pdf.showPage()
    pdf.save()
    data_root = tmp_path / "data"
    library = LibraryService(data_root=data_root, max_pdf_bytes=1_000_000)
    project = library.create_project(display_name="Research", project_id="project-1")
    paper = library.create_paper(project_id=project.project_id, paper_id="paper-1")
    file_version = library.admit_pdf(
        project_id=project.project_id, paper_id=paper.paper_id, source_path=source
    )
    parsed = NativePdfParser(ProjectPaths.create(data_root, project_id=project.project_id)).parse(
        file_version
    )
    DocumentImportService(library.repository_for(project)).publish(parsed)

    assert not RetrievalService(data_root=data_root).search(project, "control").hits


def test_repeated_words_keep_their_persisted_occurrence_specific_offsets(tmp_path: Path) -> None:
    """Would fail if retrieval rediscovered repeated evidence with text search."""
    source = tmp_path / "repeated.pdf"
    pdf = canvas.Canvas(str(source), invariant=1)
    text = pdf.beginText(36, 780)
    text.setFont("Helvetica", 6)
    text.setLeading(8)
    for _ in range(12):
        text.textLine(" ".join(["echo"] * 25))
    pdf.drawText(text)
    pdf.save()
    data_root = tmp_path / "data"
    library = LibraryService(data_root=data_root, max_pdf_bytes=1_000_000)
    project = library.create_project(display_name="Research", project_id="project-1")
    paper = library.create_paper(project_id=project.project_id, paper_id="paper-1")
    file_version = library.admit_pdf(
        project_id=project.project_id, paper_id=paper.paper_id, source_path=source
    )
    parsed = NativePdfParser(ProjectPaths.create(data_root, project_id=project.project_id)).parse(
        file_version
    )
    DocumentImportService(library.repository_for(project)).publish(parsed)

    hits = RetrievalService(data_root=data_root).search(project, "echo", limit=10).hits

    assert len(hits) == 3
    assert len({anchor.evidence_id for hit in hits for anchor in hit.anchors}) == sum(
        len(hit.anchors) for hit in hits
    )
    assert all(
        hit.start_offset <= anchor.char_start < anchor.char_end <= hit.end_offset
        for hit in hits
        for anchor in hit.anchors
    )


def test_search_isolated_to_the_requested_project(tmp_path: Path) -> None:
    """Would fail if a local project search could return another project's lineage."""
    data_root = tmp_path / "data"
    _, first_project, _, _, _ = _publish_fixture(data_root=data_root, project_id="project-a")
    _, second_project, _, _, _ = _publish_fixture(data_root=data_root, project_id="project-b")

    first_hits = RetrievalService(data_root=data_root).search(first_project, "control").hits
    second_hits = RetrievalService(data_root=data_root).search(second_project, "control").hits

    assert first_hits and second_hits
    assert {hit.project_id for hit in first_hits} == {"project-a"}
    assert {hit.project_id for hit in second_hits} == {"project-b"}


def test_search_rejects_an_uninitialized_project_database(tmp_path: Path) -> None:
    """Would fail if a file with no local schema were treated as an empty project."""
    data_root = tmp_path / "data"
    paths = ProjectPaths.create(data_root, project_id="project-1")
    paths.database_path.parent.mkdir(parents=True)
    sqlite3.connect(paths.database_path).close()
    project = Project(project_id="project-1", display_name="Research")

    with pytest.raises(RetrievalStorageError, match="could not be searched"):
        RetrievalService(data_root=data_root).search(project, "control")
