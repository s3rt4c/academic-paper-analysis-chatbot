from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from academic_chatbot.documents import import_service
from academic_chatbot.documents.chunking import LexicalChunk
from academic_chatbot.documents.import_service import DocumentImportService
from academic_chatbot.documents.native_pdf import NativePdfParser
from academic_chatbot.library.repository import (
    DocumentGenerationPublicationError,
    ProjectRepository,
)
from academic_chatbot.library.service import LibraryService
from academic_chatbot.storage.paths import ProjectPaths

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "pdfs" / "native_anchor.pdf"


def _parsed_fixture(tmp_path: Path):
    service = LibraryService(data_root=tmp_path / "data", max_pdf_bytes=1_000_000)
    project = service.create_project(display_name="Research", project_id="project-1")
    paper = service.create_paper(project_id=project.project_id, paper_id="paper-1")
    file_version = service.admit_pdf(
        project_id=project.project_id,
        paper_id=paper.paper_id,
        source_path=_FIXTURE,
    )
    paths = ProjectPaths.create(tmp_path / "data", project_id=project.project_id)
    parsed = NativePdfParser(paths).parse(file_version)
    return service, project.project_id, paths, parsed


def test_publish_persists_one_complete_active_generation_with_exact_page_chunk_and_fts_lineage(
    tmp_path: Path,
) -> None:
    """Would fail if publication exposed incomplete pages, chunks, or FTS rows."""
    service, project_id, _, parsed = _parsed_fixture(tmp_path)
    repository = service.repository_for_project_id(project_id)

    published = DocumentImportService(repository).publish(parsed)
    repeated = DocumentImportService(repository).publish(parsed)

    assert published.reused is False
    assert repeated == published.model_copy(update={"reused": True})
    assert published.file_version_id == parsed.file_version.file_version_id
    assert published.page_count == len(parsed.pages)
    assert published.chunk_count > 0

    with repository._connection() as connection:
        active = connection.execute(
            "SELECT file_version_id, document_generation_id FROM generation_publications"
        ).fetchall()
        assert [tuple(row) for row in active] == [
            (parsed.file_version.file_version_id, published.document_generation_id)
        ]
        pages = connection.execute(
            """
            SELECT physical_page_index, page_number, printed_page_label, canonical_text,
                   extraction_quality, needs_ocr
            FROM pages WHERE document_generation_id = ? ORDER BY physical_page_index
            """,
            (published.document_generation_id,),
        ).fetchall()
        assert [row[3] for row in pages] == [page.canonical_text for page in parsed.pages]
        assert [(row[0], row[1]) for row in pages] == [(0, 1), (1, 2)]
        chunks = connection.execute(
            """
            SELECT c.chunk_id, c.page_id, c.start_offset, c.end_offset, c.chunk_text,
                   p.canonical_text
            FROM chunks AS c JOIN pages AS p ON p.page_id = c.page_id
            WHERE c.document_generation_id = ? ORDER BY c.page_id, c.ordinal
            """,
            (published.document_generation_id,),
        ).fetchall()
        assert chunks
        assert all(row[4] == row[5][row[2] : row[3]] for row in chunks)
        fts_ids = connection.execute("SELECT chunk_id FROM chunk_fts ORDER BY chunk_id").fetchall()
        assert sorted(row[0] for row in chunks) == [row[0] for row in fts_ids]
        assert connection.execute("INSERT INTO chunk_fts(chunk_fts) VALUES ('integrity-check')")


def test_needs_ocr_pages_publish_status_without_searchable_or_fabricated_chunks(
    tmp_path: Path,
) -> None:
    """Would fail if an insufficient native layer silently became FTS text or invoked OCR."""
    source = tmp_path / "empty-native.pdf"
    pdf = canvas.Canvas(str(source), invariant=1)
    pdf.showPage()
    pdf.save()
    service = LibraryService(data_root=tmp_path / "data", max_pdf_bytes=1_000_000)
    project = service.create_project(display_name="Research", project_id="project-1")
    paper = service.create_paper(project_id=project.project_id, paper_id="paper-1")
    file_version = service.admit_pdf(
        project_id=project.project_id, paper_id=paper.paper_id, source_path=source
    )
    paths = ProjectPaths.create(tmp_path / "data", project_id=project.project_id)
    parsed = NativePdfParser(paths).parse(file_version)
    repository = service.repository_for_project_id(project.project_id)

    published = DocumentImportService(repository).publish(parsed)

    assert published.chunk_count == 0
    with repository._connection() as connection:
        assert tuple(
            connection.execute(
                "SELECT extraction_quality, needs_ocr, canonical_text FROM pages"
            ).fetchone()
        ) == ("empty_native_text", 1, "")
        assert connection.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM chunk_fts").fetchone()[0] == 0


def test_new_processing_profile_keeps_old_generation_immutable_and_replaces_active_fts_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if active replacement rewrote history or left stale FTS entries searchable."""
    service, project_id, _, parsed = _parsed_fixture(tmp_path)
    repository = service.repository_for_project_id(project_id)
    first = DocumentImportService(repository).publish(parsed)
    monkeypatch.setattr(import_service, "DOCUMENT_GENERATION_PROFILE_ID", "native-lexical-fts-v2")

    replacement = DocumentImportService(repository).publish(parsed)

    assert replacement.reused is False
    assert replacement.document_generation_id != first.document_generation_id
    with repository._connection() as connection:
        assert connection.execute(
            "SELECT document_generation_id FROM generation_publications"
        ).fetchone()[0] == replacement.document_generation_id
        assert connection.execute("SELECT count(*) FROM document_generations").fetchone()[0] == 2
        assert connection.execute(
            "SELECT count(*) FROM chunks WHERE document_generation_id = ?",
            (first.document_generation_id,),
        ).fetchone()[0] == first.chunk_count
        assert (
            connection.execute("SELECT count(*) FROM chunk_fts").fetchone()[0]
            == replacement.chunk_count
        )
        stale_fts = connection.execute(
            """
            SELECT 1 FROM chunk_fts AS f
            JOIN chunks AS c ON c.chunk_id = f.chunk_id
            WHERE c.document_generation_id <> ?
            """,
            (replacement.document_generation_id,),
        ).fetchone()
        assert stale_fts is None


def test_new_lexical_profile_creates_a_distinct_immutable_generation_and_active_fts_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if chunk-profile changes silently reused an incompatible generation."""
    service, project_id, _, parsed = _parsed_fixture(tmp_path)
    repository = service.repository_for_project_id(project_id)
    first = DocumentImportService(repository).publish(parsed)
    monkeypatch.setattr(import_service, "LEXICAL_CHUNK_PROFILE_ID", "lexical-chunk-v2")

    replacement = DocumentImportService(repository).publish(parsed)

    assert replacement.reused is False
    assert replacement.document_generation_id != first.document_generation_id
    assert replacement.processing_profile_id != first.processing_profile_id
    with repository._connection() as connection:
        assert connection.execute(
            "SELECT document_generation_id FROM generation_publications"
        ).fetchone()[0] == replacement.document_generation_id
        assert connection.execute("SELECT count(*) FROM document_generations").fetchone()[0] == 2
        assert {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT processing_profile_id FROM chunks "
                "WHERE document_generation_id = ?",
                (replacement.document_generation_id,),
            )
        } == {"lexical-chunk-v2"}


def test_publish_rejects_parsed_file_version_that_disagrees_with_authoritative_paper_ownership(
    tmp_path: Path,
) -> None:
    """Would fail if a caller could publish one FileVersion ID with another Paper's identity."""
    service, project_id, _, parsed = _parsed_fixture(tmp_path)
    repository = service.repository_for_project_id(project_id)
    tampered = parsed.model_copy(
        update={"file_version": parsed.file_version.model_copy(update={"paper_id": "paper-forged"})}
    )

    with pytest.raises(DocumentGenerationPublicationError, match="does not match project record"):
        DocumentImportService(repository).publish(tampered)

    with repository._connection() as connection:
        assert connection.execute("SELECT count(*) FROM document_generations").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM pages").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM chunk_fts").fetchone()[0] == 0


def test_reuse_revalidates_complete_active_fts_mapping_before_returning_success(
    tmp_path: Path,
) -> None:
    """Would fail if a corrupted active FTS row returned an unearned reused result."""
    service, project_id, _, parsed = _parsed_fixture(tmp_path)
    repository = service.repository_for_project_id(project_id)
    published = DocumentImportService(repository).publish(parsed)

    with repository._connection() as connection:
        connection.execute("DELETE FROM chunk_fts WHERE rowid = (SELECT min(rowid) FROM chunk_fts)")

    with pytest.raises(DocumentGenerationPublicationError, match="candidate generation"):
        DocumentImportService(repository).publish(parsed)

    with repository._connection() as connection:
        assert connection.execute(
            "SELECT document_generation_id FROM generation_publications"
        ).fetchone()[0] == published.document_generation_id
        assert connection.execute("SELECT count(*) FROM document_generations").fetchone()[0] == 1


def test_failure_immediately_before_active_switch_keeps_old_generation_and_fts_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if a candidate could replace an active generation before commit validation."""
    service, project_id, _, parsed = _parsed_fixture(tmp_path)
    repository = service.repository_for_project_id(project_id)
    first = DocumentImportService(repository).publish(parsed)
    monkeypatch.setattr(import_service, "DOCUMENT_GENERATION_PROFILE_ID", "native-lexical-fts-v2")

    def fail_active_switch(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic active-switch failure")

    monkeypatch.setattr(ProjectRepository, "_activate_generation", staticmethod(fail_active_switch))

    with pytest.raises(RuntimeError, match="active-switch"):
        DocumentImportService(repository).publish(parsed)

    with repository._connection() as connection:
        assert connection.execute(
            "SELECT document_generation_id FROM generation_publications"
        ).fetchone()[0] == first.document_generation_id
        assert connection.execute("SELECT count(*) FROM document_generations").fetchone()[0] == 1
        assert (
            connection.execute("SELECT count(*) FROM chunk_fts").fetchone()[0]
            == first.chunk_count
        )


@pytest.mark.parametrize(
    ("trigger_name", "table_name", "when_clause", "failure_message"),
    [
        ("fail_candidate_pages", "pages", "", "synthetic candidate creation failure"),
        (
            "fail_second_page",
            "pages",
            "WHEN NEW.physical_page_index = 1",
            "synthetic page persistence failure",
        ),
        ("fail_candidate_chunks", "chunks", "", "synthetic chunk persistence failure"),
    ],
)
def test_failure_after_candidate_creation_rolls_back_candidate_and_keeps_old_active_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trigger_name: str,
    table_name: str,
    when_clause: str,
    failure_message: str,
) -> None:
    """Would fail if any candidate persistence error exposed or retained a partial generation."""
    service, project_id, _, parsed = _parsed_fixture(tmp_path)
    repository = service.repository_for_project_id(project_id)
    first = DocumentImportService(repository).publish(parsed)
    monkeypatch.setattr(import_service, "DOCUMENT_GENERATION_PROFILE_ID", "native-lexical-fts-v2")

    with repository._connection() as connection:
        connection.execute(
            f"""
            CREATE TRIGGER {trigger_name} BEFORE INSERT ON {table_name}
            {when_clause}
            BEGIN
                SELECT RAISE(ABORT, '{failure_message}');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match=failure_message):
        DocumentImportService(repository).publish(parsed)

    with repository._connection() as connection:
        assert connection.execute(
            "SELECT document_generation_id FROM generation_publications"
        ).fetchone()[0] == first.document_generation_id
        assert connection.execute("SELECT count(*) FROM document_generations").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM pages").fetchone()[0] == first.page_count
        assert connection.execute("SELECT count(*) FROM chunks").fetchone()[0] == first.chunk_count
        assert (
            connection.execute("SELECT count(*) FROM chunk_fts").fetchone()[0]
            == first.chunk_count
        )


def test_failure_while_populating_fts_rolls_back_candidate_and_keeps_old_active_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if FTS writes could leave a candidate's page/chunk rows committed."""
    service, project_id, _, parsed = _parsed_fixture(tmp_path)
    repository = service.repository_for_project_id(project_id)
    first = DocumentImportService(repository).publish(parsed)
    monkeypatch.setattr(import_service, "DOCUMENT_GENERATION_PROFILE_ID", "native-lexical-fts-v2")

    def fail_fts_population(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic FTS population failure")

    monkeypatch.setattr(
        ProjectRepository, "_persist_chunk_fts", staticmethod(fail_fts_population), raising=False
    )

    with pytest.raises(RuntimeError, match="FTS population"):
        DocumentImportService(repository).publish(parsed)

    with repository._connection() as connection:
        assert connection.execute(
            "SELECT document_generation_id FROM generation_publications"
        ).fetchone()[0] == first.document_generation_id
        assert connection.execute("SELECT count(*) FROM document_generations").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM pages").fetchone()[0] == first.page_count
        assert connection.execute("SELECT count(*) FROM chunks").fetchone()[0] == first.chunk_count
        assert (
            connection.execute("SELECT count(*) FROM chunk_fts").fetchone()[0]
            == first.chunk_count
        )


def test_candidate_validation_rejects_balanced_missing_and_duplicate_fts_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if a total FTS row count hid a missing chunk and a duplicate other chunk."""
    service, project_id, _, parsed = _parsed_fixture(tmp_path)
    repository = service.repository_for_project_id(project_id)
    first = DocumentImportService(repository).publish(parsed)
    monkeypatch.setattr(import_service, "DOCUMENT_GENERATION_PROFILE_ID", "native-lexical-fts-v2")
    original_persist_fts = ProjectRepository._persist_chunk_fts
    missing_chunk_id: str | None = None

    def write_balanced_invalid_fts(
        connection: sqlite3.Connection, *, chunks: tuple[LexicalChunk, ...]
    ) -> None:
        nonlocal missing_chunk_id
        original_persist_fts(connection, chunks=chunks)
        chunk = chunks[0]
        if missing_chunk_id is None:
            missing_chunk_id = chunk.chunk_id
            connection.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", (chunk.chunk_id,))
        else:
            connection.execute(
                "INSERT INTO chunk_fts (chunk_id, chunk_text) VALUES (?, ?)",
                (chunk.chunk_id, chunk.text),
            )

    monkeypatch.setattr(
        ProjectRepository, "_persist_chunk_fts", staticmethod(write_balanced_invalid_fts)
    )

    with pytest.raises(DocumentGenerationPublicationError, match="candidate generation"):
        DocumentImportService(repository).publish(parsed)

    with repository._connection() as connection:
        assert connection.execute(
            "SELECT document_generation_id FROM generation_publications"
        ).fetchone()[0] == first.document_generation_id
        assert connection.execute("SELECT count(*) FROM document_generations").fetchone()[0] == 1
        assert (
            connection.execute("SELECT count(*) FROM chunk_fts").fetchone()[0]
            == first.chunk_count
        )
