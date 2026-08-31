from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from academic_chatbot.cli import main
from academic_chatbot.db.connection import open_read_only_connection
from academic_chatbot.documents import import_service
from academic_chatbot.documents.import_service import DocumentImportService
from academic_chatbot.documents.native_pdf import NativePdfParser
from academic_chatbot.domain.library import FileVersion
from academic_chatbot.library.service import LibraryService
from academic_chatbot.storage.paths import ProjectPaths

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "pdfs" / "native_anchor.pdf"


def _invoke(capsys, arguments: list[str]) -> dict[str, object]:
    assert main(arguments) == 0
    return json.loads(capsys.readouterr().out)


def test_native_document_cli_journey_returns_active_exact_evidence(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """Prove the complete native PDF path ends in active, exact local evidence."""
    data_root = tmp_path / "data"
    root = ["--data-root", str(data_root), "--max-pdf-bytes", "1000000"]
    project = _invoke(
        capsys,
        [*root, "project", "create", "--project-id", "project-1", "--display-name", "Research"],
    )
    paper = _invoke(
        capsys,
        [*root, "paper", "create", "--project-id", "project-1", "--paper-id", "paper-1"],
    )
    imported = _invoke(
        capsys,
        [
            *root,
            "import-pdf",
            "--project-id",
            "project-1",
            "--paper-id",
            "paper-1",
            "--source",
            str(_FIXTURE),
        ],
    )
    file_version = FileVersion.model_validate(imported["file_version"])
    first_generation = str(imported["generation"]["document_generation_id"])
    paths = ProjectPaths.create(data_root, project_id="project-1")
    assert paths.resolve_relative(file_version.stored_relative_path).is_file()
    first_result = _invoke(
        capsys,
        [*root, "search", "--project-id", "project-1", "--query", "control"],
    )
    first_hits = first_result["hits"]
    assert isinstance(first_hits, list) and first_hits
    assert {hit["document_generation_id"] for hit in first_hits} == {first_generation}

    library = LibraryService(data_root=data_root, max_pdf_bytes=1_000_000)
    parsed = NativePdfParser(paths).parse(file_version)
    monkeypatch.setattr(import_service, "DOCUMENT_GENERATION_PROFILE_ID", "native-lexical-fts-v2")
    replacement = DocumentImportService(
        library.repository_for_project_id("project-1")
    ).publish(parsed)
    assert replacement.document_generation_id != first_generation

    result = _invoke(capsys, [*root, "search", "--project-id", "project-1", "--query", "control"])
    hits = result["hits"]
    assert isinstance(hits, list) and hits
    assert project["project_id"] == "project-1"
    assert paper["paper_id"] == "paper-1"
    assert all(hit["project_id"] == "project-1" for hit in hits)
    assert all(hit["paper_id"] == "paper-1" for hit in hits)
    assert all(hit["file_version_id"] == file_version.file_version_id for hit in hits)
    assert {hit["document_generation_id"] for hit in hits} == {
        replacement.document_generation_id
    }
    for hit in hits:
        assert hit["page_id"]
        assert hit["chunk_id"]
        assert hit["rank"] >= 1
        assert hit["anchors"]
        assert hit["chunk_text"] == hit["anchors"][0]["canonical_page_text"][
            hit["start_offset"] : hit["end_offset"]
        ]
        assert all(
            hit["start_offset"] <= anchor["char_start"] < anchor["char_end"] <= hit["end_offset"]
            for anchor in hit["anchors"]
        )
        assert all(
            anchor["file_version_id"] == file_version.file_version_id
            for anchor in hit["anchors"]
        )
    with library.repository_for_project_id("project-1")._connection() as connection:
        assert connection.execute(
            "SELECT document_generation_id FROM generation_publications"
        ).fetchone()[0] == replacement.document_generation_id
        assert connection.execute(
            "SELECT count(*) FROM document_generations"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT count(*) FROM document_generations WHERE document_generation_id = ?",
            (first_generation,),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT 1 FROM generation_publications WHERE document_generation_id = ?",
            (first_generation,),
        ).fetchone() is None
    connection = open_read_only_connection(paths.database_path, data_root=data_root)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()


def test_python_module_entry_displays_cli_help() -> None:
    """Prove the installed module entry boundary is runnable without exercising the pipeline."""
    completed = subprocess.run(
        [sys.executable, "-m", "academic_chatbot", "--help"],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "import-pdf" in completed.stdout
