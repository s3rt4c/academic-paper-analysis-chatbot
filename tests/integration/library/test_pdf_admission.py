from __future__ import annotations

import hashlib
from pathlib import Path

from academic_chatbot.db.connection import connect_project_database
from academic_chatbot.library.service import LibraryService
from academic_chatbot.storage.paths import ProjectPaths


def test_admission_publishes_matching_filesystem_and_database_state(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    source = tmp_path / "source.pdf"
    source_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
    source.write_bytes(source_bytes)
    service = LibraryService(data_root=data_root, max_pdf_bytes=1024)
    project = service.create_project(display_name="Research", project_id="project-1")
    paper = service.create_paper(project_id=project.project_id, paper_id="paper-1")

    admitted = service.admit_pdf(
        project_id=project.project_id, paper_id=paper.paper_id, source_path=source
    )

    paths = ProjectPaths.create(data_root, project_id=project.project_id)
    stored = paths.resolve_relative(admitted.stored_relative_path)
    connection = connect_project_database(paths.database_path, data_root=data_root)
    try:
        row = connection.execute(
            "SELECT paper_id, sha256, original_relative_path FROM file_versions"
        ).fetchone()
    finally:
        connection.close()
    assert stored.read_bytes() == source_bytes
    assert hashlib.sha256(stored.read_bytes()).hexdigest() == admitted.sha256
    assert tuple(row) == (paper.paper_id, admitted.sha256, admitted.stored_relative_path)
