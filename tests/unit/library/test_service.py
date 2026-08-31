from __future__ import annotations

from pathlib import Path

import pytest

from academic_chatbot.library.repository import DuplicateProjectError
from academic_chatbot.library.service import LibraryService
from academic_chatbot.storage.paths import ProjectPaths


def test_project_creation_persists_a_stable_id_without_using_display_name_as_a_path(
    tmp_path: Path,
) -> None:
    service = LibraryService(data_root=tmp_path / "data", max_pdf_bytes=1024)

    project = service.create_project(display_name="../research papers", project_id="project-1")

    assert project.project_id == "project-1"
    assert project.display_name == "../research papers"
    paths = ProjectPaths.create(tmp_path / "data", project_id=project.project_id)
    assert paths.project_root.is_relative_to(tmp_path / "data")
    assert service.repository_for(project).project_exists(project.project_id)


def test_duplicate_project_id_has_clear_error_semantics(tmp_path: Path) -> None:
    service = LibraryService(data_root=tmp_path / "data", max_pdf_bytes=1024)
    service.create_project(display_name="One", project_id="project-1")

    with pytest.raises(DuplicateProjectError, match="project-1"):
        service.create_project(display_name="Again", project_id="project-1")


def test_paper_creation_persists_the_project_ownership(tmp_path: Path) -> None:
    service = LibraryService(data_root=tmp_path / "data", max_pdf_bytes=1024)
    project = service.create_project(display_name="One", project_id="project-1")

    paper = service.create_paper(project_id=project.project_id, paper_id="paper-1")

    assert paper.paper_id == "paper-1"
    assert paper.project_id == project.project_id
    assert service.repository_for(project).paper_exists(paper.paper_id, project.project_id)
