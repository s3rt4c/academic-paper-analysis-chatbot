"""Use-case coordination for minimal local projects, papers, and PDF admission."""

from __future__ import annotations

import uuid
from pathlib import Path

from academic_chatbot.db.migrations import MigrationRunner
from academic_chatbot.documents.admission import PdfAdmissionService
from academic_chatbot.domain.library import FileVersion, Paper, Project
from academic_chatbot.library.repository import ProjectRepository
from academic_chatbot.storage.paths import ProjectPaths


class LibraryService:
    """Create minimal library identities and coordinate safe PDF admission."""

    def __init__(self, *, data_root: Path, max_pdf_bytes: int) -> None:
        if max_pdf_bytes <= 0:
            raise ValueError("max_pdf_bytes must be positive")
        self._data_root = data_root.resolve(strict=False)
        self._max_pdf_bytes = max_pdf_bytes
        self._migrations = MigrationRunner(
            Path(__file__).parents[1] / "db" / "migrations"
        )

    def create_project(self, *, display_name: str, project_id: str | None = None) -> Project:
        project = Project(
            project_id=project_id or f"project-{uuid.uuid4().hex}", display_name=display_name
        )
        paths = ProjectPaths.create(self._data_root, project_id=project.project_id)
        self._migrations.migrate_copy(paths.database_path, data_root=self._data_root)
        ProjectRepository(paths).create_project(project)
        return project

    def create_paper(self, *, project_id: str, paper_id: str | None = None) -> Paper:
        paths = ProjectPaths.create(self._data_root, project_id=project_id)
        repository = ProjectRepository(paths)
        paper = Paper(paper_id=paper_id or f"paper-{uuid.uuid4().hex}", project_id=project_id)
        repository.create_paper(paper)
        return paper

    def admit_pdf(self, *, project_id: str, paper_id: str, source_path: Path) -> FileVersion:
        repository = self.repository_for_project_id(project_id)
        repository.ensure_paper_belongs_to_project(paper_id, project_id)
        return PdfAdmissionService(
            paths=ProjectPaths.create(self._data_root, project_id=project_id),
            repository=repository,
            max_pdf_bytes=self._max_pdf_bytes,
        ).admit(paper_id=paper_id, source_path=source_path)

    def repository_for(self, project: Project) -> ProjectRepository:
        return self.repository_for_project_id(project.project_id)

    def repository_for_project_id(self, project_id: str) -> ProjectRepository:
        return ProjectRepository(ProjectPaths.create(self._data_root, project_id=project_id))
