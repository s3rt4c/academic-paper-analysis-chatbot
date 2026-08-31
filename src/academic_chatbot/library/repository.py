"""Parameterized SQLite persistence for the minimal local library."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import cast

from academic_chatbot.db.connection import connect_project_database, immediate_transaction
from academic_chatbot.domain.library import FileVersion, Paper, Project
from academic_chatbot.storage.paths import ProjectPaths


class DuplicateProjectError(ValueError):
    """Raised when an existing project ID is requested again."""


class UnknownProjectError(ValueError):
    """Raised when a paper is assigned to a project that does not exist."""


class UnknownPaperError(ValueError):
    """Raised when admission is requested for an unknown project-owned paper."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ProjectRepository:
    """Persistence operations only; filesystem publication belongs to admission."""

    def __init__(self, paths: ProjectPaths) -> None:
        self._paths = paths

    def project_exists(self, project_id: str) -> bool:
        return self._one("SELECT 1 FROM projects WHERE project_id = ?", (project_id,)) is not None

    def create_project(self, project: Project) -> None:
        try:
            with self._connection() as connection, immediate_transaction(connection):
                connection.execute(
                    "INSERT INTO projects (project_id, created_at) VALUES (?, ?)",
                    (project.project_id, _timestamp()),
                )
        except sqlite3.IntegrityError as error:
            raise DuplicateProjectError(f"project already exists: {project.project_id}") from error

    def paper_exists(self, paper_id: str, project_id: str) -> bool:
        return self._one(
            "SELECT 1 FROM papers WHERE paper_id = ? AND project_id = ?", (paper_id, project_id)
        ) is not None

    def create_paper(self, paper: Paper) -> None:
        if not self.project_exists(paper.project_id):
            raise UnknownProjectError(f"unknown project: {paper.project_id}")
        try:
            with self._connection() as connection, immediate_transaction(connection):
                connection.execute(
                    "INSERT INTO papers (paper_id, project_id, created_at) VALUES (?, ?, ?)",
                    (paper.paper_id, paper.project_id, _timestamp()),
                )
        except sqlite3.IntegrityError as error:
            raise UnknownPaperError(f"paper already exists: {paper.paper_id}") from error

    def ensure_paper_belongs_to_project(self, paper_id: str, project_id: str) -> None:
        if not self.paper_exists(paper_id, project_id):
            message = f"paper {paper_id} does not belong to project {project_id}"
            raise UnknownPaperError(message)

    def register_file_version(self, *, file_version: FileVersion) -> FileVersion:
        with self._connection() as connection, immediate_transaction(connection):
            row = connection.execute(
                """
                SELECT file_version_id, paper_id, sha256, original_relative_path
                FROM file_versions WHERE sha256 = ?
                """,
                (file_version.sha256,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO file_versions
                        (file_version_id, paper_id, sha256, original_relative_path, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        file_version.file_version_id,
                        file_version.paper_id,
                        file_version.sha256,
                        file_version.stored_relative_path,
                        _timestamp(),
                    ),
                )
                return file_version
        return FileVersion(
            file_version_id=str(row[0]),
            paper_id=str(row[1]),
            sha256=str(row[2]),
            byte_length=file_version.byte_length,
            stored_relative_path=str(row[3]),
        )

    def file_version_exists(self, sha256: str) -> bool:
        return self._one("SELECT 1 FROM file_versions WHERE sha256 = ?", (sha256,)) is not None

    def file_version_count(self) -> int:
        row = self._one("SELECT count(*) FROM file_versions", ())
        assert row is not None
        return int(row[0])

    def _one(self, statement: str, parameters: tuple[str, ...]) -> sqlite3.Row | None:
        with self._connection() as connection:
            row = connection.execute(statement, parameters).fetchone()
            return cast(sqlite3.Row | None, row)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = connect_project_database(
            self._paths.database_path, data_root=self._paths.data_root
        )
        try:
            yield connection
        finally:
            connection.close()
