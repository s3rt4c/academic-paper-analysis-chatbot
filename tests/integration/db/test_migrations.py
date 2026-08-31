from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from academic_chatbot.db.connection import DatabasePathError, connect_project_database
from academic_chatbot.db.migrations import MigrationRunner, MigrationStateError


def _default_migration_directory() -> Path:
    return Path(__file__).parents[3] / "src" / "academic_chatbot" / "db" / "migrations"


def _migration_runner() -> MigrationRunner:
    return MigrationRunner(_default_migration_directory())


def test_new_database_reaches_document_core_migration_with_integrity_pragmas(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    database_path = data_root / "projects" / "project-1" / "project.sqlite3"

    _migration_runner().migrate_copy(database_path, data_root=data_root)

    connection = connect_project_database(database_path, data_root=data_root)
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        versions = connection.execute("SELECT version FROM schema_migrations")
        assert [tuple(row) for row in versions] == [
            (1,)
        ]
        assert connection.execute("SELECT count(*) FROM projects").fetchone()[0] == 0
    finally:
        connection.close()


def test_database_path_outside_the_data_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatabasePathError, match="escapes"):
        connect_project_database(tmp_path.parent / "outside.sqlite3", data_root=tmp_path / "data")


@pytest.mark.parametrize("invalid_path", ("../outside", "C:/outside", "a/./b", "a//b", "a/../b"))
def test_document_core_rejects_noncanonical_persisted_paths(
    tmp_path: Path, invalid_path: str
) -> None:
    data_root = tmp_path / "data"
    database_path = data_root / "project.sqlite3"
    _migration_runner().migrate_copy(database_path, data_root=data_root)
    connection = connect_project_database(database_path, data_root=data_root)
    try:
        connection.execute("INSERT INTO projects VALUES ('project-1', '2026-08-27T00:00:00Z')")
        connection.execute(
            "INSERT INTO papers VALUES ('paper-1', 'project-1', '2026-08-27T00:00:00Z')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO file_versions
                    (file_version_id, paper_id, sha256, original_relative_path, created_at)
                VALUES ('file-1', 'paper-1', ?, ?, '2026-08-27T00:00:00Z')
                """,
                ("a" * 64, invalid_path),
            )
    finally:
        connection.close()


def test_migrations_are_idempotent_and_ordered(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    database_path = data_root / "project.sqlite3"
    migration_directory = tmp_path / "migrations"
    migration_directory.mkdir()
    shutil.copy2(_default_migration_directory() / "0001_document_core.sql", migration_directory)
    (migration_directory / "0002_ordered.sql").write_text(
        "CREATE TABLE ordered_marker (value TEXT NOT NULL) STRICT;\n", encoding="utf-8"
    )
    runner = MigrationRunner(migration_directory)

    runner.migrate_copy(database_path, data_root=data_root)
    runner.migrate_copy(database_path, data_root=data_root)

    connection = connect_project_database(database_path, data_root=data_root)
    try:
        assert [
            tuple(row)
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        ] == [
            (1,),
            (2,),
        ]
        ordered_marker = connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'ordered_marker'"
        ).fetchone()
        assert ordered_marker
    finally:
        connection.close()


def test_failed_migration_preserves_previous_valid_database(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    database_path = data_root / "project.sqlite3"
    _migration_runner().migrate_copy(database_path, data_root=data_root)
    original_bytes = database_path.read_bytes()
    migration_directory = tmp_path / "migrations"
    migration_directory.mkdir()
    shutil.copy2(_default_migration_directory() / "0001_document_core.sql", migration_directory)
    (migration_directory / "0002_broken.sql").write_text(
        "CREATE TABLE transient_marker (value TEXT NOT NULL) STRICT;\nBROKEN SQL;\n",
        encoding="utf-8",
    )

    with pytest.raises(sqlite3.DatabaseError):
        MigrationRunner(migration_directory).migrate_copy(database_path, data_root=data_root)

    assert database_path.read_bytes() == original_bytes
    connection = connect_project_database(database_path, data_root=data_root)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        versions = connection.execute("SELECT version FROM schema_migrations")
        assert [tuple(row) for row in versions] == [
            (1,)
        ]
        transient_marker = connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'transient_marker'"
        ).fetchone()
        assert transient_marker is None
    finally:
        connection.close()


def test_inconsistent_migration_metadata_fails_closed(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    database_path = data_root / "project.sqlite3"
    runner = _migration_runner()
    runner.migrate_copy(database_path, data_root=data_root)
    connection = connect_project_database(database_path, data_root=data_root)
    try:
        connection.execute("UPDATE schema_migrations SET checksum_sha256 = ?", ("0" * 64,))
    finally:
        connection.close()

    with pytest.raises(MigrationStateError, match="inconsistent"):
        runner.migrate_copy(database_path, data_root=data_root)
