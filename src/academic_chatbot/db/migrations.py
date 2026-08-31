"""Deterministic, copy-then-publish SQLite schema migrations."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from academic_chatbot.db.connection import connect_project_database, immediate_transaction
from academic_chatbot.storage.paths import ensure_path_beneath

_MIGRATION_NAME = re.compile(r"(?P<version>[0-9]{4})_(?P<name>[a-z][a-z0-9_]*)\.sql\Z")


class MigrationStateError(RuntimeError):
    """Raised when migration metadata cannot be safely reconciled with files."""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str
    checksum_sha256: str


class MigrationRunner:
    """Apply package-owned SQL migrations to a checked copy before publication."""

    def __init__(self, migration_directory: Path) -> None:
        self._migration_directory = migration_directory.resolve(strict=False)

    def migrate_copy(self, database_path: Path, *, data_root: Path) -> None:
        """Migrate a copy, validate it, then atomically replace the project database."""

        database = ensure_path_beneath(root=data_root, candidate=database_path)
        if database == data_root.resolve(strict=False):
            raise MigrationStateError("database path must be a file beneath the data root")
        database.parent.mkdir(parents=True, exist_ok=True)
        candidate = self._new_candidate_path(database.parent, database.name)
        try:
            if database.exists():
                self._backup_database(database, candidate, data_root=data_root)
            self._apply_all(candidate, data_root=data_root)
            self._validate_database(candidate, data_root=data_root)
            os.replace(candidate, database)
        finally:
            self._remove_candidate_residue(candidate)

    def _new_candidate_path(self, directory: Path, database_name: str) -> Path:
        descriptor, candidate_name = tempfile.mkstemp(
            prefix=f".{database_name}.", suffix=".migration", dir=directory
        )
        os.close(descriptor)
        return Path(candidate_name)

    def _backup_database(self, source_path: Path, candidate_path: Path, *, data_root: Path) -> None:
        source = connect_project_database(source_path, data_root=data_root)
        candidate = connect_project_database(candidate_path, data_root=data_root)
        try:
            source.backup(candidate)
        finally:
            candidate.close()
            source.close()

    def _apply_all(self, candidate_path: Path, *, data_root: Path) -> None:
        migrations = self._read_migrations()
        connection = connect_project_database(candidate_path, data_root=data_root)
        try:
            self._ensure_metadata_table(connection)
            applied = self._read_applied_migrations(connection)
            self._check_applied_state(applied, migrations)
            for migration in migrations:
                if migration.version not in applied:
                    self._apply_one(connection, migration)
        finally:
            connection.close()

    def _read_migrations(self) -> list[Migration]:
        if not self._migration_directory.is_dir():
            message = f"migration directory does not exist: {self._migration_directory}"
            raise MigrationStateError(message)
        migrations: list[Migration] = []
        for path in sorted(self._migration_directory.glob("*.sql")):
            if path.is_symlink() or not path.is_file():
                raise MigrationStateError(f"migration file is not a regular file: {path.name}")
            match = _MIGRATION_NAME.fullmatch(path.name)
            if match is None:
                raise MigrationStateError(f"invalid migration filename: {path.name}")
            sql = path.read_text(encoding="utf-8")
            migrations.append(
                Migration(
                    version=int(match["version"]),
                    name=match["name"],
                    sql=sql,
                    checksum_sha256=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
                )
            )
        versions = [migration.version for migration in migrations]
        if not migrations or len(versions) != len(set(versions)):
            raise MigrationStateError("migration versions must be non-empty and unique")
        return migrations

    @staticmethod
    def _ensure_metadata_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                checksum_sha256 TEXT NOT NULL CHECK(length(checksum_sha256) = 64),
                applied_at TEXT NOT NULL
            ) STRICT
            """
        )

    @staticmethod
    def _read_applied_migrations(connection: sqlite3.Connection) -> dict[int, tuple[str, str]]:
        rows = connection.execute(
            "SELECT version, name, checksum_sha256 FROM schema_migrations ORDER BY version"
        )
        return {int(row[0]): (str(row[1]), str(row[2])) for row in rows}

    @staticmethod
    def _check_applied_state(
        applied: dict[int, tuple[str, str]], migrations: list[Migration]
    ) -> None:
        by_version = {migration.version: migration for migration in migrations}
        for version, (name, checksum) in applied.items():
            migration = by_version.get(version)
            if migration is None or migration.name != name or migration.checksum_sha256 != checksum:
                raise MigrationStateError("inconsistent migration metadata")

    @staticmethod
    def _apply_one(connection: sqlite3.Connection, migration: Migration) -> None:
        with immediate_transaction(connection):
            pending = ""
            for line in migration.sql.splitlines(keepends=True):
                pending += line
                if sqlite3.complete_statement(pending):
                    connection.execute(pending)
                    pending = ""
            if pending.strip():
                message = f"migration {migration.version} has an incomplete SQL statement"
                raise sqlite3.DatabaseError(message)
            connection.execute(
                """
                INSERT INTO schema_migrations (version, name, checksum_sha256, applied_at)
                VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """,
                (migration.version, migration.name, migration.checksum_sha256),
            )

    @staticmethod
    def _validate_database(candidate_path: Path, *, data_root: Path) -> None:
        connection = connect_project_database(candidate_path, data_root=data_root)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            foreign_key_issues = connection.execute("PRAGMA foreign_key_check").fetchall()
            if integrity is None or integrity[0] != "ok" or foreign_key_issues:
                raise sqlite3.DatabaseError("migrated database failed integrity validation")
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            connection.close()

    @staticmethod
    def _remove_candidate_residue(candidate_path: Path) -> None:
        for path in (candidate_path, Path(f"{candidate_path}-wal"), Path(f"{candidate_path}-shm")):
            if path.exists():
                path.unlink()
