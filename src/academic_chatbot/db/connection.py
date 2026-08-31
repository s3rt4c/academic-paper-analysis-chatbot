"""Centralized SQLite connection configuration for local project databases."""

from __future__ import annotations

import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from academic_chatbot.storage.paths import ensure_path_beneath


class DatabasePathError(ValueError):
    """Raised when a database location is outside the configured data root."""


def _verified_pragma(connection: sqlite3.Connection, name: str, expected: object) -> None:
    actual = connection.execute(f"PRAGMA {name}").fetchone()
    if actual is None or actual[0] != expected:
        message = f"SQLite pragma {name} was not configured as required"
        raise sqlite3.DatabaseError(message)


def connect_project_database(database_path: Path, *, data_root: Path) -> sqlite3.Connection:
    """Open a file-backed project database with the required local settings."""

    try:
        path = ensure_path_beneath(root=data_root, candidate=database_path)
    except ValueError as error:
        raise DatabasePathError(str(error)) from error
    if path == data_root.resolve(strict=False):
        raise DatabasePathError("database path must be a file beneath the data root")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        _verified_pragma(connection, "foreign_keys", 1)
        journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
        if journal_mode is None or str(journal_mode[0]).lower() != "wal":
            message = "SQLite WAL mode could not be configured for the project database"
            raise sqlite3.DatabaseError(message)
        connection.execute("PRAGMA synchronous = FULL")
        _verified_pragma(connection, "synchronous", 2)
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection
    except BaseException:
        connection.close()
        raise


def open_read_only_connection(database_path: Path, *, data_root: Path) -> sqlite3.Connection:
    """Open an existing contained project database without write-capable setup."""

    try:
        path = ensure_path_beneath(root=data_root, candidate=database_path)
    except ValueError as error:
        raise DatabasePathError(str(error)) from error
    try:
        metadata = path.stat()
    except FileNotFoundError as error:
        raise DatabasePathError("project database does not exist") from error
    except OSError as error:
        raise DatabasePathError("project database could not be inspected") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise DatabasePathError("project database must be an existing regular file")
    try:
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro", uri=True, isolation_level=None, timeout=5.0
        )
    except sqlite3.Error as error:
        raise DatabasePathError("project database could not be opened read-only") from error
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        _verified_pragma(connection, "foreign_keys", 1)
        connection.execute("PRAGMA query_only = ON")
        _verified_pragma(connection, "query_only", 1)
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection
    except BaseException:
        connection.close()
        raise


@contextmanager
def immediate_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Run an explicit transaction that acquires a write reservation up front."""

    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()
