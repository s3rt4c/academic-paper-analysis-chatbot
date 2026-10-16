"""Logical read-only admission with legitimate SQLite-owned WAL coordination."""

import hashlib
import sqlite3
from pathlib import Path

import pytest

from academic_chatbot.db.connection import open_read_only_connection
from academic_chatbot.library.repository import ProjectRepository
from tests.fixtures.evidence_bundle.database import database


def denied(*args, **kwargs):
    raise AssertionError("scoped reader invoked an application write/launch")


def state(connection):
    names = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return (
        tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM sqlite_master ORDER BY type,name"
            ).fetchall()
        ),
        {
            name[0]: sorted(
                (
                    tuple(row)
                    for row in connection.execute(
                        'SELECT * FROM "' + name[0].replace('"', '""') + '"'
                    ).fetchall()
                ),
                key=repr,
            )
            for name in names
        },
    )


def files(root):
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("journal", ["DELETE", "WAL"])
def test_admission_logical_read_only_exact_database_family(tmp_path, monkeypatch, journal):
    db = database(tmp_path)
    repository = ProjectRepository(db.paths)
    assert hasattr(repository, "sole_current_paper_source_read_only"), "read-only admission absent"
    writer = db.connect()
    writer.execute("PRAGMA journal_mode=" + journal)
    baseline = state(writer)
    writer.close()
    before = files(db.paths.data_root)
    monkeypatch.setattr(repository, "_connection", denied)
    monkeypatch.setattr("academic_chatbot.library.repository.connect_project_database", denied)
    monkeypatch.setattr(Path, "mkdir", denied)
    monkeypatch.setattr("socket.create_connection", denied)
    monkeypatch.setattr("subprocess.Popen", denied)
    source = repository.sole_current_paper_source_read_only(
        project_id="project-one", paper_id="paper-one"
    )
    assert source.file_version_id == db.request.scope.file_version_id
    after = files(db.paths.data_root)
    assert all(after.get(name) == digest for name, digest in before.items())
    added = set(after) - set(before)
    family = {str(db.paths.database_path) + suffix for suffix in ("-wal", "-shm")}
    assert added <= family
    if journal == "DELETE":
        assert not added
    elif str(db.paths.database_path) + "-wal" in added:
        assert Path(str(db.paths.database_path) + "-wal").stat().st_size == 0
    connection = open_read_only_connection(db.paths.database_path, data_root=db.paths.data_root)
    try:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        for sql in (
            "INSERT INTO projects VALUES ('evil','now')",
            "UPDATE projects SET created_at='evil'",
            "DELETE FROM projects",
            "CREATE TABLE evil(x)",
            "ALTER TABLE projects ADD COLUMN evil",
            "DROP TABLE projects",
        ):
            with pytest.raises(sqlite3.DatabaseError):
                connection.execute(sql)
        assert state(connection) == baseline
    finally:
        connection.close()


def test_committed_wal_visibility_and_fresh_admission(tmp_path):
    db = database(tmp_path)
    repository = ProjectRepository(db.paths)
    assert hasattr(repository, "sole_current_paper_source_read_only"), "read-only admission absent"
    writer = db.connect()
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE synthetic_marker(value)")
    writer.execute("INSERT INTO synthetic_marker VALUES ('committed-one')")
    baseline = state(writer)
    before_main = db.paths.database_path.read_bytes()
    wal = Path(str(db.paths.database_path) + "-wal")
    before_wal = wal.read_bytes()
    reader = open_read_only_connection(db.paths.database_path, data_root=db.paths.data_root)
    try:
        reader.execute("BEGIN")
        assert reader.execute("SELECT value FROM synthetic_marker").fetchone()[0] == "committed-one"
        assert state(reader) == baseline
        first = repository.sole_current_paper_source_read_only(
            project_id="project-one", paper_id="paper-one"
        )
        assert first.file_version_id == db.request.scope.file_version_id
        assert db.paths.database_path.read_bytes() == before_main
        assert wal.read_bytes() == before_wal
        writer.execute("DELETE FROM generation_publications")
        assert reader.execute("SELECT count(*) FROM generation_publications").fetchone()[0] == 1
        from academic_chatbot.retrieval.scoped_models import ScopedRetrievalError

        with pytest.raises(ScopedRetrievalError) as caught:
            repository.sole_current_paper_source_read_only(
                project_id="project-one", paper_id="paper-one"
            )
        assert caught.value.code == "SOURCE_NOT_CURRENT"
    finally:
        reader.rollback()
        reader.close()
        writer.close()


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_unsafe_existing_sidecar_is_rejected(tmp_path, suffix):
    db = database(tmp_path)
    repository = ProjectRepository(db.paths)
    assert hasattr(repository, "sole_current_paper_source_read_only"), "read-only admission absent"
    path = Path(str(db.paths.database_path) + suffix)
    # Setup finishes/checkpoints on last writer close; no manually manufactured WAL data.
    if path.exists():
        pytest.fail("fixture did not finish writer lifecycle")
    path.mkdir()
    from academic_chatbot.retrieval.scoped_models import ScopedRetrievalError

    with pytest.raises(ScopedRetrievalError) as caught:
        repository.sole_current_paper_source_read_only(
            project_id="project-one", paper_id="paper-one"
        )
    assert caught.value.code == "UNSAFE_PATH"


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_all_scoped_readers_use_actual_ro_uri_and_no_application_writers(
    tmp_path, monkeypatch, mode
):
    from academic_chatbot.db import connection as db_connection
    from academic_chatbot.retrieval.scoped_service import SourceScopedRetrievalService
    from academic_chatbot.retrieval.service import RetrievalService
    from tests.fixtures.retrieval.scoped import crowded

    fixture = crowded(tmp_path, vectors=mode != "lexical")
    writer = fixture.db.connect()
    baseline = state(writer)
    writer.close()
    before = files(fixture.db.paths.data_root)
    calls = []
    actual = db_connection.sqlite3.connect

    def connect(database_uri, *args, **kwargs):
        assert kwargs.get("uri") is True and "mode=ro" in database_uri
        assert "immutable" not in database_uri and "nolock" not in database_uri
        result = actual(database_uri, *args, **kwargs)
        calls.append(database_uri)
        return result

    monkeypatch.setattr(db_connection.sqlite3, "connect", connect)
    monkeypatch.setattr("academic_chatbot.library.repository.connect_project_database", denied)
    monkeypatch.setattr("academic_chatbot.db.migrations.MigrationRunner.migrate_copy", denied)
    monkeypatch.setattr(Path, "mkdir", denied)
    monkeypatch.setattr(Path, "write_text", denied)
    monkeypatch.setattr(Path, "write_bytes", denied)
    monkeypatch.setattr("socket.create_connection", denied)
    monkeypatch.setattr("socket.socket", denied)
    monkeypatch.setattr("subprocess.Popen", denied)
    monkeypatch.setattr("academic_chatbot.embeddings.embedder.OfflineEmbedder.open", denied)
    repository = ProjectRepository(fixture.db.paths)
    monkeypatch.setattr(repository, "_connection", denied)
    reader = SourceScopedRetrievalService(
        data_root=fixture.db.paths.data_root,
        repository=repository,
        lexical_service=RetrievalService(data_root=fixture.db.paths.data_root),
        semantic_service=fixture.semantic,
    )
    request = reader.prepare_request(
        project_id="project-one",
        paper_id="selected",
        query="Alpha",
        mode=mode,
        embedding_profile_id=fixture.semantic.profile.embedding_profile_id
        if mode != "lexical"
        else None,
    )
    result = reader.search_scoped(request)
    assert len(result.hits) == 2 and calls
    after = files(fixture.db.paths.data_root)
    family = {str(fixture.db.paths.database_path) + suffix for suffix in ("-wal", "-shm")}
    assert set(after) - set(before) <= family
    assert all(after.get(name) == digest for name, digest in before.items() if name not in family)
    connection = open_read_only_connection(
        fixture.db.paths.database_path, data_root=fixture.db.paths.data_root
    )
    try:
        assert state(connection) == baseline
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
    finally:
        connection.close()


def test_legitimate_shm_readmark_changes_preserve_main_and_wal(tmp_path):
    db = database(tmp_path)
    writer = db.connect()
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE synthetic_marker(value)")
    writer.execute("INSERT INTO synthetic_marker VALUES ('one')")
    reader = open_read_only_connection(db.paths.database_path, data_root=db.paths.data_root)
    fresh = None
    try:
        reader.execute("BEGIN")
        assert reader.execute("SELECT value FROM synthetic_marker").fetchone()[0] == "one"
        writer.execute("UPDATE synthetic_marker SET value='two'")
        main = db.paths.database_path.read_bytes()
        wal = Path(str(db.paths.database_path) + "-wal").read_bytes()
        shm = Path(str(db.paths.database_path) + "-shm").read_bytes()
        fresh = open_read_only_connection(db.paths.database_path, data_root=db.paths.data_root)
        assert fresh.execute("SELECT value FROM synthetic_marker").fetchone()[0] == "two"
        assert reader.execute("SELECT value FROM synthetic_marker").fetchone()[0] == "one"
        assert db.paths.database_path.read_bytes() == main
        assert Path(str(db.paths.database_path) + "-wal").read_bytes() == wal
        # Reader coordination may change SHM. Prove this fixture exercises that permission.
        assert Path(str(db.paths.database_path) + "-shm").read_bytes() != shm
    finally:
        if fresh:
            fresh.close()
        reader.rollback()
        reader.close()
        writer.close()


@pytest.mark.parametrize("entry", ["ancestor", "database", "-wal", "-shm"])
def test_scoped_guard_rejects_reparse_metadata_without_resolving_it(tmp_path, monkeypatch, entry):
    import stat
    from types import SimpleNamespace

    from academic_chatbot.library.repository import guard_scoped_database
    from academic_chatbot.retrieval.scoped_models import ScopedRetrievalError

    db = database(tmp_path)
    target = (
        db.paths.data_root
        if entry == "ancestor"
        else (
            db.paths.database_path
            if entry == "database"
            else Path(str(db.paths.database_path) + entry)
        )
    )
    if entry in {"-wal", "-shm"}:
        target.write_bytes(b"synthetic unsafe alias")
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        metadata = original(path, *args, **kwargs)
        if path == target:
            return SimpleNamespace(
                st_mode=metadata.st_mode, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT
            )
        return metadata

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ScopedRetrievalError) as caught:
        guard_scoped_database(db.paths.data_root, "project-one")
    assert caught.value.code == "UNSAFE_PATH"
