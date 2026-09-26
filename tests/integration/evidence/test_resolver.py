from pathlib import Path

import pytest

from academic_chatbot.evidence.resolver import EvidenceReadResolver, EvidenceResolutionError
from tests.fixtures.evidence_bundle.database import database


def test_resolver_accepts_current_generation_of_explicit_older_file_version(tmp_path: Path) -> None:
    db = database(tmp_path, older=True)
    result = EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
    assert result.resolved_scope.file_version_id == db.request.scope.file_version_id
    assert len(result.groups) == len(db.request.candidates) > 0


def test_resolver_rejects_superseded_generation_without_substitution(tmp_path: Path) -> None:
    db = database(tmp_path)
    db.execute("DELETE FROM generation_publications")
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
    assert error.value.error.code == "GENERATION_NOT_CURRENT"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("expected_text_sha256", "a" * 64, "TEXT_DIGEST_MISMATCH"),
        ("expected_anchor_ids", ("wrong",), "ANCHOR_MISMATCH"),
        ("end_offset", 1, "RANGE_MISMATCH"),
    ],
)
def test_resolver_compares_exact_range_digest_and_anchor_ids(
    tmp_path: Path, field, value, code
) -> None:
    db = database(tmp_path)
    ref = db.request.candidates[0]
    bad = ref.model_copy(update={"lexical": ref.lexical.model_copy(update={field: value})})
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(
            db.request.model_copy(update={"candidates": (bad,)})
        )
    assert error.value.error.code == code


def test_resolver_closes_resources_after_every_failure_stage(tmp_path: Path, monkeypatch) -> None:
    from academic_chatbot.evidence import resolver

    db = database(tmp_path)
    opened = []
    original = resolver.open_read_only_connection

    def capture(*args, **kwargs):
        connection = original(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(resolver, "open_read_only_connection", capture)
    db.execute("DELETE FROM page_anchors")
    with pytest.raises(EvidenceResolutionError):
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
    import sqlite3

    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")


@pytest.mark.parametrize(
    ("sql", "code"),
    [
        ("UPDATE papers SET project_id = 'foreign-project'", "SOURCE_SCOPE_MISMATCH"),
        ("UPDATE file_versions SET paper_id = 'foreign-paper'", "SOURCE_SCOPE_MISMATCH"),
        (
            "UPDATE document_generations SET file_version_id = 'foreign-version'",
            "SOURCE_SCOPE_MISMATCH",
        ),
        ("DELETE FROM projects", "SOURCE_NOT_FOUND"),
        ("DELETE FROM papers", "SOURCE_NOT_FOUND"),
        ("DELETE FROM file_versions", "SOURCE_NOT_FOUND"),
        ("DELETE FROM document_generations", "SOURCE_NOT_FOUND"),
        ("DELETE FROM chunks", "REFERENCE_NOT_FOUND"),
        ("DELETE FROM pages", "REFERENCE_NOT_FOUND"),
        ("UPDATE chunks SET chunk_text = 'contradictory'", "STORAGE_INTEGRITY"),
        ("UPDATE pages SET canonical_text = 'contradictory'", "STORAGE_INTEGRITY"),
        ("UPDATE pages SET canonical_text_sha256 = '" + "a" * 64 + "'", "STORAGE_INTEGRITY"),
        ("UPDATE pages SET parser_profile_sha256 = NULL", "STORAGE_INTEGRITY"),
        ("UPDATE pages SET physical_page_index = NULL", "STORAGE_INTEGRITY"),
        ("UPDATE pages SET page_number = 2", "STORAGE_INTEGRITY"),
        ("UPDATE chunks SET processing_profile_id = 'foreign-profile'", "STORAGE_INTEGRITY"),
        ("DELETE FROM chunk_fts", "STORAGE_INTEGRITY"),
        ("UPDATE chunk_fts SET chunk_text = 'contradictory'", "STORAGE_INTEGRITY"),
        ("DELETE FROM page_anchors WHERE char_start = 0", "STORAGE_INTEGRITY"),
        ("UPDATE page_anchors SET boxes_sha256 = '" + "b" * 64 + "'", "STORAGE_INTEGRITY"),
        (
            "UPDATE page_anchors SET page_anchor_id = 'bad' WHERE char_start = 0",
            "STORAGE_INTEGRITY",
        ),
        ("UPDATE page_anchors SET evidence_id = 'bad' WHERE char_start = 0", "STORAGE_INTEGRITY"),
    ],
)
def test_authoritative_corruption_fails(tmp_path, sql, code):
    db = database(tmp_path)
    db.corrupt(sql)
    before = db.logical_state()
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
    assert error.value.error.code == code
    assert db.logical_state() == before
    assert str(error.value) == error.value.error.message
    assert str(db.paths.database_path) not in error.value.error.model_dump_json()


@pytest.mark.parametrize("field", ["project_id", "document_generation_id", "page_id"])
def test_cross_scope_candidate_fails(tmp_path, field):
    db = database(tmp_path)
    ref = db.request.candidates[0]
    parent = ref.parent.model_copy(update={field: "foreign"})
    ref = ref.model_copy(update={"parent": parent})
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(
            db.request.model_copy(update={"candidates": (ref,)})
        )
    assert error.value.error.code in ("SOURCE_SCOPE_MISMATCH", "REFERENCE_NOT_FOUND")


def test_all_candidates_and_input_order_preserved(tmp_path, monkeypatch):
    from academic_chatbot.evidence import resolver
    from academic_chatbot.evidence.models import PreviewBudget

    db = database(tmp_path)
    first = db.request.candidates[0]
    refs = (first.model_copy(update={"reported_rank": 9}), first, first)
    reads = []
    original = resolver._read_page

    def read(*args):
        reads.append(args[1])
        return original(*args)

    monkeypatch.setattr(resolver, "_read_page", read)
    request = db.request.model_copy(
        update={
            "candidates": refs,
            "preview_budget": PreviewBudget(max_content_bytes=77, max_entries=1),
        }
    )
    result = EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
    assert tuple(g.reference.reported_rank for g in result.groups) == (9, 1, 1)
    assert tuple(g.input_position for g in result.groups) == (0, 1, 2)
    assert len(reads) == 1
    bad = first.model_copy(
        update={"lexical": first.lexical.model_copy(update={"expected_text_sha256": "a" * 64})}
    )
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(
            request.model_copy(update={"candidates": (*refs, bad)})
        )
    assert error.value.error.input_position == 3
    assert error.value.error.code == "TEXT_DIGEST_MISMATCH"


def test_request_revalidation_before_storage(tmp_path, monkeypatch):
    from academic_chatbot.evidence import resolver

    db = database(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid request reached storage")

    monkeypatch.setattr(resolver, "open_read_only_connection", forbidden)
    bad = db.request.candidates[0].model_copy(update={"reported_rank": True})
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(
            db.request.model_copy(update={"candidates": (bad,)})
        )
    assert error.value.error.code == "INVALID_REQUEST"


@pytest.mark.parametrize("failure", [False, True])
def test_read_only_no_parser_model_network_or_output(tmp_path, monkeypatch, failure):
    import hashlib
    import socket
    import sqlite3

    from academic_chatbot.documents.native_pdf import NativePdfParser
    from academic_chatbot.embeddings.embedder import OfflineEmbedder
    from academic_chatbot.embeddings.repository import EmbeddingRepository
    from academic_chatbot.evidence import resolver
    from academic_chatbot.library.repository import ProjectRepository
    from academic_chatbot.retrieval.semantic import SemanticRetrievalService
    from academic_chatbot.retrieval.service import RetrievalService

    db = database(tmp_path)
    request = db.semantic()
    if failure:
        db.execute("DELETE FROM page_anchors")
    before = db.logical_state()

    def files():
        return {
            str(p.relative_to(db.paths.project_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in db.paths.project_root.rglob("*")
            if p.is_file() and "project.sqlite3" not in p.name
        }

    artifacts = files()

    def forbidden(*args, **kwargs):
        pytest.fail("resolver invoked a forbidden operation")

    for cls, name in (
        (NativePdfParser, "parse"),
        (OfflineEmbedder, "open"),
        (ProjectRepository, "_connection"),
        (EmbeddingRepository, "_connection"),
        (SemanticRetrievalService, "search"),
        (RetrievalService, "search"),
    ):
        monkeypatch.setattr(cls, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    original = resolver.open_read_only_connection
    opened = []

    def capture(*args, **kwargs):
        conn = original(*args, **kwargs)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM projects")
        opened.append(conn)
        return conn

    monkeypatch.setattr(resolver, "open_read_only_connection", capture)
    if failure:
        with pytest.raises(EvidenceResolutionError):
            EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
    else:
        result = EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
        assert result.model_dump_json()
        with pytest.raises(ValueError):
            result.groups[0].input_position = 4
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")
    assert db.logical_state() == before
    assert files() == artifacts


def test_lexical_never_opens_vectors_and_empty_candidates_are_valid(tmp_path, monkeypatch):
    from academic_chatbot.evidence import resolver

    db = database(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("lexical request touched vector state")

    monkeypatch.setattr(resolver, "_validate_vector_generation", forbidden)
    result = EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
    assert result.groups
    empty = EvidenceReadResolver(data_root=db.paths.data_root).resolve(
        db.request.model_copy(update={"candidates": ()})
    )
    assert empty.groups == () and empty.coverage.packed_page_ids == ()


def test_wal_snapshot_survives_concurrent_publication(tmp_path, monkeypatch):
    from academic_chatbot.documents.import_service import document_generation_id_for
    from academic_chatbot.evidence import resolver

    db = database(tmp_path)
    assert db.rows("PRAGMA journal_mode = WAL")[0][0] == "wal"
    new_id = document_generation_id_for(
        file_version_id=db.request.scope.file_version_id,
        processing_profile_id="synthetic-new-pipeline",
    )
    db.execute(
        "INSERT INTO document_generations VALUES (?, ?, ?, ?)",
        (new_id, db.request.scope.file_version_id, "synthetic-new-pipeline", "2026-09-06"),
    )
    original = resolver._resolve_scope

    def change(connection, scope):
        # Establish snapshot with a source SELECT, then publish through another connection.
        connection.execute("SELECT project_id FROM projects").fetchone()
        db.execute("UPDATE generation_publications SET document_generation_id = ?", (new_id,))
        return original(connection, scope)

    monkeypatch.setattr(resolver, "_resolve_scope", change)
    result = EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
    assert result.resolved_scope.document_generation_id == db.request.scope.document_generation_id
    monkeypatch.setattr(resolver, "_resolve_scope", original)
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
    assert error.value.error.code == "GENERATION_NOT_CURRENT"


@pytest.mark.parametrize(
    "stage", ["_resolve_scope", "_read_coverage", "_admit_pages", "_read_page", "_resolve_group"]
)
def test_cleanup_failure_does_not_hide_primary(tmp_path, monkeypatch, stage):
    import sqlite3

    from academic_chatbot.evidence import resolver

    db = database(tmp_path)
    original = resolver.open_read_only_connection
    connections = []

    class Proxy:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, *args):
            return self.conn.execute(*args)

        def rollback(self):
            self.conn.rollback()
            raise OSError("private cleanup detail")

        def close(self):
            self.conn.close()
            raise OSError("private close detail")

    def capture(*args, **kwargs):
        conn = original(*args, **kwargs)
        connections.append(conn)
        return Proxy(conn)

    def fail(*args, **kwargs):
        raise EvidenceResolutionError(resolver.m.EvidenceErrorCode.ANCHOR_MISMATCH)

    monkeypatch.setattr(resolver, "open_read_only_connection", capture)
    monkeypatch.setattr(resolver, stage, fail)
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
    assert error.value.error.code == "ANCHOR_MISMATCH"
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")


@pytest.mark.parametrize("target", ["root", "project", "database"])
def test_unsafe_reparse_path_rejected_before_open(tmp_path, monkeypatch, target):
    import stat
    from types import SimpleNamespace

    from academic_chatbot.evidence import resolver

    db = database(tmp_path)
    selected = {
        "root": db.paths.data_root,
        "project": db.paths.project_root,
        "database": db.paths.database_path,
    }[target]
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        if path == selected:
            return SimpleNamespace(
                st_mode=info.st_mode, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT
            )
        return info

    monkeypatch.setattr(Path, "lstat", lstat)

    def forbidden(*args, **kwargs):
        pytest.fail("unsafe DB opened")

    monkeypatch.setattr(resolver, "open_read_only_connection", forbidden)
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
    assert error.value.error.code == "UNSAFE_PATH"


def test_known_gaps_take_precedence_without_hiding_unknown_metadata(tmp_path):
    db = database(tmp_path)
    db.execute(
        "INSERT INTO pages (page_id, document_generation_id, page_number) VALUES (?, ?, ?)",
        ("legacy", db.request.scope.document_generation_id, 2),
    )
    db.execute(
        "UPDATE pages SET extraction_quality = 'low_native_text', needs_ocr = 1 "
        "WHERE page_number = 1"
    )
    result = EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
    assert result.coverage.status == "known_native_gaps"
    assert result.coverage.pages[1].extraction_category == "unknown"
    assert result.coverage.pages[1].physical_page_index is None
    assert result.coverage.document_completeness == "not_assessed"
    assert result.coverage.quality_basis == "native_text_presence_heuristic"
    assert result.coverage.packed_page_ids == ()


def test_missing_database_is_not_initialized(tmp_path):
    root = tmp_path / "missing-data"
    with pytest.raises(EvidenceResolutionError) as error:
        db = database(tmp_path)
        EvidenceReadResolver(data_root=root).resolve(db.request)
    assert error.value.error.code == "STORAGE_UNAVAILABLE"
    assert not root.exists()


def test_anchor_order_assertion_is_exact(tmp_path):
    db = database(tmp_path)
    ref = db.request.candidates[0]
    ref = ref.model_copy(
        update={
            "lexical": ref.lexical.model_copy(
                update={"expected_anchor_ids": tuple(reversed(ref.lexical.expected_anchor_ids))}
            )
        }
    )
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(
            db.request.model_copy(update={"candidates": (ref,)})
        )
    assert error.value.error.code == "ANCHOR_MISMATCH"
