"""Preview preserves logical storage and closes handles under guarded execution."""

import hashlib
import socket
import sqlite3
import subprocess
import urllib.request

import pytest

from academic_chatbot.db.migrations import MigrationRunner
from academic_chatbot.documents.admission import PdfAdmissionService
from academic_chatbot.documents.import_service import DocumentImportService
from academic_chatbot.documents.native_pdf import NativePdfParser
from academic_chatbot.embeddings import embedder, reconciliation
from academic_chatbot.embeddings.embedder import OfflineEmbedder
from academic_chatbot.embeddings.repository import EmbeddingRepository
from academic_chatbot.embeddings.tokenizer import VerifiedTokenizerAdapter
from academic_chatbot.embeddings.vector_build import ProjectVectorBuilder
from academic_chatbot.evidence import models as m
from academic_chatbot.evidence import resolver
from academic_chatbot.evidence.resolver import EvidenceReadResolver, EvidenceResolutionError
from academic_chatbot.evidence.serialization import canonical_bundle_bytes
from academic_chatbot.evidence.service import EvidenceBundleService
from academic_chatbot.library.repository import ProjectRepository
from academic_chatbot.library.service import LibraryService
from academic_chatbot.ports.model import StructuredLocalModel
from academic_chatbot.retrieval.exact_memmap import ExactVectorStore
from academic_chatbot.retrieval.hybrid_service import HybridRetrievalService
from academic_chatbot.retrieval.semantic import SemanticRetrievalService
from academic_chatbot.retrieval.service import RetrievalService
from tests.fixtures.evidence_bundle.database import database


def install_preview_guards(monkeypatch):
    """Guard known application/runtime entry points, not a universal OS sandbox."""

    def forbidden(*args, **kwargs):
        pytest.fail("preview invoked a forbidden write, retrieval, or runtime entry point")

    for target, name in (
        (LibraryService, "__init__"),
        (MigrationRunner, "migrate_copy"),
        (MigrationRunner, "_apply_all"),
        (ProjectRepository, "_connection"),
        (ProjectRepository, "_persist_chunk_fts"),
        (EmbeddingRepository, "_connection"),
        (PdfAdmissionService, "admit"),
        (DocumentImportService, "publish"),
        (NativePdfParser, "parse"),
        (ProjectVectorBuilder, "build"),
        (OfflineEmbedder, "open"),
        (OfflineEmbedder, "embed_documents"),
        (OfflineEmbedder, "embed_queries"),
        (VerifiedTokenizerAdapter, "open"),
        (embedder, "_create_verified_cpu_session"),
        (embedder.ort, "InferenceSession"),
        (StructuredLocalModel, "generate"),
        (RetrievalService, "search"),
        (SemanticRetrievalService, "search"),
        (HybridRetrievalService, "search"),
        (socket, "create_connection"),
        (socket.socket, "connect"),
        (urllib.request, "urlopen"),
        (subprocess, "Popen"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    monkeypatch.setattr(reconciliation, "reconcile_vector_generations", forbidden)


def application_state(db):
    """Snapshot every present table (including FTS) and application-owned files."""
    logical = db.logical_state()
    fts = [
        tuple(row)
        for row in db.rows(
            "SELECT chunk_id, chunk_text FROM chunk_fts "
            "WHERE chunk_fts MATCH 'Alpha' ORDER BY chunk_id"
        )
    ]
    inventory = []
    hashes = {}
    for path in sorted(db.paths.project_root.rglob("*")):
        if path.name.endswith(("-wal", "-shm")):
            continue
        relative = path.relative_to(db.paths.project_root).as_posix()
        inventory.append((relative, path.is_dir()))
        # DB consistency is proved logically, independent of SQLite file metadata.
        if path.is_file() and path != db.paths.database_path:
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return logical, fts, inventory, hashes


def capture_resources(monkeypatch):
    connections, stores, mappings = [], [], []
    original_connect = resolver.open_read_only_connection
    original_open = ExactVectorStore.open

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM projects WHERE 0")
        connections.append(connection)
        return connection

    def opening(*args, **kwargs):
        store = original_open(*args, **kwargs)
        stores.append(store)
        mappings.append(store._vectors._mmap)
        return store

    monkeypatch.setattr(resolver, "open_read_only_connection", connect)
    monkeypatch.setattr(ExactVectorStore, "open", opening)
    return connections, stores, mappings


def assert_closed(resources):
    connections, stores, mappings = resources
    assert len(connections) == 1
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    for store in stores:
        with pytest.raises(RuntimeError, match="closed"):
            store._ensure_open()
    assert all(mapping.closed for mapping in mappings)


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_preview_success_preserves_all_storage_and_closes_resources(tmp_path, monkeypatch, mode):
    db = database(tmp_path)
    request = db.retrieved(mode=mode)
    before = application_state(db)
    assert before[1]
    install_preview_guards(monkeypatch)
    resources = capture_resources(monkeypatch)
    bundle = EvidenceBundleService(
        resolver=EvidenceReadResolver(data_root=db.paths.data_root)
    ).build(request)
    assert bundle.entries and canonical_bundle_bytes(bundle)
    assert_closed(resources)
    assert len(resources[1]) == (0 if mode == "lexical" else 1)
    assert application_state(db) == before


@pytest.mark.parametrize(
    "failure,code",
    [
        ("forged", "TEXT_DIGEST_MISMATCH"),
        ("version", "SOURCE_NOT_FOUND"),
        ("generation", "GENERATION_NOT_CURRENT"),
        ("semantic", "VECTOR_INTEGRITY"),
        ("resource", "RESOURCE_LIMIT"),
    ],
)
def test_preview_failure_preserves_storage_and_closes_resources(
    tmp_path, monkeypatch, failure, code
):
    db = database(tmp_path)
    request = db.retrieved(mode="semantic")
    if failure == "forged":
        candidate = request.candidates[0]
        candidate = candidate.model_copy(
            update={
                "semantic": candidate.semantic.model_copy(update={"expected_text_sha256": "a" * 64})
            }
        )
        request = request.model_copy(update={"candidates": (candidate,)})
    elif failure == "version":
        request = request.model_copy(
            update={"scope": request.scope.model_copy(update={"file_version_id": "absent"})}
        )
    elif failure == "generation":
        db.execute("DELETE FROM generation_publications")
    elif failure == "semantic":
        db.corrupt("DELETE FROM vector_generation_spans")
    else:
        monkeypatch.setattr(m, "MAX_PAGE_TEXT_BYTES", 1)
    before = application_state(db)
    install_preview_guards(monkeypatch)
    resources = capture_resources(monkeypatch)
    with pytest.raises(EvidenceResolutionError) as caught:
        EvidenceBundleService(resolver=EvidenceReadResolver(data_root=db.paths.data_root)).build(
            request
        )
    assert caught.value.error.code == code
    assert_closed(resources)
    assert application_state(db) == before
