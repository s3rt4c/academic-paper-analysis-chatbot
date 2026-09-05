from __future__ import annotations

import hashlib
import json
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from academic_chatbot.cli import main
from academic_chatbot.documents.import_service import DocumentImportService
from academic_chatbot.documents.native_pdf import NativePdfParser
from academic_chatbot.domain.library import Project
from academic_chatbot.embeddings.profile import approved_bge_small_en_v15_profile
from academic_chatbot.embeddings.repository import EmbeddingRepository
from academic_chatbot.library.service import LibraryService
from academic_chatbot.retrieval.semantic import SemanticRetrievalService
from academic_chatbot.storage.paths import ProjectPaths
from tests.integration.embeddings.test_vector_publication import _builder, _Embedder, _profile

_PDF = Path(__file__).parents[1] / "fixtures" / "pdfs" / "native_anchor.pdf"


@dataclass
class _QueryEmbedder:
    profile: object
    vector: np.ndarray

    def embed_queries(self, texts: tuple[str, ...]) -> np.ndarray:
        return self.vector[np.newaxis, :].astype(np.float32, copy=True)


def test_native_pdf_to_semantic_hit_preserves_active_evidence(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    library = LibraryService(data_root=data_root, max_pdf_bytes=1_000_000)
    project = library.create_project(display_name="Native", project_id="project-one")
    library.create_paper(project_id=project.project_id, paper_id="paper-one")
    file_version = library.admit_pdf(
        project_id=project.project_id, paper_id="paper-one", source_path=_PDF
    )
    paths = ProjectPaths.create(data_root, project_id=project.project_id)
    parsed = NativePdfParser(paths).parse(file_version)
    DocumentImportService(library.repository_for_project_id(project.project_id)).publish(parsed)
    profile = _profile()
    repository = EmbeddingRepository(paths)
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
    document_embedder = _Embedder(profile)
    built = _builder(repository, paths, document_embedder).build(project_id=project.project_id)
    query_vector = document_embedder.embed_documents((document_embedder.calls[0][0],))[0]
    results = SemanticRetrievalService(
        data_root=data_root,
        profile=profile,
        embedder=_QueryEmbedder(profile, query_vector),
    ).search(Project(project_id=project.project_id, display_name="Native"), "semantic native")

    assert built.generation.vector_generation_id == results.hits[0].vector_generation_id
    assert results.hits[0].file_version_id == file_version.file_version_id
    assert results.hits[0].paper_id == "paper-one"
    assert results.hits[0].anchors


@dataclass(frozen=True)
class _OfflineFake:
    profile: object

    @property
    def _tokenizer(self) -> _TokenizerFake:
        return _TokenizerFake(self.profile)

    @classmethod
    def open(cls, _model_root: Path, profile: object) -> _OfflineFake:
        return cls(profile)

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        return _vectors(texts, dimension=self.profile.dimension)  # type: ignore[attr-defined]

    def embed_queries(self, texts: tuple[str, ...]) -> np.ndarray:
        return _vectors(texts, dimension=self.profile.dimension)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class _TokenizerFake:
    profile: object

    def prepare(self, _role: object, _texts: tuple[str, ...]) -> object:
        return object()


class _ManifestFake:
    def canonical_payload(self) -> dict[str, object]:
        return {"manifest": "offline-fake-v1"}


def _vectors(texts: tuple[str, ...], *, dimension: int) -> np.ndarray:
    rows = np.empty((len(texts), dimension), dtype=np.float32)
    for index, text in enumerate(texts):
        raw = hashlib.sha256(text.encode("utf-8")).digest()
        rows[index] = np.frombuffer((raw * 12)[:dimension], dtype=np.uint8).astype(np.float32)
    rows += np.float32(1.0)
    rows /= np.sqrt(np.sum(rows * rows, axis=1, keepdims=True, dtype=np.float32))
    return rows


def test_cli_native_pdf_build_then_semantic_search_is_offline_and_evidence_bearing(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("normal semantic application flow must remain offline and in-process")

    for name in ("connect", "connect_ex", "send", "sendto"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)

    def artifacts(_root: Path, *, profile: object) -> object:
        assert profile == approved_bge_small_en_v15_profile()
        return SimpleNamespace(manifest=_ManifestFake())

    monkeypatch.setattr("academic_chatbot.cli.load_verified_artifacts", artifacts)
    monkeypatch.setattr("academic_chatbot.cli.OfflineEmbedder", _OfflineFake)
    monkeypatch.setattr("academic_chatbot.retrieval.semantic.load_verified_artifacts", artifacts)
    monkeypatch.setattr("academic_chatbot.retrieval.semantic.OfflineEmbedder", _OfflineFake)
    root = ["--data-root", str(tmp_path / "data"), "--max-pdf-bytes", "1000000"]
    profile = approved_bge_small_en_v15_profile()

    assert (
        main([*root, "project", "create", "--project-id", "p", "--display-name", "Research"]) == 0
    )
    assert main([*root, "paper", "create", "--project-id", "p", "--paper-id", "paper"]) == 0
    assert (
        main(
            [*root, "import-pdf", "--project-id", "p", "--paper-id", "paper", "--source", str(_PDF)]
        )
        == 0
    )
    capsys.readouterr()
    build = [
        *root,
        "semantic-index",
        "build",
        "--project-id",
        "p",
        "--embedding-profile-id",
        profile.embedding_profile_id,
        "--model-root",
        str(tmp_path / "models"),
    ]
    assert main(build) == 0
    build_payload = json.loads(capsys.readouterr().out)
    assert build_payload["active"] is True
    assert build_payload["current"] is True
    assert build_payload["empty"] is False
    assert build_payload["coverage"]["embeddable_spans"] > 0
    assert main(build) == 0
    reused_payload = json.loads(capsys.readouterr().out)
    assert reused_payload["reused"] is True
    assert reused_payload["vector_generation_id"] == build_payload["vector_generation_id"]

    assert main([*root, "search", "--project-id", "p", "--query", "accuracy"]) == 0
    lexical = json.loads(capsys.readouterr().out)
    assert lexical["hits"]
    assert (
        main(
            [
                *root,
                "search",
                "--mode",
                "semantic",
                "--project-id",
                "p",
                "--query",
                "accuracy",
                "--embedding-profile-id",
                profile.embedding_profile_id,
                "--model-root",
                str(tmp_path / "models"),
            ]
        )
        == 0
    )
    semantic = json.loads(capsys.readouterr().out)
    assert semantic["vector_generation_id"] == build_payload["vector_generation_id"]
    assert semantic["hits"][0]["paper_id"] == "paper"
    assert semantic["hits"][0]["anchors"]
    assert semantic["hits"][0]["raw_semantic_score"] > 0.0

    hybrid_arguments = [
        *root,
        "search",
        "--mode",
        "hybrid",
        "--project-id",
        "p",
        "--query",
        "accuracy",
        "--embedding-profile-id",
        profile.embedding_profile_id,
        "--model-root",
        str(tmp_path / "models"),
    ]
    assert main(hybrid_arguments) == 0
    hybrid = json.loads(capsys.readouterr().out)
    assert hybrid["mode"] == "hybrid"
    assert hybrid["fusion_profile_id"] == "rrf-v1"
    assert hybrid["lexical_state"] == "healthy_results"
    assert hybrid["semantic_state"] == "healthy_results"
    assert hybrid["hits"]
    hit = hybrid["hits"][0]
    assert hit["trace"]["channel_membership"] == "both"
    assert hit["lexical_contribution"]["lexical_hit"]["anchors"]
    assert hit["semantic_contribution"]["semantic_hit"]["anchors"]

    assert main(hybrid_arguments) == 0
    assert json.loads(capsys.readouterr().out) == hybrid
