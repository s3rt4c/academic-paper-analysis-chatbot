from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from academic_chatbot.cli import main
from academic_chatbot.embeddings.artifacts import EmbeddingArtifactError
from academic_chatbot.embeddings.profile import approved_bge_small_en_v15_profile
from academic_chatbot.embeddings.repository import VectorGenerationCoverage
from academic_chatbot.retrieval.semantic import SemanticRetrievalResults

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "pdfs" / "native_anchor.pdf"


def test_cli_orchestrates_project_paper_import_and_search_as_json(tmp_path: Path, capsys) -> None:
    """Would fail if the module CLI duplicated or omitted the approved vertical slice."""
    root = ["--data-root", str(tmp_path / "data"), "--max-pdf-bytes", "1000000"]
    assert (
        main([*root, "project", "create", "--project-id", "p", "--display-name", "Research"]) == 0
    )
    assert main([*root, "paper", "create", "--project-id", "p", "--paper-id", "paper"]) == 0
    assert (
        main(
            [
                *root,
                "import-pdf",
                "--project-id",
                "p",
                "--paper-id",
                "paper",
                "--source",
                str(_FIXTURE),
            ]
        )
        == 0
    )
    assert main([*root, "search", "--project-id", "p", "--query", "control"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["hits"]


def test_cli_returns_clean_error_for_argument_errors(capsys) -> None:
    """Would fail if an ordinary CLI mistake escaped as argparse SystemExit."""
    assert main([]) == 2
    captured = capsys.readouterr()
    assert "usage:" in captured.err
    assert "Traceback" not in captured.err


def test_cli_returns_clean_error_for_nonlexical_search(tmp_path: Path, capsys) -> None:
    """Would fail if a user query error emitted a traceback or appeared successful."""
    assert (
        main(
            [
                "--data-root",
                str(tmp_path / "data"),
                "--max-pdf-bytes",
                "1000000",
                "search",
                "--project-id",
                "p",
                "--query",
                "()",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "meaningful lexical" in captured.err
    assert "Traceback" not in captured.err


def test_cli_semantic_mode_uses_the_verified_model_root_and_emits_mode_json(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    profile_id = "ep-sha256-" + "a" * 64

    class _SemanticService:
        @classmethod
        def open_from_model_root(cls, **kwargs: object) -> _SemanticService:
            assert kwargs["profile_id"] == profile_id
            assert kwargs["model_root"] == tmp_path / "models"
            return cls()

        def search(self, project: object, query: str, limit: int) -> SemanticRetrievalResults:
            assert query == "meaningful"
            assert limit == 3
            return SemanticRetrievalResults(
                project_id="p",
                query=query,
                embedding_profile_id=profile_id,
                vector_generation_id="vector-generation-sha256-" + "b" * 64,
                hits=(),
            )

    monkeypatch.setattr("academic_chatbot.cli.SemanticRetrievalService", _SemanticService)
    assert (
        main(
            [
                "--data-root",
                str(tmp_path / "data"),
                "--max-pdf-bytes",
                "1000000",
                "search",
                "--mode",
                "semantic",
                "--project-id",
                "p",
                "--query",
                "meaningful",
                "--limit",
                "3",
                "--embedding-profile-id",
                profile_id,
                "--model-root",
                str(tmp_path / "models"),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "semantic"
    assert payload["embedding_profile_id"] == profile_id


def test_cli_semantic_mode_requires_a_profile_and_model_root(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "--data-root",
                str(tmp_path / "data"),
                "--max-pdf-bytes",
                "1000000",
                "search",
                "--mode",
                "semantic",
                "--project-id",
                "p",
                "--query",
                "meaningful",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "requires --embedding-profile-id and --model-root" in captured.err
    assert "Traceback" not in captured.err


def test_cli_exposes_semantic_index_build_arguments(tmp_path: Path, capsys) -> None:
    profile_id = "ep-sha256-3f8fd2dbcff088eb61b2ef1ecbc6de57644a425722a586fef32059516146a929"

    assert (
        main(
            [
                "--data-root",
                str(tmp_path / "data"),
                "--max-pdf-bytes",
                "1000000",
                "semantic-index",
                "build",
                "--project-id",
                "p",
                "--embedding-profile-id",
                profile_id,
                "--model-root",
                str(tmp_path / "models"),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "project does not exist" in captured.err
    assert "Traceback" not in captured.err


def test_cli_semantic_index_build_composes_verified_boundaries_as_json(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    profile = approved_bge_small_en_v15_profile()
    root = ["--data-root", str(tmp_path / "data"), "--max-pdf-bytes", "1000000"]
    assert (
        main([*root, "project", "create", "--project-id", "p", "--display-name", "Research"]) == 0
    )
    capsys.readouterr()
    calls: dict[str, object] = {}

    class _Manifest:
        def canonical_payload(self) -> dict[str, object]:
            return {"manifest": "verified"}

    def load_artifacts(model_root: Path, *, profile: object) -> object:
        calls["model_root"] = model_root
        calls["profile"] = profile
        return SimpleNamespace(manifest=_Manifest())

    class _Embedder:
        def __init__(self) -> None:
            self.profile = profile
            self._tokenizer = SimpleNamespace(profile=profile)

        @classmethod
        def open(cls, model_root: Path, opened_profile: object) -> _Embedder:
            assert model_root == tmp_path / "models"
            assert opened_profile == profile
            return cls()

    class _Repository:
        def __init__(self, _paths: object) -> None:
            self.generation: object | None = None

        def register_profile(self, registered: object, *, artifact_manifest_sha256: str) -> None:
            assert registered == profile
            assert len(artifact_manifest_sha256) == 64
            calls["registered"] = True

        def active_generation(self, *, project_id: str, embedding_profile_id: str) -> object:
            assert project_id == "p"
            assert embedding_profile_id == profile.embedding_profile_id
            return self.generation

    @dataclass(frozen=True)
    class _Generation:
        vector_generation_id: str
        project_id: str
        embedding_profile_id: str
        source_snapshot_sha256: str
        artifact_relative_dir: str
        coverage: VectorGenerationCoverage

    repository = _Repository(None)

    class _Builder:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["profile"] == profile
            assert kwargs["tokenizer"].profile == profile  # type: ignore[attr-defined]
            calls["builder"] = True

        def build(self, *, project_id: str) -> object:
            assert project_id == "p"
            generation = _Generation(
                vector_generation_id="vector-generation-sha256-" + "a" * 64,
                project_id="p",
                embedding_profile_id=profile.embedding_profile_id,
                source_snapshot_sha256="b" * 64,
                artifact_relative_dir="indexes/semantic/current",
                coverage=VectorGenerationCoverage(1, 1, 0, 0, 1, 0),
            )
            repository.generation = generation
            return SimpleNamespace(generation=generation, empty=False, reused=False)

    monkeypatch.setattr("academic_chatbot.cli.load_verified_artifacts", load_artifacts)
    monkeypatch.setattr("academic_chatbot.cli.OfflineEmbedder", _Embedder)
    monkeypatch.setattr("academic_chatbot.cli.EmbeddingRepository", lambda _paths: repository)
    monkeypatch.setattr("academic_chatbot.cli.ProjectVectorBuilder", _Builder)

    assert (
        main(
            [
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
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert calls == {
        "model_root": tmp_path / "models",
        "profile": profile,
        "registered": True,
        "builder": True,
    }
    assert payload == {
        "active": True,
        "artifact_relative_path": "indexes/semantic/current",
        "coverage": {
            "eligible_native_chunks": 1,
            "embeddable_spans": 1,
            "excluded_unembeddable_spans": 0,
            "needs_ocr_pages": 0,
            "indexed_documents": 1,
            "unindexed_documents": 0,
        },
        "current": True,
        "embedding_profile_id": profile.embedding_profile_id,
        "empty": False,
        "project_id": "p",
        "reused": False,
        "source_snapshot_sha256": "b" * 64,
        "vector_generation_id": "vector-generation-sha256-" + "a" * 64,
    }


def test_cli_semantic_index_build_reports_artifact_failure_without_traceback(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    profile = approved_bge_small_en_v15_profile()
    root = ["--data-root", str(tmp_path / "data"), "--max-pdf-bytes", "1000000"]
    assert (
        main([*root, "project", "create", "--project-id", "p", "--display-name", "Research"]) == 0
    )
    capsys.readouterr()

    def mismatch(_root: Path, *, profile: object) -> object:
        raise EmbeddingArtifactError("onnx/model.onnx SHA-256 does not match the manifest")

    monkeypatch.setattr("academic_chatbot.cli.load_verified_artifacts", mismatch)
    assert (
        main(
            [
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
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "embedding artifact verification failed" in captured.err
    assert "Traceback" not in captured.err
