from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from academic_chatbot.cli import main
from academic_chatbot.domain.library import Project
from academic_chatbot.embeddings.artifacts import EmbeddingArtifactError
from academic_chatbot.embeddings.profile import approved_bge_small_en_v15_profile
from academic_chatbot.embeddings.repository import VectorGenerationCoverage
from academic_chatbot.retrieval.hybrid_service import HybridRetrievalIntegrityError
from academic_chatbot.retrieval.semantic import (
    SemanticArtifactIntegrityError,
    SemanticIndexStaleError,
    SemanticIndexUnavailableError,
    SemanticRetrievalResults,
)
from academic_chatbot.retrieval.service import RetrievalResults

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


@dataclass(frozen=True)
class _HybridResult:
    """Minimal Task 3 boundary double with already-validated JSON payload."""

    payload: dict[str, object]

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return self.payload


def _hybrid_payload(
    *,
    membership: str,
    lexical_state: str,
    semantic_state: str,
) -> dict[str, object]:
    identity = {
        "project_id": "p",
        "document_generation_id": "generation-1",
        "page_id": "page-1",
        "chunk_id": "chunk-1",
    }
    lexical: dict[str, object] | None = None
    semantic: dict[str, object] | None = None
    if membership in {"lexical_only", "both"}:
        lexical = {
            "lexical_hit": {
                **identity,
                "paper_id": "paper-1",
                "file_version_id": "file-1",
                "physical_page_index": 0,
                "display_page_number": 1,
                "printed_page_label": "1",
                "chunk_ordinal": 0,
                "chunk_text": "parent evidence",
                "start_offset": 0,
                "end_offset": 15,
                "rank": 2,
                "raw_bm25_score": 7.25,
                "anchors": [{"anchor_kind": "lexical"}],
            }
        }
    if membership in {"semantic_only", "both"}:
        semantic = {
            "semantic_hit": {
                **identity,
                "paper_id": "paper-1",
                "file_version_id": "file-1",
                "physical_page_index": 0,
                "display_page_number": 1,
                "printed_page_label": "1",
                "embedding_span_id": "span-1",
                "embedding_profile_id": "profile-1",
                "vector_generation_id": "vector-1",
                "start_offset": 1,
                "end_offset": 8,
                "embedding_span_text": "arent e",
                "rank": 3,
                "raw_semantic_score": 0.875,
                "anchors": [{"anchor_kind": "semantic"}],
            }
        }
    return {
        "project_id": "p",
        "query": "verbatim  query",
        "fusion_profile_id": "rrf-v1",
        "lexical_state": lexical_state,
        "semantic_state": semantic_state,
        "hits": [
            {
                "identity": identity,
                "parent_chunk": {
                    "identity": identity,
                    "paper_id": "paper-1",
                    "file_version_id": "file-1",
                    "physical_page_index": 0,
                    "display_page_number": 1,
                    "printed_page_label": "1",
                    "chunk_ordinal": 0,
                    "start_offset": 0,
                    "end_offset": 15,
                    "chunk_text": "parent evidence",
                },
                "lexical_contribution": lexical,
                "semantic_contribution": semantic,
                "trace": {
                    "fusion_profile_id": "rrf-v1",
                    "fusion_score": {"numerator": 5, "denominator": 126},
                    "fusion_rank": 1,
                    "channel_membership": membership,
                    "lexical_rank": 2 if lexical is not None else None,
                    "semantic_rank": 3 if semantic is not None else None,
                },
            }
        ],
    }


def _install_hybrid_service(
    monkeypatch,
    *,
    result: _HybridResult | None = None,
    error: Exception | None = None,
) -> dict[str, object]:
    calls: dict[str, object] = {}

    class _Service:
        @classmethod
        def open_from_model_root(cls, **kwargs: object) -> _Service:
            calls["open"] = kwargs
            if error is not None:
                raise error
            return cls()

        def search(self, project: object, query: str, limit: int) -> _HybridResult:
            calls["search"] = {"project": project, "query": query, "limit": limit}
            if error is not None:
                raise error
            assert result is not None
            return result

    monkeypatch.setattr("academic_chatbot.cli.HybridRetrievalService", _Service, raising=False)
    return calls


def _hybrid_arguments(tmp_path: Path, *, query: str = "verbatim  query") -> list[str]:
    return [
        "--data-root",
        str(tmp_path / "data"),
        "--max-pdf-bytes",
        "1000000",
        "search",
        "--mode",
        "hybrid",
        "--project-id",
        "p",
        "--query",
        query,
        "--limit",
        "17",
        "--embedding-profile-id",
        "profile-1",
        "--model-root",
        str(tmp_path / "models"),
    ]


def test_cli_hybrid_mode_opens_task3_service_forwards_inputs_and_serializes_dual_evidence(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """Would fail if CLI bypassed Task 3 or altered its accepted evidence payload."""
    expected = _hybrid_payload(
        membership="both", lexical_state="healthy_results", semantic_state="healthy_results"
    )
    calls = _install_hybrid_service(monkeypatch, result=_HybridResult(expected))

    assert main(_hybrid_arguments(tmp_path)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert calls["open"] == {
        "data_root": tmp_path / "data",
        "project_id": "p",
        "profile_id": "profile-1",
        "model_root": tmp_path / "models",
    }
    search_call = calls["search"]
    assert isinstance(search_call, dict)
    project = search_call["project"]
    assert isinstance(project, Project)
    assert project.project_id == "p"
    assert project.display_name == "retrieval"
    assert search_call["query"] == "verbatim  query"
    assert search_call["limit"] == 17
    assert payload == {"mode": "hybrid", **expected}
    hit = payload["hits"][0]
    assert hit["trace"] == {
        "channel_membership": "both",
        "fusion_profile_id": "rrf-v1",
        "fusion_rank": 1,
        "fusion_score": {"denominator": 126, "numerator": 5},
        "lexical_rank": 2,
        "semantic_rank": 3,
    }
    assert hit["lexical_contribution"]["lexical_hit"]["raw_bm25_score"] == 7.25
    assert hit["semantic_contribution"]["semantic_hit"]["raw_semantic_score"] == 0.875


def test_cli_hybrid_mode_requires_the_existing_semantic_inputs(tmp_path: Path, capsys) -> None:
    """Would fail if hybrid silently inferred a profile or model root."""
    arguments = _hybrid_arguments(tmp_path)
    arguments = arguments[: arguments.index("--embedding-profile-id")]

    assert main(arguments) == 2

    assert (
        "hybrid search requires --embedding-profile-id and --model-root"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    "membership, lexical_state, semantic_state",
    [
        ("lexical_only", "healthy_results", "healthy_empty"),
        ("semantic_only", "healthy_empty", "healthy_results"),
    ],
)
def test_cli_hybrid_mode_preserves_single_channel_result_states(
    tmp_path: Path,
    capsys,
    monkeypatch,
    membership: str,
    lexical_state: str,
    semantic_state: str,
) -> None:
    """Would fail if successful healthy-empty states became unavailable or lost evidence labels."""
    expected = _hybrid_payload(
        membership=membership, lexical_state=lexical_state, semantic_state=semantic_state
    )
    _install_hybrid_service(monkeypatch, result=_HybridResult(expected))

    assert main(_hybrid_arguments(tmp_path)) == 0

    payload = json.loads(capsys.readouterr().out)
    hit = payload["hits"][0]
    assert payload["lexical_state"] == lexical_state
    assert payload["semantic_state"] == semantic_state
    assert hit["trace"]["channel_membership"] == membership
    assert (hit["lexical_contribution"] is None) is (membership == "semantic_only")
    assert (hit["semantic_contribution"] is None) is (membership == "lexical_only")


@pytest.mark.parametrize(
    "error",
    [
        SemanticIndexUnavailableError("index unavailable"),
        SemanticIndexStaleError("index stale"),
        SemanticArtifactIntegrityError("artifact corrupt"),
        HybridRetrievalIntegrityError("parent evidence invalid"),
    ],
)
def test_cli_hybrid_mode_propagates_task3_failures_as_nonzero_errors(
    tmp_path: Path, capsys, monkeypatch, error: Exception
) -> None:
    """Would fail if a hybrid failure fell back to lexical results."""
    _install_hybrid_service(monkeypatch, error=error)

    assert main(_hybrid_arguments(tmp_path)) == 2

    captured = capsys.readouterr()
    assert str(error) in captured.err
    assert not captured.out
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("mode", [None, "lexical"])
def test_cli_lexical_modes_do_not_initialize_hybrid_runtime(
    tmp_path: Path, capsys, monkeypatch, mode: str | None
) -> None:
    """Would fail if adding hybrid made default or lexical search open a semantic model runtime."""

    class _LexicalService:
        def __init__(self, *, data_root: Path) -> None:
            assert data_root == tmp_path / "data"

        def search(self, project: object, query: str, limit: int) -> RetrievalResults:
            assert query == "lexical query"
            assert limit == 10
            return RetrievalResults(project_id="p", query=query, hits=())

    class _HybridService:
        @classmethod
        def open_from_model_root(cls, **_kwargs: object) -> _HybridService:
            raise AssertionError("lexical mode must not initialize hybrid retrieval")

    monkeypatch.setattr("academic_chatbot.cli.RetrievalService", _LexicalService)
    monkeypatch.setattr(
        "academic_chatbot.cli.HybridRetrievalService", _HybridService, raising=False
    )
    arguments = [
        "--data-root",
        str(tmp_path / "data"),
        "--max-pdf-bytes",
        "1000000",
        "search",
        "--project-id",
        "p",
        "--query",
        "lexical query",
    ]
    if mode is not None:
        arguments.extend(["--mode", mode])

    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == {
        "hits": [],
        "project_id": "p",
        "query": "lexical query",
    }


def test_cli_search_help_lists_hybrid_without_opening_retrieval_runtime(
    capsys, monkeypatch
) -> None:
    """Would fail if help hid hybrid or performed model-backed search work."""

    class _HybridService:
        @classmethod
        def open_from_model_root(cls, **_kwargs: object) -> _HybridService:
            raise AssertionError("help must not open hybrid retrieval")

    monkeypatch.setattr(
        "academic_chatbot.cli.HybridRetrievalService", _HybridService, raising=False
    )

    assert main(["search", "--help"]) == 0

    output = capsys.readouterr().out
    assert "{lexical,semantic,hybrid}" in output
