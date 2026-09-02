from __future__ import annotations

import json
from pathlib import Path

from academic_chatbot.cli import main
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
