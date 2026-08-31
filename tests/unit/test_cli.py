from __future__ import annotations

import json
from pathlib import Path

from academic_chatbot.cli import main

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "pdfs" / "native_anchor.pdf"


def test_cli_orchestrates_project_paper_import_and_search_as_json(tmp_path: Path, capsys) -> None:
    """Would fail if the module CLI duplicated or omitted the approved vertical slice."""
    root = ["--data-root", str(tmp_path / "data"), "--max-pdf-bytes", "1000000"]
    assert (
        main([*root, "project", "create", "--project-id", "p", "--display-name", "Research"])
        == 0
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
