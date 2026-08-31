from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

from academic_chatbot.cli import main

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "pdfs" / "native_anchor.pdf"


def _invoke(capsys, arguments: list[str]) -> dict[str, object]:
    assert main(arguments) == 0
    return json.loads(capsys.readouterr().out)


def test_normal_native_document_cli_flow_uses_no_network_or_subprocess(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """Guard the normal local application journey against network and child processes."""
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("normal Phase 1A document flow must remain local and in-process")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket.socket, "send", forbidden)
    monkeypatch.setattr(socket.socket, "sendto", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "call", forbidden)
    monkeypatch.setattr(subprocess, "check_call", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    data_root = tmp_path / "data"
    root = ["--data-root", str(data_root), "--max-pdf-bytes", "1000000"]

    _invoke(
        capsys,
        [*root, "project", "create", "--project-id", "project-1", "--display-name", "Research"],
    )
    _invoke(
        capsys,
        [*root, "paper", "create", "--project-id", "project-1", "--paper-id", "paper-1"],
    )
    _invoke(
        capsys,
        [
            *root,
            "import-pdf",
            "--project-id",
            "project-1",
            "--paper-id",
            "paper-1",
            "--source",
            str(_FIXTURE),
        ],
    )
    result = _invoke(capsys, [*root, "search", "--project-id", "project-1", "--query", "control"])

    assert result["hits"]
