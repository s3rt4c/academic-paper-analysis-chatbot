from __future__ import annotations

import json
from pathlib import Path

import pytest

from academic_chatbot import cli
from academic_chatbot.generation.models import (
    AnswerCitation,
    LocalGenerationAbstained,
    LocalGenerationAbstentionReason,
    LocalGenerationAnswered,
    LocalGenerationFailed,
    LocalGenerationFailureCode,
    LocalGenerationRejected,
    LocalGenerationRejectionCode,
)


def _arguments(tmp_path: Path) -> list[str]:
    absolute = tmp_path.resolve()
    return [
        "--data-root",
        str(absolute / "data"),
        "--max-pdf-bytes",
        "1000000",
        "ask",
        "--project-id",
        "project-1",
        "--paper-id",
        "paper-1",
        "--question",
        "What is the objective?",
        "--embedding-profile-id",
        "profile-1",
        "--embedding-model-root",
        str(absolute / "embedding"),
        "--runtime-dir",
        str(absolute / "runtime"),
        "--runtime-manifest",
        str(absolute / "runtime.json"),
        "--model",
        str(absolute / "model.gguf"),
        "--model-manifest",
        str(absolute / "model.json"),
    ]


class _Service:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def ask(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _answered() -> LocalGenerationAnswered:
    return LocalGenerationAnswered(
        answer="The objective is synthetic.",
        citation_labels=("E1",),
        citations=(
            AnswerCitation(
                citation_label="E1",
                paper_id="paper-1",
                file_version_id="file-1",
                document_generation_id="generation-1",
                physical_page_index=0,
                display_page_number=1,
                printed_page_label="1",
                page_id="page-1",
                start_offset=0,
                end_offset=9,
                text_sha256="a" * 64,
                anchor_ids=("anchor-1",),
            ),
        ),
        bundle_fingerprint="b" * 64,
        rendered_prompt_tokens=100,
        completion_tokens=20,
    )


@pytest.mark.parametrize(
    ("result", "exit_code"),
    [
        (_answered(), 0),
        (
            LocalGenerationAbstained(
                reason=LocalGenerationAbstentionReason.EVIDENCE_INSUFFICIENT
            ),
            3,
        ),
        (
            LocalGenerationRejected(code=LocalGenerationRejectionCode.INVALID_REQUEST),
            2,
        ),
        (
            LocalGenerationFailed(code=LocalGenerationFailureCode.RUNTIME_UNAVAILABLE),
            2,
        ),
    ],
)
def test_ask_prints_one_canonical_result(
    tmp_path: Path, monkeypatch, capsys, result, exit_code
) -> None:
    service = _Service(result)
    monkeypatch.setattr(cli, "_build_ask_service", lambda _arguments: service, raising=False)

    assert cli.main(_arguments(tmp_path)) == exit_code

    captured = capsys.readouterr()
    expected = json.dumps(
        result.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    assert captured.out == expected + "\n"
    assert captured.err == ""
    assert service.calls == [
        {
            "project_id": "project-1",
            "paper_id": "paper-1",
            "question": "What is the objective?",
        }
    ]


def test_answered_json_contains_only_frozen_public_projection(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        cli,
        "_build_ask_service",
        lambda _arguments: _Service(_answered()),
        raising=False,
    )

    assert cli.main(_arguments(tmp_path)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "answered"
    assert payload["citation_labels"] == ["E1"]
    assert payload["citations"][0]["physical_page_index"] == 0
    assert not {
        "source_text",
        "raw_model_output",
        "model_path",
        "runtime_path",
        "sql",
        "exception",
    }.intersection(payload)


@pytest.mark.parametrize("missing", ["--paper-id", "--question", "--model"])
def test_required_ask_arguments_fail_as_sanitized_json(
    tmp_path: Path, capsys, missing: str
) -> None:
    arguments = _arguments(tmp_path)
    index = arguments.index(missing)
    del arguments[index : index + 2]

    assert cli.main(arguments) == 2

    captured = capsys.readouterr()
    assert json.loads(captured.out) == LocalGenerationRejected(
        code=LocalGenerationRejectionCode.INVALID_REQUEST
    ).model_dump(mode="json")
    assert captured.err == ""


def test_repeated_path_argument_is_rejected(tmp_path: Path, capsys) -> None:
    arguments = _arguments(tmp_path)
    arguments.extend(["--model", str(tmp_path.resolve() / "other.gguf")])

    assert cli.main(arguments) == 2

    captured = capsys.readouterr()
    assert json.loads(captured.out)["code"] == "INVALID_REQUEST"
    assert "other.gguf" not in captured.out + captured.err


def test_builder_exception_is_sanitized_without_path_or_traceback(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    private_path = str(tmp_path.resolve() / "private" / "model.gguf")

    def fail(_arguments):
        raise OSError(private_path)

    monkeypatch.setattr(cli, "_build_ask_service", fail, raising=False)

    assert cli.main(_arguments(tmp_path)) == 2

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["outcome"] == "failed"
    assert payload["code"] == "RUNTIME_UNAVAILABLE"
    assert private_path not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err


def test_relative_external_path_is_rejected_before_builder(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    arguments = _arguments(tmp_path)
    arguments[arguments.index("--model") + 1] = "relative.gguf"
    called = False

    def build(_arguments):
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "_build_ask_service", build, raising=False)

    assert cli.main(arguments) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "INVALID_REQUEST"
    assert called is False


@pytest.mark.parametrize(
    ("option", "value"),
    [("--project-id", ""), ("--paper-id", "   "), ("--question", "")],
)
def test_blank_request_values_are_rejected_before_builder(
    tmp_path: Path, monkeypatch, capsys, option: str, value: str
) -> None:
    arguments = _arguments(tmp_path)
    arguments[arguments.index(option) + 1] = value
    called = False

    def build(_arguments):
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "_build_ask_service", build, raising=False)

    assert cli.main(arguments) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "INVALID_REQUEST"
    assert called is False
