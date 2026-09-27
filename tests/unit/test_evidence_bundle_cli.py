"""Exact binary transport and isolated preview dispatch tests."""

import io
import json
from pathlib import Path

import pytest

from academic_chatbot import cli
from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.resolver import EvidenceResolutionError
from academic_chatbot.evidence.serialization import canonical_bundle_bytes, canonical_json_bytes
from tests.unit.evidence.test_packing import group, resolved
from tests.unit.evidence.test_service import build

ROOT = ["--data-root", "unused", "--max-pdf-bytes", "1000"]
COMMAND = ["evidence-bundle", "preview", "--request-stdin"]


class BinaryOnly:
    def __init__(self, data=b""):
        self.buffer = io.BytesIO(data)

    def write(self, text):
        pytest.fail("preview used text-mode output")

    def read(self, *args):
        pytest.fail("preview used text-mode input")


def setup(
    monkeypatch, *, text="alpha", status=m.CoverageStatus.NO_FLAGGED_NATIVE_GAPS, empty=False
):
    data = resolved(*(() if empty else (group(text),)), status=status)
    bundle = build(data)
    calls = []

    class Resolver:
        def __init__(self, *, data_root):
            calls.append(("resolver", data_root))

        def resolve(self, request):
            pytest.fail("CLI resolved independently of service")

    class Service:
        def __init__(self, *, resolver):
            assert isinstance(resolver, Resolver)

        def build(self, request):
            calls.append(("build", request))
            return bundle

    streams = [BinaryOnly(data.request.model_dump_json().encode()), BinaryOnly(), BinaryOnly()]
    for name, stream in zip(("stdin", "stdout", "stderr"), streams, strict=True):
        monkeypatch.setattr(cli.sys, name, stream)
    monkeypatch.setattr(cli, "EvidenceReadResolver", Resolver)
    monkeypatch.setattr(cli, "EvidenceBundleService", Service)
    return data, bundle, streams, calls


def assert_error(streams, code):
    assert streams[1].buffer.getvalue() == b""
    raw = streams[2].buffer.getvalue()
    payload = json.loads(raw)
    assert set(payload) == {"code", "message", "input_position"}
    assert payload["code"] == code
    assert raw == canonical_json_bytes(payload) + b"\n"
    assert b"private-secret" not in raw


def test_preview_dispatch_never_constructs_library_service(monkeypatch):
    data, _, _, calls = setup(monkeypatch)
    from academic_chatbot.db.migrations import MigrationRunner

    def forbidden(*args, **kwargs):
        pytest.fail("preview initialized write-capable storage")

    for cls in (
        cli.LibraryService,
        cli.ProjectRepository,
        cli.EmbeddingRepository,
        MigrationRunner,
    ):
        monkeypatch.setattr(cls, "__init__", forbidden)
    assert cli.main([*ROOT, *COMMAND]) == 0
    assert calls == [("resolver", Path("unused")), ("build", data.request)]


@pytest.mark.parametrize("text", ["ASCII", "Türkçe", "漢字😀", '"quote"', "back\\slash"])
@pytest.mark.parametrize(
    "status", [m.CoverageStatus.NO_FLAGGED_NATIVE_GAPS, m.CoverageStatus.KNOWN_NATIVE_GAPS]
)
def test_preview_success_writes_exact_utf8_json_and_lf(monkeypatch, text, status):
    _, bundle, streams, calls = setup(monkeypatch, text=text, status=status)
    before = bundle.model_dump_json()
    assert cli.main([*ROOT, *COMMAND]) == 0
    assert streams[1].buffer.getvalue() == canonical_bundle_bytes(bundle) + b"\n"
    assert streams[2].buffer.getvalue() == b""
    assert bundle.model_dump_json() == before
    assert len(calls) == 2


def test_preview_insufficient_returns_three(monkeypatch):
    _, bundle, streams, _ = setup(monkeypatch, empty=True)
    assert cli.main([*ROOT, *COMMAND]) == 3
    assert streams[1].buffer.getvalue() == canonical_bundle_bytes(bundle) + b"\n"
    assert streams[2].buffer.getvalue() == b""


def test_preview_failure_has_empty_stdout(monkeypatch):
    _, _, streams, _ = setup(monkeypatch)

    class Failed:
        def __init__(self, **kwargs):
            pass

        def build(self, request):
            raise EvidenceResolutionError(m.EvidenceErrorCode.GENERATION_NOT_CURRENT, 2)

    monkeypatch.setattr(cli, "EvidenceBundleService", Failed)
    assert cli.main([*ROOT, *COMMAND]) == 2
    assert_error(streams, "GENERATION_NOT_CURRENT")
    assert json.loads(streams[2].buffer.getvalue())["input_position"] == 2


@pytest.mark.parametrize(
    "argv",
    [
        [*ROOT, "evidence-bundle", "preview"],
        [*ROOT, *COMMAND, "--unknown", "private-secret"],
        ["--data-root", "private-secret", *COMMAND],
        ["--max-pdf-bytes", "1000", *COMMAND],
        ["--data-root", "unused", "--max-pdf-bytes", "private-secret", *COMMAND],
        ["--data-root", "unused", "--max-pdf-bytes", "0", *COMMAND],
        [*ROOT, "evidence-bundle", "private-secret"],
        [*ROOT, "evidence-bundle"],
    ],
)
def test_preview_argument_error_is_safe_structured_json(monkeypatch, argv):
    _, _, streams, calls = setup(monkeypatch)
    assert cli.main(argv) == 2
    assert_error(streams, "INVALID_REQUEST")
    assert calls == []


@pytest.mark.parametrize("value", ["evidence-bundle", "preview"])
@pytest.mark.parametrize("equals", [False, True])
def test_option_value_is_not_misidentified_as_command(monkeypatch, capsys, value, equals):
    calls = []

    class Library:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(cli, "LibraryService", Library)
    monkeypatch.setattr(cli, "_dispatch", lambda *args: {"legacy": True})
    root = [f"--data-root={value}"] if equals else ["--data-root", value]
    assert (
        cli.main(
            [*root, "--max-pdf-bytes", "1000", "search", "--project-id", "p", "--query", "preview"]
        )
        == 0
    )
    assert calls[0]["data_root"] == Path(value)
    assert json.loads(capsys.readouterr().out) == {"legacy": True}


@pytest.mark.parametrize(
    "argv", [["evidence-bundle", "--help"], ["evidence-bundle", "preview", "--help"], ["--help"]]
)
def test_help_is_normal_and_never_constructs_storage(monkeypatch, capsys, argv):
    def forbidden(*args, **kwargs):
        pytest.fail("help initialized storage")

    monkeypatch.setattr(cli, "LibraryService", forbidden)
    assert cli.main(argv) == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out and captured.err == ""


@pytest.mark.parametrize(
    "raw", [b"", b"private-secret", b"\xff", b'{"private-secret":NaN}', b"[]", b"{} {}"]
)
def test_input_failures_are_safe_before_service_construction(monkeypatch, raw):
    _, _, streams, calls = setup(monkeypatch)
    streams[0].buffer = io.BytesIO(raw)
    assert cli.main([*ROOT, *COMMAND]) == 2
    assert_error(streams, "INVALID_REQUEST")
    assert calls == []


def test_unexpected_internal_exception_is_safe(monkeypatch):
    _, _, streams, _ = setup(monkeypatch)

    def fail(**kwargs):
        raise RuntimeError("private-secret")

    monkeypatch.setattr(cli, "EvidenceBundleService", fail)
    assert cli.main([*ROOT, *COMMAND]) == 2
    assert_error(streams, "STORAGE_INTEGRITY")


def test_missing_storage_is_read_only_and_creates_nothing(monkeypatch, tmp_path):
    from academic_chatbot.evidence.service import EvidenceBundleService

    _, _, streams, _ = setup(monkeypatch, empty=True)
    monkeypatch.setattr(
        cli,
        "EvidenceReadResolver",
        __import__(
            "academic_chatbot.evidence.resolver", fromlist=["EvidenceReadResolver"]
        ).EvidenceReadResolver,
    )
    monkeypatch.setattr(cli, "EvidenceBundleService", EvidenceBundleService)
    missing = tmp_path / "absent"
    assert cli.main(["--data-root", str(missing), "--max-pdf-bytes", "1000", *COMMAND]) == 2
    assert_error(streams, "STORAGE_UNAVAILABLE")
    assert not missing.exists()


@pytest.mark.parametrize(
    "raw", [b" " * (m.MAX_REQUEST_BYTES + 1), b"[" * 17], ids=["bytes", "depth"]
)
def test_resource_limits_fail_before_service(monkeypatch, raw):
    _, _, streams, calls = setup(monkeypatch)
    streams[0].buffer = io.BytesIO(raw)
    assert cli.main([*ROOT, *COMMAND]) == 2
    assert_error(streams, "RESOURCE_LIMIT")
    assert calls == []


@pytest.mark.parametrize(
    "code", [m.EvidenceErrorCode.RESOURCE_LIMIT, m.EvidenceErrorCode.STORAGE_INTEGRITY]
)
def test_preparation_error_preserves_safe_code(monkeypatch, code):
    _, _, streams, _ = setup(monkeypatch)

    def fail(bundle):
        raise cli.EvidencePreparationError(code, 1)

    monkeypatch.setattr(cli, "canonical_bundle_bytes", fail)
    assert cli.main([*ROOT, *COMMAND]) == 2
    assert_error(streams, code.value)
    assert json.loads(streams[2].buffer.getvalue())["input_position"] == 1


@pytest.mark.parametrize("flag", ["--data-root", "--max-pdf-bytes"])
def test_malformed_trailing_global_option_keeps_preview_json_errors(monkeypatch, flag):
    _, _, streams, calls = setup(monkeypatch)
    assert cli.main([*ROOT, *COMMAND, flag]) == 2
    assert_error(streams, "INVALID_REQUEST")
    assert calls == []
