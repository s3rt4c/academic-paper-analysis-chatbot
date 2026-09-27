"""Synthetic binary CLI coverage for EvidenceGroup v1."""

import io
import json
from pathlib import Path

from academic_chatbot import cli
from academic_chatbot.evidence import groups
from academic_chatbot.evidence.resolver import EvidenceReadResolver
from academic_chatbot.evidence.serialization import canonical_json_bytes
from tests.fixtures.evidence_bundle.database import database


class BinaryOnly:
    def __init__(self, raw: bytes = b"") -> None:
        self.buffer = io.BytesIO(raw)

    def read(self, *args):
        if self.buffer.getbuffer().nbytes:
            raise AssertionError("text stdin was used")
        return b""

    def write(self, value: bytes) -> int:
        return self.buffer.write(value)


def group_request(db, *, max_entries: int = 8, max_content_bytes: int = 16384):
    base = db.semantic(maximum_words=12).model_copy(
        update={
            "preview_budget": db.request.preview_budget.model_copy(
                update={"max_entries": max_entries, "max_content_bytes": max_content_bytes}
            )
        }
    )
    resolved = EvidenceReadResolver(data_root=db.paths.data_root).resolve(base)
    members = tuple(
        groups.EvidenceGroupMemberRequest(
            input_position=position,
            child_identity=groups.authoritative_child_identity(
                resolved.groups[position].ranges[0]
            ),
        )
        for position in (0, 1)
    )
    return groups.EvidenceGroupRequest(
        base_request=base,
        units=(
            groups.EvidenceGroupUnitRequest(
                grouping_policy_id="explicit-pair-v1", members=members
            ),
        ),
    )


def run_group_cli(monkeypatch, db, raw: bytes):
    stdin = BinaryOnly(raw)
    stdout = BinaryOnly()
    stderr = BinaryOnly()
    with monkeypatch.context() as patch:
        patch.setattr(cli.sys, "stdin", stdin)
        patch.setattr(cli.sys, "stdout", stdout)
        patch.setattr(cli.sys, "stderr", stderr)
        code = cli.main(
            [
                "--data-root",
                str(db.paths.data_root),
                "--max-pdf-bytes",
                "1000000",
                "evidence-group",
                "preview",
                "--request-stdin",
            ]
        )
    return code, stdout.buffer.getvalue(), stderr.buffer.getvalue()


def test_group_preview_emits_canonical_two_child_output(monkeypatch, tmp_path: Path):
    db = database(tmp_path)
    request = group_request(db)

    code, output, error = run_group_cli(monkeypatch, db, request.model_dump_json().encode())

    payload = json.loads(output)
    assert code == 0
    assert error == b""
    assert output == canonical_json_bytes(payload) + b"\n"
    assert payload["schema_version"] == "evidence-group-bundle-v1"
    assert payload["policy_id"] == "evidence-group-v1"
    assert len(payload["entries"]) == 2
    assert payload["dispositions"][0]["citation_labels"] == ["E1", "E2"]
    assert payload["logical_units"][0]["kind"] == "evidence_group"
    assert payload["budget"]["used_entries"] == 2


def test_group_preview_omits_both_children_when_one_slot_remains(monkeypatch, tmp_path: Path):
    db = database(tmp_path)
    request = group_request(db, max_entries=1)

    code, output, error = run_group_cli(monkeypatch, db, request.model_dump_json().encode())

    payload = json.loads(output)
    assert code == 3
    assert error == b""
    assert payload["entries"] == []
    assert payload["logical_units"] == []
    assert payload["dispositions"][0]["omission"]["reason"] == "entry_limit_exceeded"
    assert payload["dispositions"][0]["citation_labels"] == []


def test_group_preview_invalid_request_has_safe_structured_stderr(monkeypatch, tmp_path: Path):
    db = database(tmp_path)

    code, output, error = run_group_cli(monkeypatch, db, b'{"schema_version":"bad"}')

    payload = json.loads(error)
    assert code == 2
    assert output == b""
    assert payload["code"] == "INVALID_REQUEST"
    assert "schema_version" not in error.decode("utf-8")
    assert error == canonical_json_bytes(payload) + b"\n"
