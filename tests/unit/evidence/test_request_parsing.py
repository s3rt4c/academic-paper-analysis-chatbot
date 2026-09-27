"""Bounded transport admission before strict request-model validation."""

import io
import json
from unittest.mock import Mock

import pytest

from academic_chatbot.evidence import models as m
from academic_chatbot.evidence import serialization as s
from tests.fixtures.evidence_bundle.contracts import request_data


def encoded_request():
    return json.dumps(request_data()).encode("utf-8")


def assert_failure(raw, code=m.EvidenceErrorCode.INVALID_REQUEST):
    with pytest.raises(s.EvidencePreparationError) as caught:
        s.parse_bundle_request(raw)
    assert caught.value.error == m.EvidenceBundleError(code=code)
    assert str(caught.value) == caught.value.error.message


class RecordingStream(io.BytesIO):
    def __init__(self, raw, short_reads=False):
        super().__init__(raw)
        self.sizes = []
        self.short_reads = short_reads

    def read(self, size=-1):
        assert 0 < size <= 65536
        self.sizes.append(size)
        return super().read(min(size, 7) if self.short_reads else size)


@pytest.mark.parametrize("excess", [0, 1, 100000])
def test_stdin_actual_read_is_bounded(excess, monkeypatch):
    validate = Mock(side_effect=AssertionError("model construction during read"))
    monkeypatch.setattr(m.EvidenceBundleRequest, "model_validate_json", validate)
    stream = RecordingStream(b" " * (m.MAX_REQUEST_BYTES + excess))
    if excess:
        with pytest.raises(s.EvidencePreparationError) as caught:
            s.read_request_bytes(stream)
        assert caught.value.error.code == m.EvidenceErrorCode.RESOURCE_LIMIT
        assert stream.tell() == m.MAX_REQUEST_BYTES + 1
    else:
        assert s.read_request_bytes(stream) == b" " * m.MAX_REQUEST_BYTES
    assert len(stream.sizes) > 1
    assert sum(stream.sizes) <= m.MAX_REQUEST_BYTES + 1
    validate.assert_not_called()


def test_reader_accumulates_short_reads_until_eof():
    raw = encoded_request()
    assert s.read_request_bytes(RecordingStream(raw, short_reads=True)) == raw


def test_exact_request_byte_limit_is_admitted():
    raw = encoded_request()
    padded = raw + b" " * (m.MAX_REQUEST_BYTES - len(raw))
    assert s.parse_bundle_request(padded) == m.EvidenceBundleRequest.model_validate(request_data())


def test_parse_overflow_precedes_decode_and_model(monkeypatch):
    decode = Mock(side_effect=AssertionError("JSON decoded before size admission"))
    validate = Mock(side_effect=AssertionError("model constructed before size admission"))
    monkeypatch.setattr(s.json, "loads", decode)
    monkeypatch.setattr(m.EvidenceBundleRequest, "model_validate_json", validate)
    assert_failure(b" " * (m.MAX_REQUEST_BYTES + 1), m.EvidenceErrorCode.RESOURCE_LIMIT)
    decode.assert_not_called()
    validate.assert_not_called()


@pytest.mark.parametrize("container", ["array", "object"])
@pytest.mark.parametrize("depth", [16, 17])
def test_depth_is_checked_before_json_decode(depth, container, monkeypatch):
    raw = (
        "[" * depth + "0" + "]" * depth
        if container == "array"
        else '{"x":' * depth + "0" + "}" * depth
    ).encode()
    decode = Mock(wraps=s.json.loads)
    monkeypatch.setattr(s.json, "loads", decode)
    code = (
        m.EvidenceErrorCode.RESOURCE_LIMIT if depth == 17 else m.EvidenceErrorCode.INVALID_REQUEST
    )
    assert_failure(raw, code)
    assert decode.call_count == (0 if depth == 17 else 1)


@pytest.mark.parametrize("value", ['[{"' * 30, 'x\\"' + "[" * 30, "x\\\\" + "{" * 30])
def test_depth_scan_respects_strings_quotes_and_backslashes(value):
    data = request_data()
    # Valid identifiers can contain quotes/brackets but cannot contain backslashes.
    raw = json.dumps({**data, "unknown": value}).encode()
    assert_failure(raw)


@pytest.mark.parametrize("raw", [
    b'{"scope":{},"scope":{}}',
    b'{"scope":{"project_id":"a","project_id":"b"}}',
    b'{"candidates":[{"parent":{"x":1,"x":2}}]}',
    b'{"scope":{"x":1,"\\u0078":2}}',
])
def test_duplicate_keys_are_rejected(raw, monkeypatch):
    validate = Mock(side_effect=AssertionError("duplicate reached model"))
    monkeypatch.setattr(m.EvidenceBundleRequest, "model_validate_json", validate)
    assert_failure(raw)
    validate.assert_not_called()


@pytest.mark.parametrize("literal", [b"NaN", b"Infinity", b"-Infinity", b"1e999"])
def test_nonfinite_values_are_rejected_before_model(literal, monkeypatch):
    validate = Mock(side_effect=AssertionError("nonfinite reached model"))
    monkeypatch.setattr(m.EvidenceBundleRequest, "model_validate_json", validate)
    assert_failure(b'{"unknown":' + literal + b"}")
    validate.assert_not_called()


@pytest.mark.parametrize("raw", [
    b"", b" \r\n\t ", b"[]", b"null", b"true", b"123", b"{} {}",
    b'{"x":}', b'{"x":1,}', b'{"x":"unterminated}', b"}" + b"[" * 17,
])
def test_malformed_or_nonobject_json_is_safe(raw):
    assert_failure(raw)


@pytest.mark.parametrize("raw", [
    b"\xff", b"\xc0\xaf", b"\xed\xa0\x80", b'{"x":"\\ud800"}',
    b'{"x":"\\udfff"}', b'{"\\ud800":0}',
])
def test_invalid_utf8_and_surrogate_escape_are_rejected(raw, monkeypatch):
    validate = Mock(side_effect=AssertionError("invalid Unicode reached model"))
    monkeypatch.setattr(m.EvidenceBundleRequest, "model_validate_json", validate)
    assert_failure(raw)
    validate.assert_not_called()


@pytest.mark.parametrize("ascii_escapes", [False, True])
def test_valid_unicode_and_json_arrays_use_strict_json_model_path(ascii_escapes):
    data = request_data()
    data["scope"]["paper_id"] = 'Çal\u0131şma-研究-😀-[{"quoted"}]'
    raw = json.dumps(data, ensure_ascii=ascii_escapes).encode()
    request = s.parse_bundle_request(raw)
    assert request.scope.paper_id == data["scope"]["paper_id"]
    assert isinstance(request.candidates, tuple)
    assert isinstance(request.candidates[0].lexical.expected_anchor_ids, tuple)


@pytest.mark.parametrize("value", [True, False, "1", 1.0, None])
@pytest.mark.parametrize("field", ["reported_rank", "start_offset", "max_entries"])
def test_strict_field_coercion_is_rejected(field, value):
    data = request_data()
    target = data["candidates"][0]
    if field == "start_offset":
        target = target["lexical"]
    elif field == "max_entries":
        target = data["preview_budget"]
    target[field] = value
    assert_failure(json.dumps(data).encode())


def test_model_rejection_does_not_expose_caller_values():
    data = request_data()
    data["private-C:\\source"] = "SECRET-PASSAGE-" + "a" * 64
    assert_failure(json.dumps(data).encode())
