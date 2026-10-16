"""Strict, all-or-nothing canonical wire bytes reserve one CLI LF."""

import importlib
import json

import pytest

from academic_chatbot.retrieval.scoped_models import ScopedRetrievalError
from tests.unit.evidence.test_inspection_models import models, request, success_data


def serializer():
    name = "academic_chatbot.evidence.inspection_serialization"
    assert importlib.util.find_spec(name), "inspection serializer is absent"
    return importlib.import_module(name)


def test_canonical_request_and_envelope_preserve_nulls_unicode_and_order():
    s = serializer()
    req = request()
    assert (
        s.canonical_inspection_request_bytes(req)
        == json.dumps(
            req.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    )
    result = models().InspectionSuccess(**success_data())
    raw = s.canonical_inspection_result_bytes(result)
    assert (
        raw
        == json.dumps(
            result.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    )
    assert raw == s.canonical_inspection_result_bytes(result)
    assert "  özgün ç query  ".encode() in raw and b'"semantic_identity":null' in raw
    assert not raw.endswith(b"\n")
    failure = models().inspection_failure(ScopedRetrievalError("INVALID_REQUEST"))
    assert b'"request_fingerprint":null' in s.canonical_inspection_result_bytes(failure)


def test_lf_inclusive_output_boundary_never_emits_partial_bytes(monkeypatch):
    s = serializer()
    result = models().InspectionSuccess(**success_data())
    raw = s.canonical_inspection_result_bytes(result)
    monkeypatch.setattr(s, "MAX_RESULT_BYTES", len(raw) + 1)
    assert s.canonical_inspection_result_bytes(result) == raw
    monkeypatch.setattr(s, "MAX_RESULT_BYTES", len(raw))
    with pytest.raises(ScopedRetrievalError) as caught:
        s.canonical_inspection_result_bytes(result)
    assert caught.value.code == "RESOURCE_LIMIT"


def test_request_encoding_is_bounded_and_revalidates_constructed_values(monkeypatch):
    s = serializer()
    req = request()
    raw = s.canonical_inspection_request_bytes(req)
    monkeypatch.setattr(s, "MAX_REQUEST_BYTES", len(raw) - 1)
    with pytest.raises(ScopedRetrievalError) as caught:
        s.canonical_inspection_request_bytes(req)
    assert caught.value.code == "RESOURCE_LIMIT"
    for bad_query in ("\ud800", float("nan"), "x" * 4097):
        forged = req.model_copy(
            update={
                "retrieval_request": req.retrieval_request.model_copy(update={"query": bad_query})
            }
        )
        with pytest.raises(ScopedRetrievalError) as caught:
            s.canonical_inspection_request_bytes(forged)
        assert caught.value.code == "INVALID_REQUEST"


def test_actual_four_mib_boundary_and_large_token_fragment_encoding(monkeypatch):
    from academic_chatbot.evidence.models import CoveragePage, EvidenceBundlePayload
    from academic_chatbot.evidence.serialization import seal_bundle

    s = serializer()
    data = success_data()
    payload = EvidenceBundlePayload.model_validate(
        {name: getattr(data["bundle"], name) for name in EvidenceBundlePayload.model_fields}
    )
    page = CoveragePage(
        page_id="synthetic-page",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label="",
        extraction_category="empty",
        needs_ocr=True,
        has_chunks=False,
    )

    def result_with_label(label):
        changed = payload.model_copy(
            update={
                "coverage": payload.coverage.model_copy(
                    update={"pages": (page.model_copy(update={"printed_page_label": label}),)}
                )
            }
        )
        bundle = seal_bundle(changed)
        return models().InspectionSuccess(
            **{**data, "bundle": bundle, "bundle_fingerprint": bundle.fingerprint}
        )

    base = result_with_label("")
    fill = 4 * 1024 * 1024 - 1 - len(s.canonical_inspection_result_bytes(base))
    exact = result_with_label("x" * fill)
    oversized = result_with_label("x" * (fill + 1))
    original = json.JSONEncoder
    encoded_lengths = []

    class Token(str):
        def __getitem__(self, key):
            return Token(super().__getitem__(key))

        def encode(self, *args, **kwargs):
            encoded_lengths.append(len(self))
            assert len(self) <= 4096, "unbounded UTF-8 token encoding"
            return super().encode(*args, **kwargs)

    class Encoder(original):
        def iterencode(self, value, *args, **kwargs):
            return (Token(token) for token in super().iterencode(value, *args, **kwargs))

    monkeypatch.setattr(json, "JSONEncoder", Encoder)
    raw = s.canonical_inspection_result_bytes(exact)
    assert len(raw) + 1 == 4 * 1024 * 1024 and encoded_lengths
    with pytest.raises(ScopedRetrievalError) as caught:
        s.canonical_inspection_result_bytes(oversized)
    assert caught.value.code == "RESOURCE_LIMIT"
