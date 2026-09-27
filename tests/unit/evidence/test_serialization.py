"""Independent canonical byte/hash oracles; no bundle sealing here."""

from hashlib import sha256
from importlib import import_module

import pytest

from academic_chatbot.evidence import models as m
from tests.fixtures.evidence_bundle.contracts import payload_data


@pytest.mark.parametrize(
    "value,expected,digest",
    [
        ({}, b"{}", "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"),
        (
            {"b": [2, 1], "a": None},
            b'{"a":null,"b":[2,1]}',
            "64befe554fb7858d5aedd620f047ac02bca15dfe6f2f13dc790b1d18d532ef8c",
        ),
        (
            {"evidence": [], "context": [], "concern_profile_id": "stated-study-objective-v1"},
            b'{"concern_profile_id":"stated-study-objective-v1","context":[],"evidence":[]}',
            "a741f66f6c5920488b51ee561db23ade275b51c5c938b6e2d7e0cab737101ed6",
        ),
    ],
)
def test_canonical_json_exact_bytes(value, expected, digest):
    encode = import_module("academic_chatbot.evidence.serialization").canonical_json_bytes
    assert encode(value) == expected
    assert sha256(encode(value)).hexdigest() == digest
    assert not encode(value).endswith(b"\n")
    assert encode(value) == encode(value)


def test_canonical_json_preserves_unicode_and_array_order():
    encode = import_module("academic_chatbot.evidence.serialization").canonical_json_bytes
    assert encode({"é": ["e\u0301", "é", '\n"']}) == '{"é":["é","é","\\n\\""]}'.encode()
    assert encode([1, 2]) != encode([2, 1])


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), {"bad": "\ud800"}, {1: "non-string-key"}, object()],
)
def test_noncanonical_values_rejected(value):
    encode = import_module("academic_chatbot.evidence.serialization").canonical_json_bytes
    with pytest.raises((ValueError, TypeError, UnicodeError)):
        encode(value)


def _payload():
    data = payload_data()
    data["coverage"]["status"] = m.CoverageStatus(data["coverage"]["status"])
    data["state"] = m.PreviewState(data["state"])
    data["insufficient_reason"] = m.InsufficientReason(data["insufficient_reason"])
    return m.EvidenceBundlePayload.model_validate(data)


def _oracle(value):
    import json

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def test_payload_fingerprint_independent_oracle_and_sealing():
    s = import_module("academic_chatbot.evidence.serialization")
    payload = _payload()
    expected = _oracle(payload.model_dump(mode="json"))
    assert s.canonical_payload_bytes(payload) == expected
    assert s.compute_bundle_fingerprint(payload) == sha256(expected).hexdigest()
    bundle = s.seal_bundle(payload)
    assert bundle.fingerprint == sha256(expected).hexdigest()
    assert s.canonical_bundle_bytes(bundle) == _oracle(bundle.model_dump(mode="json"))
    assert s.canonical_bundle_bytes(bundle) == s.canonical_bundle_bytes(s.seal_bundle(payload))
    assert not s.canonical_bundle_bytes(bundle).endswith(b"\n")
    assert b'"generation_enabled":false' in expected
    assert b'"generation_token_count":null' in expected
    assert b'"fingerprint"' not in expected


def test_sealing_and_final_validation_hash_only_explicit_payload(monkeypatch):
    s = import_module("academic_chatbot.evidence.serialization")
    compute = s.compute_bundle_fingerprint
    seen = []

    def checked_compute(payload):
        assert type(payload) is m.EvidenceBundlePayload
        assert "fingerprint" not in type(payload).model_fields
        seen.append(payload)
        return compute(payload)

    monkeypatch.setattr(s, "compute_bundle_fingerprint", checked_compute)
    payload = _payload()
    bundle = s.seal_bundle(payload)
    assert s.canonical_bundle_bytes(bundle) == _oracle(bundle.model_dump(mode="json"))
    assert len(seen) == 2
    assert seen[0] == seen[1] == payload


def test_fingerprint_covers_omissions_coverage_and_budget():
    s = import_module("academic_chatbot.evidence.serialization")
    original = _payload()
    variants = [
        original.model_copy(
            update={
                "coverage": original.coverage.model_copy(update={"flagged_page_ids": ("page-a",)})
            }
        ),
        original.model_copy(
            update={"budget": original.budget.model_copy(update={"used_content_bytes": 78})}
        ),
    ]
    from tests.fixtures.evidence_bundle.contracts import request_data

    reference = m.EvidenceBundleRequest.model_validate(request_data()).candidates[0]
    for reason in (m.OmissionReason.PREVIEW_BYTES_EXCEEDED, m.OmissionReason.ENTRY_LIMIT_EXCEEDED):
        variants.append(
            original.model_copy(
                update={
                    "dispositions": (
                        m.CandidateDisposition(
                            input_position=0,
                            reference=reference,
                            omission=m.EvidenceOmission(reason=reason),
                        ),
                    )
                }
            )
        )
    digests = [s.compute_bundle_fingerprint(value) for value in [original, *variants]]
    assert len(set(digests)) == len(digests)
    for value, digest in zip([original, *variants], digests, strict=True):
        assert digest == sha256(_oracle(value.model_dump(mode="json"))).hexdigest()


def test_fingerprint_preserves_array_order():
    s = import_module("academic_chatbot.evidence.serialization")
    original = _payload()
    variants = [
        original.model_copy(
            update={"coverage": original.coverage.model_copy(update={"candidate_page_ids": pages})}
        )
        for pages in (("page-a", "page-b"), ("page-b", "page-a"))
    ]
    assert s.compute_bundle_fingerprint(variants[0]) != s.compute_bundle_fingerprint(variants[1])


def test_final_serialization_rejects_invalid_fingerprint():
    s = import_module("academic_chatbot.evidence.serialization")
    invalid = s.seal_bundle(_payload()).model_copy(update={"fingerprint": "0" * 64})
    with pytest.raises(s.EvidencePreparationError) as caught:
        s.canonical_bundle_bytes(invalid)
    assert caught.value.error.code == m.EvidenceErrorCode.STORAGE_INTEGRITY
    assert "0" * 64 not in str(caught.value)


@pytest.mark.parametrize("text", ["ASCII", "Türkçe 😀", 'quote" slash\\\n\t\u0001'])
def test_audit_cap_includes_terminal_lf_exact_fit_and_one_over(monkeypatch, text):
    s = import_module("academic_chatbot.evidence.serialization")
    payload = _payload()
    preview = m.ContentPreview(
        context=(),
        evidence=(m.ContentEvidence(citation_label="E1", physical_page_index=0, text=text),),
    )
    payload = payload.model_copy(update={"content_preview": preview})
    bundle = s.seal_bundle(payload)
    expected = _oracle(bundle.model_dump(mode="json"))
    monkeypatch.setattr(m, "MAX_AUDIT_BYTES", len(expected) + 1)
    assert s.canonical_bundle_bytes(bundle) == expected
    monkeypatch.setattr(m, "MAX_AUDIT_BYTES", len(expected))
    with pytest.raises(s.EvidencePreparationError) as caught:
        s.canonical_bundle_bytes(bundle)
    assert caught.value.error.code == m.EvidenceErrorCode.RESOURCE_LIMIT


def test_real_audit_cap_rejects_single_oversized_string_without_unbounded_encoder(monkeypatch):
    s = import_module("academic_chatbot.evidence.serialization")
    payload = _payload().model_copy(
        update={
            "content_preview": m.ContentPreview(
                context=(),
                evidence=(
                    m.ContentEvidence(
                        citation_label="E1", physical_page_index=0, text="😀" * m.MAX_AUDIT_BYTES
                    ),
                ),
            )
        }
    )

    def forbidden(*args, **kwargs):
        pytest.fail("unbounded canonical byte encoder used")

    monkeypatch.setattr(s, "canonical_json_bytes", forbidden)
    with pytest.raises(s.EvidencePreparationError) as caught:
        s.seal_bundle(payload)
    assert caught.value.error.code == m.EvidenceErrorCode.RESOURCE_LIMIT


def test_preparation_error_uses_fixed_structured_message():
    s = import_module("academic_chatbot.evidence.serialization")
    error = s.EvidencePreparationError(m.EvidenceErrorCode.INVALID_ORDER, input_position=3)
    assert isinstance(error, ValueError)
    assert error.error.input_position == 3
    assert str(error) == error.error.message == "Candidate order is invalid"


def test_real_four_mib_boundary_includes_lf():
    s = import_module("academic_chatbot.evidence.serialization")

    def with_text(text):
        return _payload().model_copy(
            update={
                "content_preview": m.ContentPreview(
                    context=(),
                    evidence=(
                        m.ContentEvidence(citation_label="E1", physical_page_index=0, text=text),
                    ),
                )
            }
        )

    empty = s.seal_bundle(with_text(""))
    overhead = len(_oracle(empty.model_dump(mode="json")))
    text_length = m.MAX_AUDIT_BYTES - 1 - overhead
    exact = s.seal_bundle(with_text("x" * text_length))
    assert len(s.canonical_bundle_bytes(exact)) + 1 == 4 * 1024 * 1024
    overflow = s.seal_bundle(with_text("x" * (text_length + 1)))
    with pytest.raises(s.EvidencePreparationError) as caught:
        s.canonical_bundle_bytes(overflow)
    assert caught.value.error.code == m.EvidenceErrorCode.RESOURCE_LIMIT


def test_large_encoder_token_is_encoded_in_bounded_fragments_and_stops(monkeypatch):
    s = import_module("academic_chatbot.evidence.serialization")
    encoded_lengths = []

    class ObservedToken(str):
        def __getitem__(self, key):
            return ObservedToken(super().__getitem__(key))

        def encode(self, *args, **kwargs):
            encoded_lengths.append(len(self))
            return super().encode(*args, **kwargs)

    def tokens(self, value):
        yield ObservedToken("😀" * (2 * m.MAX_AUDIT_BYTES))
        pytest.fail("encoder continued after rejected oversized token")

    monkeypatch.setattr(s.json.JSONEncoder, "iterencode", tokens)
    with pytest.raises(s.EvidencePreparationError):
        s.canonical_payload_bytes(_payload())
    assert encoded_lengths
    assert max(encoded_lengths) <= 4096
    assert sum(encoded_lengths) * 4 <= m.MAX_AUDIT_BYTES + 16384
