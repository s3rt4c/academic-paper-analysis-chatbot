"""Strict value contract tests, deliberately independent of storage."""

import json
from importlib import import_module

import pytest
from pydantic import ValidationError

from tests.fixtures.evidence_bundle.contracts import (
    PROFILE,
    VECTOR,
    parent_data,
    payload_data,
    range_data,
    request_data,
    resolved_range_data,
    scope_data,
)


def models():
    return import_module("academic_chatbot.evidence.models")


@pytest.mark.parametrize("field", ["reported_rank", "start_offset", "end_offset"])
@pytest.mark.parametrize("value", [True, False, 1.0, "1"])
def test_request_rejects_bool_rank_and_offsets(field, value):
    model = models().EvidenceBundleRequest
    data = request_data()
    target = data["candidates"][0]
    (target if field == "reported_rank" else target["lexical"])[field] = value
    with pytest.raises(ValidationError):
        model.model_validate(data)


@pytest.mark.parametrize(
    "data",
    [
        dict(mode="lexical", embedding_profile_id=PROFILE),
        dict(mode="lexical", vector_generation_id=VECTOR),
        dict(mode="lexical", fusion_profile_id="rrf-v1"),
        dict(mode="semantic"),
        dict(
            mode="semantic",
            embedding_profile_id=PROFILE,
            vector_generation_id=VECTOR,
            fusion_profile_id="rrf-v1",
        ),
        dict(mode="hybrid", embedding_profile_id=PROFILE, vector_generation_id=VECTOR),
        dict(
            mode="hybrid",
            embedding_profile_id=PROFILE,
            vector_generation_id=VECTOR,
            fusion_profile_id="other",
        ),
        dict(mode="unknown"),
    ],
)
def test_origin_requires_exact_mode_fields(data):
    model = models().CandidateOrigin
    with pytest.raises(ValidationError):
        model.model_validate(data)


@pytest.mark.parametrize("mode", ["lexical", "semantic"])
def test_candidate_requires_mode_compatible_contributions(mode):
    model = models().EvidenceBundleRequest
    data = request_data()
    if mode == "lexical":
        data["candidates"][0]["semantic"] = {**range_data(), "embedding_span_id": "span-1"}
    else:
        data["origin"] = dict(mode=mode, embedding_profile_id=PROFILE, vector_generation_id=VECTOR)
    with pytest.raises(ValidationError):
        model.model_validate(data)


@pytest.mark.parametrize(
    "level,field",
    [
        ("request", "query"),
        ("candidate", "text"),
        ("range", "text"),
        ("range", "boxes"),
        ("scope", "display_name"),
        ("scope", "byte_length"),
    ],
)
def test_request_rejects_source_text_and_unknown_fields(level, field):
    model = models().EvidenceBundleRequest
    data = request_data()
    target = {
        "request": data,
        "candidate": data["candidates"][0],
        "range": data["candidates"][0]["lexical"],
        "scope": data["scope"],
    }[level]
    target[field] = "untrusted input"
    with pytest.raises(ValidationError):
        model.model_validate(data)


@pytest.mark.parametrize("field", list(scope_data()))
def test_source_scope_requires_four_assertions(field):
    model = models().BundleSourceScope
    data = scope_data()
    del data[field]
    with pytest.raises(ValidationError):
        model.model_validate(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation_enabled", True),
        ("generation_enabled", 0),
        ("generation_readiness", "ready"),
        ("support_assessment", "supported"),
        ("generation_token_count", 0),
        ("confidence", 0.9),
    ],
)
def test_output_models_cannot_enable_generation(field, value):
    model = models().EvidenceBundlePayload
    data = payload_data()
    data[field] = value
    with pytest.raises(ValidationError):
        model.model_validate_json(json.dumps(data))


def test_valid_requests_and_generation_disabled_payload():
    m = models()
    request = m.EvidenceBundleRequest.model_validate(request_data())
    assert request.scope.project_id == "project-1"
    assert request.preview_budget.max_content_bytes == 16384
    for mode in ("semantic", "hybrid"):
        data = request_data()
        data["origin"] = dict(mode=mode, embedding_profile_id=PROFILE, vector_generation_id=VECTOR)
        if mode == "hybrid":
            data["origin"]["fusion_profile_id"] = "rrf-v1"
        else:
            data["candidates"][0]["semantic"] = {**range_data(), "embedding_span_id": "span-1"}
            data["candidates"][0]["lexical"] = None
        assert m.EvidenceBundleRequest.model_validate(data).origin.mode == mode
    payload = m.EvidenceBundlePayload.model_validate_json(json.dumps(payload_data()))
    assert payload.generation_enabled is False
    assert payload.generation_token_count is None
    assert payload.generation_readiness == "not_assessed"
    assert payload.support_assessment == "not_performed"
    assert "fingerprint" not in payload.model_dump()
    assert not {"supported", "inferred", "not_reported", "confidence"} & payload.model_dump().keys()


@pytest.mark.parametrize("value", ["", " ", "x" * 513, "bad\ud800", 123, True, b"id"])
def test_identifiers_and_nested_parent_are_strict(value):
    m = models()
    for parent in (False, True):
        data = request_data()
        target = data["candidates"][0]["parent"] if parent else data["scope"]
        target["project_id"] = value
        with pytest.raises(ValidationError):
            m.EvidenceBundleRequest.model_validate(data)


@pytest.mark.parametrize("digest", ["a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 64 + "\n"])
def test_digest_format(digest):
    model = models().LexicalRangeRef
    with pytest.raises(ValidationError):
        model.model_validate({**range_data(), "expected_text_sha256": digest})


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_content_bytes", 76),
        ("max_content_bytes", 65537),
        ("max_content_bytes", True),
        ("max_entries", 0),
        ("max_entries", 201),
        ("max_entries", 8.0),
        ("profile_id", "tokens-v1"),
    ],
)
def test_preview_budget_bounds(field, value):
    model = models().PreviewBudget
    with pytest.raises(ValidationError):
        model.model_validate({field: value})


def test_collection_bounds_and_immutable_json_arrays():
    m = models()
    data = request_data()
    parsed = m.EvidenceBundleRequest.model_validate_json(json.dumps(data))
    assert isinstance(parsed.candidates, tuple)
    assert isinstance(parsed.candidates[0].lexical.expected_anchor_ids, tuple)
    with pytest.raises(ValidationError):
        parsed.scope.project_id = "changed"
    data["candidates"] = list(data["candidates"])
    with pytest.raises(ValidationError):
        m.EvidenceBundleRequest.model_validate(data)
    data["candidates"] = tuple(data["candidates"] * 101)
    with pytest.raises(ValidationError):
        m.EvidenceBundleRequest.model_validate(data)
    data["candidates"] = ()
    assert m.EvidenceBundleRequest.model_validate(data).candidates == ()
    assert m.PreviewBudget(max_content_bytes=77, max_entries=200).max_content_bytes == 77
    for anchors in ((), ("id",) * 121):
        with pytest.raises(ValidationError):
            m.LexicalRangeRef.model_validate({**range_data(), "expected_anchor_ids": anchors})


@pytest.mark.parametrize(
    "field,value",
    [("reported_rank", 0), ("reported_rank", -1), ("start_offset", -1), ("end_offset", 0)],
)
def test_rank_and_nonempty_range(field, value):
    m = models()
    data = request_data()
    target = data["candidates"][0]
    (target if field == "reported_rank" else target["lexical"])[field] = value
    with pytest.raises(ValidationError):
        m.EvidenceBundleRequest.model_validate(data)


def test_schema_concern_and_empty_contribution_rejected():
    m = models()
    for field in ("schema_version", "concern_profile_id"):
        data = request_data()
        data[field] = "unknown"
        with pytest.raises(ValidationError):
            m.EvidenceBundleRequest.model_validate(data)
    with pytest.raises(ValidationError):
        m.EvidenceCandidateRef(parent=parent_data(), reported_rank=1)


def test_exact_enums_and_safe_error_serialization():
    m = models()
    codes = """INVALID_REQUEST RESOURCE_LIMIT UNSAFE_PATH STORAGE_UNAVAILABLE STORAGE_INTEGRITY
    SOURCE_NOT_FOUND SOURCE_SCOPE_MISMATCH GENERATION_NOT_CURRENT REFERENCE_NOT_FOUND RANGE_MISMATCH
    TEXT_DIGEST_MISMATCH ANCHOR_MISMATCH PROFILE_MISMATCH VECTOR_UNAVAILABLE VECTOR_NOT_CURRENT
    VECTOR_INTEGRITY INVALID_ORDER CONTRADICTORY_DUPLICATE""".split()
    assert {e.value for e in m.EvidenceErrorCode} == set(codes)
    assert {e.value for e in m.PreviewState} == {
        "preview_ready",
        "preview_partial",
        "insufficient_evidence",
    }
    assert {e.value for e in m.OmissionReason} == {
        "duplicate_candidate",
        "preview_bytes_exceeded",
        "entry_limit_exceeded",
    }
    for code in m.EvidenceErrorCode:
        error = m.EvidenceBundleError(code=code)
        assert error.model_dump(mode="json")["input_position"] is None
        assert error.message
        with pytest.raises(ValidationError):
            m.EvidenceBundleError(code=code, message="raw source/path/sql")


@pytest.mark.parametrize("label", ["E0", "E01", "e1", "E-1", "E1\n", "E1 extra"])
def test_citation_label_syntax(label):
    model = models().ContentEvidence
    with pytest.raises(ValidationError):
        model(citation_label=label, physical_page_index=0, text="source")


def test_payload_fingerprint_and_context_contract():
    m = models()
    data = payload_data()
    for digest in ("bad", "A" * 64):
        with pytest.raises(ValidationError):
            m.EvidenceBundle.model_validate_json(json.dumps({**data, "fingerprint": digest}))
    bundle = m.EvidenceBundle.model_validate_json(json.dumps({**data, "fingerprint": "a" * 64}))
    assert bundle.fingerprint == "a" * 64
    context = m.ContentContext(
        related_citation_labels=("E1",), physical_page_index=0, text="source"
    )
    assert context.citable is False and context.trust == "untrusted_source_data"
    with pytest.raises(ValidationError):
        m.ContentContext(**{**context.model_dump(), "citation_label": "E1"})


def test_resolved_values_are_detached_immutable_and_compact():
    m = models()
    packet = m.ResolvedEvidenceRange.model_validate(resolved_range_data())
    assert packet.anchors[0].boxes[0].char_start == 0
    assert "canonical_page_text" not in packet.anchors[0].model_dump()
    assert packet.source.scope.model_dump().keys() == {*scope_data(), "source_pdf_sha256"}
    group = m.ResolvedCandidateGroup(
        input_position=0,
        reference=m.EvidenceBundleRequest.model_validate(request_data()).candidates[0],
        ranges=(packet,),
    )
    coverage = m.EvidenceCoverage.model_validate_json(json.dumps(payload_data()["coverage"]))
    resolved = m.ResolvedEvidenceInput(
        request=m.EvidenceBundleRequest.model_validate(request_data()),
        resolved_scope=packet.source.scope,
        coverage=coverage,
        groups=(group,),
    )
    assert isinstance(resolved.groups, tuple)
    with pytest.raises(ValidationError):
        resolved.groups[0].ranges[0].text = "changed"
    with pytest.raises(ValidationError):
        resolved.groups[0].ranges[0].anchors[0].boxes[0].x0 = 3.0
    for invalid in (iter([packet]), object(), {"handle": object()}):
        with pytest.raises(ValidationError):
            m.ResolvedCandidateGroup(input_position=0, reference=group.reference, ranges=invalid)
    entry = m.EvidenceBundleEntry(**packet.model_dump(), citation_label="E1")
    context = m.EvidenceContext(
        parent=packet.source.parent,
        physical_page_index=0,
        start_offset=0,
        end_offset=10,
        text=packet.text,
        text_sha256=packet.source.text_sha256,
        anchors=packet.anchors,
        related_citation_labels=("E1",),
    )
    assert type(context) is not type(entry)
    assert "citation_label" not in context.model_dump()
    assert context.citable is False


@pytest.mark.parametrize(
    "field,value", [("char_start", False), ("char_end", "5"), ("x0", True), ("x1", "10.0")]
)
def test_compact_anchor_rejects_nested_box_coercion(field, value):
    m = models()
    anchor = resolved_range_data()["anchors"][0]
    box = anchor["boxes"][0].model_dump()
    box[field] = value
    anchor["boxes"] = (box,)
    with pytest.raises(ValidationError):
        m.BundleAnchor.model_validate(anchor)


def test_nested_box_instance_cannot_bypass_validation():
    m = models()
    anchor = resolved_range_data()["anchors"][0]
    anchor["boxes"] = (anchor["boxes"][0].model_copy(update={"x0": float("nan")}),)
    with pytest.raises(ValidationError):
        m.BundleAnchor.model_validate(anchor)


def test_all_frozen_admission_constants():
    m = models()
    expected = dict(
        MAX_REQUEST_BYTES=2097152,
        MAX_JSON_DEPTH=16,
        MAX_CANDIDATE_GROUPS=100,
        MAX_CONTRIBUTIONS=2,
        MAX_IDENTIFIER_CHARS=512,
        MAX_CHUNK_ANCHORS=120,
        MAX_SOURCE_PAGES=1000,
        MAX_PAGE_TEXT_BYTES=1048576,
        MAX_RESOLVED_TEXT_BYTES=16777216,
        DEFAULT_CONTENT_BYTES=16384,
        MIN_CONTENT_BYTES=77,
        MAX_CONTENT_BYTES=65536,
        DEFAULT_ENTRIES=8,
        MAX_ENTRIES=200,
        MAX_AUDIT_BYTES=4194304,
        MAX_VECTOR_ROWS=100000,
        MAX_VECTOR_METADATA_BYTES=16777216,
        MAX_VECTOR_FILE_BYTES=134217728,
        MAX_MANIFEST_BYTES=65536,
        MAX_PROFILE_BYTES=65536,
        MAX_SOURCE_SNAPSHOT_ROWS=10000,
    )
    assert {name: getattr(m, name) for name in expected} == expected


def test_semantic_provenance_requires_exact_identity_fields():
    m = models()
    good = dict(
        channel="semantic",
        embedding_profile_id=PROFILE,
        vector_generation_id=VECTOR,
        embedding_span_id="span-1",
    )
    assert m.ContributionProvenance(**good).embedding_span_id == "span-1"
    for name in ("embedding_profile_id", "vector_generation_id", "embedding_span_id"):
        with pytest.raises(ValidationError):
            m.ContributionProvenance(**{**good, name: None})
    with pytest.raises(ValidationError):
        m.ContributionProvenance(**{**good, "channel": "lexical"})
