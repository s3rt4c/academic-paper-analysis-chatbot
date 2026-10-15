"""Inspection contract rejects contradictory authority, without a second evidence format."""

import hashlib
import importlib
import json

import pytest
from pydantic import TypeAdapter, ValidationError

from academic_chatbot.evidence.models import EvidenceBundlePayload, PreviewBudget
from academic_chatbot.evidence.serialization import seal_bundle
from academic_chatbot.retrieval.scoped_models import ScopedRetrievalRequest, policy_for
from tests.fixtures.evidence_bundle.contracts import payload_data, scope_data


def models():
    name = "academic_chatbot.evidence.inspection_models"
    assert importlib.util.find_spec(name), "inspection contracts are absent"
    return importlib.import_module(name)


def request(**changes):
    return models().EvidenceInspectionRequest(
        retrieval_request=ScopedRetrievalRequest(
            scope=scope_data(),
            query="  özgün ç query  ",
            mode="lexical",
            limit=10,
            policy=policy_for("lexical"),
            semantic_identity=None,
        ),
        **changes,
    )


def success_data():
    req = request()
    bundle = seal_bundle(EvidenceBundlePayload.model_validate_json(json.dumps(payload_data())))
    return dict(
        outcome="insufficient",
        request_fingerprint=req.fingerprint,
        query=req.retrieval_request.query,
        source=bundle.scope,
        retrieval_identity=dict(
            retrieval_request_fingerprint=req.retrieval_request.fingerprint,
            mode="lexical",
            limit=10,
            policy=policy_for("lexical"),
            semantic_identity=None,
            channels=dict(
                lexical=dict(state="empty", requested_limit=10, returned_count=0),
                semantic=dict(state="not_run", requested_limit=None, returned_count=0),
            ),
        ),
        bundle_fingerprint=bundle.fingerprint,
        bundle=bundle,
    )


def test_request_binds_original_query_scope_and_preview_budget():
    req = request()
    assert req.preview_budget == PreviewBudget(max_entries=8, max_content_bytes=16384)
    expected = hashlib.sha256(
        json.dumps(
            req.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert req.fingerprint == expected
    assert req.fingerprint != request(preview_budget=PreviewBudget(max_entries=1)).fingerprint
    assert req.retrieval_request.query == "  özgün ç query  "
    for entries, size in ((1, 77), (200, 65536)):
        assert request(preview_budget=PreviewBudget(max_entries=entries, max_content_bytes=size))
    for changes in (
        {"unknown": True},
        {"schema_version": "other"},
        {"preview_budget": {"max_entries": True}},
        {"preview_budget": {"max_content_bytes": 65537}},
    ):
        with pytest.raises(ValidationError):
            request(**changes)


@pytest.mark.parametrize(
    "mutation", ["state", "fingerprint", "source", "query", "budget", "identity", "channels"]
)
def test_success_envelope_binds_bundle_and_complete_request(mutation):
    m = models()
    data = success_data()
    valid = TypeAdapter(m.EvidenceInspectionResult).validate_python(data)
    assert set(valid.model_dump()) == {
        "schema_version",
        "inspection_profile_id",
        "outcome",
        "request_fingerprint",
        "query",
        "source",
        "retrieval_identity",
        "bundle_fingerprint",
        "bundle",
    }
    assert valid.bundle.coverage.document_completeness == "not_assessed"
    if mutation == "state":
        data["outcome"] = "ready"
    elif mutation == "fingerprint":
        data["bundle_fingerprint"] = "0" * 64
    elif mutation == "source":
        data["source"] = data["source"].model_copy(update={"paper_id": "foreign"})
    elif mutation == "query":
        data["query"] = "changed"
    elif mutation == "budget":
        data["request_fingerprint"] = request(
            preview_budget=PreviewBudget(max_entries=1)
        ).fingerprint
    elif mutation == "identity":
        data["retrieval_identity"]["retrieval_request_fingerprint"] = "0" * 64
    else:
        data["retrieval_identity"]["channels"]["lexical"]["requested_limit"] = 50
    with pytest.raises(ValidationError):
        TypeAdapter(m.EvidenceInspectionResult).validate_python(data)


def test_error_envelope_is_fixed_safe_and_contains_no_evidence():
    m = models()
    from academic_chatbot.evidence.models import EvidenceErrorCode
    from academic_chatbot.evidence.resolver import EvidenceResolutionError
    from academic_chatbot.retrieval.scoped_models import ScopedRetrievalError

    for error, outcome, code in (
        (ScopedRetrievalError("SOURCE_NOT_CURRENT"), "rejected", "SOURCE_NOT_CURRENT"),
        (ScopedRetrievalError("VECTOR_NOT_CURRENT"), "failed", "VECTOR_NOT_CURRENT"),
        (EvidenceResolutionError(EvidenceErrorCode.ANCHOR_MISMATCH), "failed", "ANCHOR_MISMATCH"),
        (RuntimeError("SECRET C:/private/path"), "failed", "INTERNAL_INVARIANT_FAILURE"),
    ):
        result = m.inspection_failure(error)
        assert result.outcome == outcome and result.error.code == code
        assert result.request_fingerprint is None
        assert set(result.model_dump()) == {
            "schema_version",
            "inspection_profile_id",
            "outcome",
            "request_fingerprint",
            "error",
        }
        assert "SECRET" not in result.model_dump_json()
        for update in (
            {"query": "SECRET"},
            {"outcome": "supported"},
            {"error": {"code": code, "message": "SECRET"}},
        ):
            with pytest.raises(ValidationError):
                TypeAdapter(m.EvidenceInspectionResult).validate_python(
                    {**result.model_dump(), **update}
                )


def test_retrieval_channels_cannot_contradict_empty_authoritative_bundle():
    data = success_data()
    data["retrieval_identity"]["channels"]["lexical"].update(state="nonempty", returned_count=1)
    with pytest.raises(ValidationError):
        TypeAdapter(models().EvidenceInspectionResult).validate_python(data)
