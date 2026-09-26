from __future__ import annotations

import json
from hashlib import sha256

import pytest
from pydantic import TypeAdapter, ValidationError

from academic_chatbot.analysis import supported_objective_verifier as verifier
from academic_chatbot.analysis.models import (
    GroupedAnalysisEvidenceUnit,
    StandaloneAnalysisEvidenceUnit,
    SupportedProposal,
)
from academic_chatbot.evidence.models import (
    MAX_AUDIT_BYTES,
    CandidateDisposition,
    EvidenceBundleEntry,
    EvidenceBundlePayload,
    EvidenceBundleRequest,
)
from academic_chatbot.evidence.serialization import seal_bundle
from tests.fixtures.evidence_bundle.contracts import (
    payload_data,
    request_data,
    resolved_range_data,
)


class _UnusedResolver:
    def resolve(self, request):
        raise AssertionError("authority must not be called")


class _AuthorityVerifier(verifier.SupportedObjectiveCitationVerifier):
    def __init__(self, outcome=None, error: Exception | None = None) -> None:
        super().__init__(resolver=_UnusedResolver())
        self.calls = []
        self._outcome = outcome
        self._error = error

    def _revalidate_authority(self, bundle, bound_units):
        self.calls.append((bundle, bound_units))
        if self._error is not None:
            raise self._error
        return self._outcome


def _empty_bundle():
    payload = EvidenceBundlePayload.model_validate_json(json.dumps(payload_data()))
    return seal_bundle(payload)


def _proposal(bundle_fingerprint: str) -> SupportedProposal:
    return SupportedProposal(
        value="Evaluate alpha.",
        evidence_units=(
            StandaloneAnalysisEvidenceUnit(
                bundle_fingerprint=bundle_fingerprint,
                citation_labels=("E1",),
            ),
        ),
    )


def _bundle(
    texts: tuple[str, ...] = ("Evaluate alpha.",),
    *,
    entry_labels: tuple[str, ...] | None = None,
    disposition_labels: tuple[str, ...] | None = None,
):
    labels = entry_labels or tuple(f"E{index}" for index in range(1, len(texts) + 1))
    payload = EvidenceBundlePayload.model_validate_json(json.dumps(payload_data()))
    entries = []
    for label, value in zip(labels, texts, strict=True):
        entry = resolved_range_data()
        entry["text"] = value
        entry["source"]["text_sha256"] = sha256(value.encode("utf-8")).hexdigest()
        entry["citation_label"] = label
        entries.append(EvidenceBundleEntry.model_validate(entry))
    cited = disposition_labels if disposition_labels is not None else labels
    reference = EvidenceBundleRequest.model_validate_json(
        json.dumps(request_data())
    ).candidates[0]
    disposition = CandidateDisposition(
        input_position=0,
        reference=reference,
        citation_labels=cited,
    )
    payload = payload.model_copy(
        update={"entries": tuple(entries), "dispositions": (disposition,)}
    )
    return seal_bundle(payload)


def _verification_request(
    bundle,
    *,
    value: str = "Evaluate alpha.",
    units=None,
):
    return verifier.SupportedObjectiveVerificationRequest(
        policy_id=verifier.SUPPORTED_OBJECTIVE_VERIFIER_POLICY_ID,
        proposal=SupportedProposal(
            value=value,
            evidence_units=units
            or (
                StandaloneAnalysisEvidenceUnit(
                    bundle_fingerprint=bundle.fingerprint,
                    citation_labels=tuple(entry.citation_label for entry in bundle.entries),
                ),
            ),
        ),
        bundle=bundle,
    )


def _request(*, policy_id: str = "supported-objective-exact-text-v1"):
    bundle = _empty_bundle()
    return verifier.SupportedObjectiveVerificationRequest(
        policy_id=policy_id,
        proposal=_proposal(bundle.fingerprint),
        bundle=bundle,
    )


def test_frozen_policy_and_request_contract() -> None:
    assert verifier.SUPPORTED_OBJECTIVE_VERIFIER_POLICY_ID == (
        "supported-objective-exact-text-v1"
    )
    request = _request()
    assert request.schema_version == "supported-objective-verification-request-v1"

    with pytest.raises(ValidationError):
        request.policy_id = "other-policy-v1"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        verifier.SupportedObjectiveVerificationRequest.model_validate(
            {**request.model_dump(mode="python"), "extra": True}
        )
    with pytest.raises(ValidationError):
        verifier.SupportedObjectiveVerificationRequest.model_validate(
            {**request.model_dump(mode="python"), "policy_id": 1}
        )
    with pytest.raises(ValidationError):
        verifier.SupportedObjectiveVerificationRequest.model_validate(
            {**request.model_dump(mode="python"), "policy_id": "bad\ud800"}
        )


def test_result_contracts_are_strict_frozen_and_discriminated() -> None:
    fingerprint = sha256(b"proposal").hexdigest()
    bundle_fingerprint = sha256(b"bundle").hexdigest()
    verified = verifier.SupportedObjectiveVerified(
        proposal_fingerprint=fingerprint,
        bundle_fingerprint=bundle_fingerprint,
    )
    abstained = verifier.SupportedObjectiveAbstained(
        proposal_fingerprint=fingerprint,
        bundle_fingerprint=bundle_fingerprint,
        reason=verifier.SupportedObjectiveAbstentionReason.VALUE_NOT_EXACT_SOURCE_TEXT,
    )

    assert verified.schema_version == "supported-objective-verification-result-v1"
    assert set(verified.model_dump()) == {
        "schema_version",
        "outcome",
        "policy_id",
        "concern_profile_id",
        "proposal_fingerprint",
        "bundle_fingerprint",
    }
    assert set(abstained.model_dump()) == {
        "schema_version",
        "outcome",
        "policy_id",
        "concern_profile_id",
        "proposal_fingerprint",
        "bundle_fingerprint",
        "reason",
    }
    adapter = TypeAdapter(verifier.SupportedObjectiveVerificationResult)
    assert adapter.validate_python(verified).outcome == "verified"
    assert adapter.validate_python(abstained).outcome == "abstained"
    with pytest.raises(ValidationError):
        verified.outcome = "abstained"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        verifier.SupportedObjectiveVerified.model_validate(
            {**verified.model_dump(), "proposal_fingerprint": 1}
        )


@pytest.mark.parametrize(
    ("result", "fields", "message"),
    [
        (
            verifier.SupportedObjectiveRejected(
                code=verifier.SupportedObjectiveRejectionCode.INVALID_REQUEST
            ),
            {"schema_version", "outcome", "code", "message"},
            "Verification request is invalid",
        ),
        (
            verifier.SupportedObjectiveFailed(
                code=verifier.SupportedObjectiveFailureCode.INTERNAL_INVARIANT_FAILURE
            ),
            {"schema_version", "outcome", "code", "message"},
            "Verification failed safely",
        ),
    ],
)
def test_safe_results_have_only_fixed_fields(result, fields, message) -> None:
    dumped = result.model_dump()
    assert set(dumped) == fields
    assert dumped["message"] == message
    serialized = result.model_dump_json()
    for secret in (
        "Evaluate alpha.",
        "proposal value",
        "C:\\\\private\\\\paper.pdf",
        "nested database exception",
    ):
        assert secret not in serialized
    with pytest.raises(ValidationError):
        type(result)(code=result.code, message="nested database exception")


def test_policy_mismatch_and_malformed_request_are_safely_rejected() -> None:
    service = verifier.SupportedObjectiveCitationVerifier(resolver=_UnusedResolver())

    policy = service.verify(_request(policy_id="other-policy-v1"))
    malformed = service.verify({"proposal": "private proposal value"})

    assert policy.outcome == "rejected"
    assert policy.code == verifier.SupportedObjectiveRejectionCode.POLICY_MISMATCH
    assert malformed.outcome == "rejected"
    assert malformed.code == verifier.SupportedObjectiveRejectionCode.INVALID_REQUEST
    assert "private proposal value" not in malformed.model_dump_json()


def test_exact_utf8_standalone_text_verifies_after_authority() -> None:
    bundle = _bundle()
    service = _AuthorityVerifier()

    result = service.verify(_verification_request(bundle))

    assert result.outcome == "verified"
    assert result.bundle_fingerprint == bundle.fingerprint
    assert len(service.calls) == 1


@pytest.mark.parametrize(
    "value",
    [
        "evaluate alpha.",
        " Evaluate alpha.",
        "Evaluate alpha. ",
        "Evaluate\r\nalpha.",
        "Evaluate álpha.",
        "Evaluate a\u0301lpha.",
        "Evaluate alpha!",
        "Evaluate alpha",
        "Evaluate alpha. Additional text.",
        "Do not evaluate alpha.",
        "Evaluate alpha only.",
        "Evaluate alpha then beta.",
        "Evaluate 1 alpha.",
        "Evaluate +1 alpha.",
        "Evaluate 1.0 alpha.",
        "Evaluate 100% alpha.",
        "Evaluate 1/1 alpha.",
        "Evaluate one alpha.",
    ],
)
def test_every_textual_difference_abstains_without_normalization(value: str) -> None:
    bundle = _bundle()
    service = _AuthorityVerifier()

    result = service.verify(_verification_request(bundle, value=value))

    assert result.outcome == "abstained"
    assert result.reason == (
        verifier.SupportedObjectiveAbstentionReason.VALUE_NOT_EXACT_SOURCE_TEXT
    )
    assert len(service.calls) == 1


def test_every_standalone_citation_must_independently_match() -> None:
    matching = _bundle(("Evaluate alpha.", "Evaluate alpha."))
    differing = _bundle(("Evaluate alpha.", "Different text."))

    accepted = _AuthorityVerifier().verify(_verification_request(matching))
    abstained = _AuthorityVerifier().verify(_verification_request(differing))

    assert accepted.outcome == "verified"
    assert abstained.outcome == "abstained"
    assert abstained.reason == "value_not_exact_source_text"


def test_proposal_fingerprint_binds_every_proposal_field_deterministically() -> None:
    bundle = _bundle()
    proposal = _verification_request(bundle).proposal
    changed = proposal.model_copy(update={"value": "Evaluate beta."})

    assert verifier._proposal_fingerprint(proposal) == verifier._proposal_fingerprint(proposal)
    assert verifier._proposal_fingerprint(proposal) != verifier._proposal_fingerprint(changed)


@pytest.mark.parametrize(
    "bundle_factory,units_factory",
    [
        (
            lambda: _bundle(entry_labels=("E2",)),
            lambda bundle: (
                StandaloneAnalysisEvidenceUnit(
                    bundle_fingerprint=bundle.fingerprint, citation_labels=("E1",)
                ),
            ),
        ),
        (
            lambda: _bundle(
                ("Evaluate alpha.", "Evaluate alpha."),
                entry_labels=("E1", "E1"),
                disposition_labels=("E1",),
            ),
            lambda bundle: (
                StandaloneAnalysisEvidenceUnit(
                    bundle_fingerprint=bundle.fingerprint, citation_labels=("E1",)
                ),
            ),
        ),
        (
            lambda: _bundle(),
            lambda bundle: (
                StandaloneAnalysisEvidenceUnit(
                    bundle_fingerprint=bundle.fingerprint,
                    citation_labels=("E1", "E1"),
                ),
            ),
        ),
        (
            lambda: _bundle(),
            lambda bundle: (
                StandaloneAnalysisEvidenceUnit(
                    bundle_fingerprint=bundle.fingerprint, citation_labels=("E1",)
                ),
                StandaloneAnalysisEvidenceUnit(
                    bundle_fingerprint=bundle.fingerprint, citation_labels=("E1",)
                ),
            ),
        ),
        (
            lambda: _bundle(),
            lambda bundle: (
                StandaloneAnalysisEvidenceUnit(
                    bundle_fingerprint="f" * 64, citation_labels=("E1",)
                ),
            ),
        ),
        (
            lambda: _bundle(),
            lambda bundle: (
                GroupedAnalysisEvidenceUnit(
                    bundle_fingerprint=bundle.fingerprint,
                    group_id="group-1",
                    citation_labels=("E1", "E2"),
                ),
            ),
        ),
    ],
)
def test_invalid_or_duplicate_plain_references_reject_before_authority(
    bundle_factory, units_factory
) -> None:
    bundle = bundle_factory()
    service = _AuthorityVerifier()
    request = _verification_request(bundle, units=units_factory(bundle))

    result = service.verify(request)

    assert result.outcome == "rejected"
    assert result.code == verifier.SupportedObjectiveRejectionCode.EVIDENCE_REFERENCE_INVALID
    assert service.calls == []


def test_invalid_bundle_fingerprint_rejects_before_authority() -> None:
    bundle = _bundle().model_copy(update={"fingerprint": "f" * 64})
    service = _AuthorityVerifier()

    result = service.verify(_verification_request(bundle))

    assert result.outcome == "rejected"
    assert result.code == verifier.SupportedObjectiveRejectionCode.BUNDLE_FINGERPRINT_INVALID
    assert service.calls == []


def test_authority_result_precedes_direct_support_and_unexpected_errors_are_safe() -> None:
    bundle = _bundle()
    rejected = verifier.SupportedObjectiveRejected(
        code=verifier.SupportedObjectiveRejectionCode.SOURCE_NOT_CURRENT
    )
    blocked = _AuthorityVerifier(outcome=rejected).verify(
        _verification_request(bundle, value="Different text.")
    )
    failed = _AuthorityVerifier(
        error=RuntimeError("C:\\private\\paper.pdf nested database exception")
    ).verify(_verification_request(bundle))

    assert blocked == rejected
    assert failed.outcome == "failed"
    assert failed.code == verifier.SupportedObjectiveFailureCode.INTERNAL_INVARIANT_FAILURE
    assert "private" not in failed.model_dump_json()
    assert "nested" not in failed.model_dump_json()


def test_oversized_proposal_rejects_before_authority() -> None:
    bundle = _bundle()
    service = _AuthorityVerifier()
    request = _verification_request(bundle, value="x" * MAX_AUDIT_BYTES)

    result = service.verify(request)

    assert result.outcome == "rejected"
    assert result.code == verifier.SupportedObjectiveRejectionCode.RESOURCE_LIMIT
    assert service.calls == []
