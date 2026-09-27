import pytest
from pydantic import TypeAdapter, ValidationError

import academic_chatbot.analysis as analysis
from academic_chatbot.analysis import (
    GroupedAnalysisEvidenceUnit,
    StandaloneAnalysisEvidenceUnit,
)
from academic_chatbot.domain.enums import FinalFieldStatus


def standalone_unit() -> dict[str, object]:
    return {
        "kind": "standalone",
        "bundle_fingerprint": "a" * 64,
        "citation_labels": ("E1",),
    }


def grouped_unit() -> dict[str, object]:
    return {
        "kind": "evidence_group",
        "bundle_fingerprint": "a" * 64,
        "group_id": "sha256-" + "b" * 64,
        "citation_labels": ("E1", "E2"),
        "requires_all": True,
    }


def second_standalone_unit() -> dict[str, object]:
    return {
        "kind": "standalone",
        "bundle_fingerprint": "b" * 64,
        "citation_labels": ("E2",),
    }


def coverage_blocker() -> dict[str, object]:
    return {
        "page_id": "page-1",
        "physical_page_index": 0,
        "reason": "low",
    }


def supported_proposal() -> dict[str, object]:
    return {
        "concern_profile_id": "stated-study-objective-v1",
        "proposed_status": FinalFieldStatus.SUPPORTED,
        "value": "Evaluate the stated objective.",
        "evidence_units": (standalone_unit(),),
        "generation_enabled": False,
    }


def inferred_proposal() -> dict[str, object]:
    return {
        "concern_profile_id": "stated-study-objective-v1",
        "proposed_status": FinalFieldStatus.INFERRED,
        "value": "A bounded deterministic inference.",
        "evidence_units": (standalone_unit(), second_standalone_unit()),
        "deterministic_inference_rule_id": "objective-rule-v1",
        "generation_enabled": False,
    }


def not_reported_proposal() -> dict[str, object]:
    return {
        "concern_profile_id": "stated-study-objective-v1",
        "proposed_status": FinalFieldStatus.NOT_REPORTED,
        "value": None,
        "evidence_units": (),
        "negative_search_channels": (
            "section_headings",
            "alias_registry",
            "full_text_fts",
            "table_figure_captions",
            "metadata",
        ),
        "negative_search_policy_id": "objective-negative-search-v1",
        "coverage_proof_id": "objective-coverage-proof-v1",
        "generation_enabled": False,
    }


def conflicting_proposal() -> dict[str, object]:
    return {
        "concern_profile_id": "stated-study-objective-v1",
        "proposed_status": FinalFieldStatus.CONFLICTING,
        "value": None,
        "evidence_units": (standalone_unit(), second_standalone_unit()),
        "deterministic_conflict_rule_id": "objective-conflict-v1",
        "conflict_dimension": "incompatible objective scope",
        "generation_enabled": False,
    }


def unreadable_proposal() -> dict[str, object]:
    return {
        "concern_profile_id": "stated-study-objective-v1",
        "proposed_status": FinalFieldStatus.UNREADABLE,
        "value": None,
        "evidence_units": (),
        "coverage_blockers": (coverage_blocker(),),
        "generation_enabled": False,
    }


@pytest.mark.parametrize("labels", [("E1",), ("E1", "E2")])
def test_standalone_accepts_one_or_two_labels(labels: tuple[str, ...]) -> None:
    data = standalone_unit()
    data["citation_labels"] = labels

    unit = StandaloneAnalysisEvidenceUnit.model_validate(data)

    assert unit.citation_labels == labels
    assert unit.bundle_fingerprint == "a" * 64


@pytest.mark.parametrize(
    "change",
    [
        {"bundle_fingerprint": None},
        {"group_id": "sha256-" + "b" * 64},
        {"requires_all": True},
        {"citation_labels": ()},
        {"citation_labels": ("E1", "E2", "E3")},
    ],
)
def test_standalone_rejects_missing_scope_group_fields_and_bad_label_counts(
    change: dict[str, object],
) -> None:
    data = standalone_unit()
    data.update(change)

    with pytest.raises(ValidationError):
        StandaloneAnalysisEvidenceUnit.model_validate(data)


def test_group_requires_two_distinct_labels_group_identity_and_atomicity() -> None:
    unit = GroupedAnalysisEvidenceUnit.model_validate(grouped_unit())

    assert unit.citation_labels == ("E1", "E2")
    assert unit.group_id == "sha256-" + "b" * 64
    assert unit.requires_all is True


@pytest.mark.parametrize(
    "change",
    [
        {"citation_labels": ("E1",)},
        {"citation_labels": ("E1", "E2", "E3")},
        {"citation_labels": ("E1", "E1")},
        {"group_id": None},
        {"requires_all": False},
    ],
)
def test_group_rejects_incomplete_duplicate_or_non_atomic_shapes(
    change: dict[str, object],
) -> None:
    data = grouped_unit()
    data.update(change)

    with pytest.raises(ValidationError):
        GroupedAnalysisEvidenceUnit.model_validate(data)


@pytest.mark.parametrize("label", ["E0", "E01", "e1", "E-1", "1"])
def test_evidence_units_reject_malformed_labels(label: str) -> None:
    data = standalone_unit()
    data["citation_labels"] = (label,)

    with pytest.raises(ValidationError):
        StandaloneAnalysisEvidenceUnit.model_validate(data)


@pytest.mark.parametrize("fingerprint", ["A" * 64, "a" * 63, "a" * 65])
def test_evidence_units_reject_noncanonical_fingerprints(fingerprint: str) -> None:
    data = standalone_unit()
    data["bundle_fingerprint"] = fingerprint

    with pytest.raises(ValidationError):
        StandaloneAnalysisEvidenceUnit.model_validate(data)


def test_evidence_units_are_frozen_strict_and_extra_forbid() -> None:
    unit = StandaloneAnalysisEvidenceUnit.model_validate(standalone_unit())

    with pytest.raises(ValidationError):
        unit.bundle_fingerprint = "b" * 64

    data = standalone_unit()
    data["citation_labels"] = ["E1"]
    with pytest.raises(ValidationError):
        StandaloneAnalysisEvidenceUnit.model_validate(data)

    data = standalone_unit()
    data["source_text"] = "untrusted text"
    with pytest.raises(ValidationError):
        StandaloneAnalysisEvidenceUnit.model_validate(data)


def test_evidence_units_reject_invalid_unicode() -> None:
    data = grouped_unit()
    data["group_id"] = "\ud800"

    with pytest.raises(ValidationError):
        GroupedAnalysisEvidenceUnit.model_validate(data)


def test_supported_proposal_accepts_only_a_nonblank_value_with_evidence() -> None:
    proposal = analysis.SupportedProposal.model_validate(supported_proposal())

    assert proposal.proposed_status is FinalFieldStatus.SUPPORTED
    assert proposal.value == "Evaluate the stated objective."
    assert len(proposal.evidence_units) == 1
    assert proposal.generation_enabled is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("value", None),
        ("value", "   "),
        ("evidence_units", ()),
    ],
)
def test_supported_proposal_rejects_missing_value_or_evidence(
    field: str, value: object
) -> None:
    data = supported_proposal()
    data[field] = value

    with pytest.raises(ValidationError):
        analysis.SupportedProposal.model_validate(data)


def test_inferred_proposal_requires_distinct_units_and_a_rule() -> None:
    proposal = analysis.InferredProposal.model_validate(inferred_proposal())

    assert proposal.proposed_status is FinalFieldStatus.INFERRED
    assert len(proposal.evidence_units) == 2
    assert proposal.deterministic_inference_rule_id == "objective-rule-v1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("evidence_units", (standalone_unit(),)),
        ("evidence_units", (standalone_unit(), standalone_unit())),
        ("deterministic_inference_rule_id", None),
        ("deterministic_inference_rule_id", "  "),
    ],
)
def test_inferred_proposal_rejects_insufficient_duplicate_or_ruleless_shapes(
    field: str, value: object
) -> None:
    data = inferred_proposal()
    data[field] = value

    with pytest.raises(ValidationError):
        analysis.InferredProposal.model_validate(data)


def test_not_reported_requires_exact_negative_search_proof_shape() -> None:
    proposal = analysis.NotReportedProposal.model_validate(not_reported_proposal())

    assert proposal.proposed_status is FinalFieldStatus.NOT_REPORTED
    assert proposal.value is None
    assert proposal.evidence_units == ()
    assert proposal.negative_search_channels == (
        "section_headings",
        "alias_registry",
        "full_text_fts",
        "table_figure_captions",
        "metadata",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("value", "No objective found"),
        ("evidence_units", (standalone_unit(),)),
        (
            "negative_search_channels",
            (
                "section_headings",
                "alias_registry",
                "full_text_fts",
                "table_figure_captions",
            ),
        ),
        (
            "negative_search_channels",
            (
                "alias_registry",
                "section_headings",
                "full_text_fts",
                "table_figure_captions",
                "metadata",
            ),
        ),
        (
            "negative_search_channels",
            (
                "section_headings",
                "alias_registry",
                "full_text_fts",
                "table_figure_captions",
                "metadata",
                "retrieval_top_k",
            ),
        ),
        ("negative_search_policy_id", None),
        ("coverage_proof_id", None),
    ],
)
def test_not_reported_rejects_shortcuts_and_incomplete_proofs(
    field: str, value: object
) -> None:
    data = not_reported_proposal()
    data[field] = value

    with pytest.raises(ValidationError):
        analysis.NotReportedProposal.model_validate(data)


def test_conflicting_requires_distinct_units_rule_and_dimension() -> None:
    proposal = analysis.ConflictingProposal.model_validate(conflicting_proposal())

    assert proposal.proposed_status is FinalFieldStatus.CONFLICTING
    assert proposal.value is None
    assert len(proposal.evidence_units) == 2
    assert proposal.conflict_dimension == "incompatible objective scope"


@pytest.mark.parametrize(
    "field,value",
    [
        ("value", "Preferred objective"),
        ("evidence_units", (standalone_unit(),)),
        ("evidence_units", (standalone_unit(), standalone_unit())),
        ("deterministic_conflict_rule_id", None),
        ("conflict_dimension", None),
        ("conflict_dimension", "  "),
    ],
)
def test_conflicting_rejects_value_insufficient_units_or_missing_rule_shape(
    field: str, value: object
) -> None:
    data = conflicting_proposal()
    data[field] = value

    with pytest.raises(ValidationError):
        analysis.ConflictingProposal.model_validate(data)


@pytest.mark.parametrize("reason", ["low", "empty", "unknown", "needs_ocr"])
def test_unreadable_accepts_each_frozen_coverage_reason(reason: str) -> None:
    data = unreadable_proposal()
    blocker = coverage_blocker()
    blocker["reason"] = reason
    data["coverage_blockers"] = (blocker,)

    proposal = analysis.UnreadableProposal.model_validate(data)

    assert proposal.proposed_status is FinalFieldStatus.UNREADABLE
    assert proposal.coverage_blockers[0].reason == reason


@pytest.mark.parametrize(
    "field,value",
    [
        ("value", "Recovered objective"),
        ("coverage_blockers", ()),
        (
            "coverage_blockers",
            (
                {
                    "page_id": "page-1",
                    "physical_page_index": 0,
                    "reason": "corrupt",
                },
            ),
        ),
    ],
)
def test_unreadable_rejects_value_missing_blocker_or_unknown_reason(
    field: str, value: object
) -> None:
    data = unreadable_proposal()
    data[field] = value

    with pytest.raises(ValidationError):
        analysis.UnreadableProposal.model_validate(data)


@pytest.mark.parametrize("physical_page_index", [-1, 1.0, "1"])
def test_coverage_blocker_requires_strict_nonnegative_physical_index(
    physical_page_index: object,
) -> None:
    data = coverage_blocker()
    data["physical_page_index"] = physical_page_index

    with pytest.raises(ValidationError):
        analysis.CoverageBlocker.model_validate(data)


@pytest.mark.parametrize(
    "model_name,factory",
    [
        ("SupportedProposal", supported_proposal),
        ("InferredProposal", inferred_proposal),
        ("NotReportedProposal", not_reported_proposal),
        ("ConflictingProposal", conflicting_proposal),
        ("UnreadableProposal", unreadable_proposal),
    ],
)
@pytest.mark.parametrize("invalid", [True, 0, None, "false"])
def test_every_proposal_rejects_nonfalse_generation_values(
    model_name: str, factory: object, invalid: object
) -> None:
    data = factory()
    data["generation_enabled"] = invalid

    with pytest.raises(ValidationError):
        getattr(analysis, model_name).model_validate(data)


@pytest.mark.parametrize(
    "extra",
    [
        "confidence",
        "model_output",
        "support_assessment",
        "generation_token_count",
        "source_text",
    ],
)
def test_proposals_reject_unfrozen_or_untrusted_fields(extra: str) -> None:
    data = supported_proposal()
    data[extra] = "not authorized"

    with pytest.raises(ValidationError):
        analysis.SupportedProposal.model_validate(data)


def test_proposals_are_strict_frozen_and_reject_invalid_unicode() -> None:
    proposal = analysis.SupportedProposal.model_validate(supported_proposal())

    with pytest.raises(ValidationError):
        proposal.value = "Changed"

    data = supported_proposal()
    data["value"] = 123
    with pytest.raises(ValidationError):
        analysis.SupportedProposal.model_validate(data)

    data = supported_proposal()
    data["value"] = "\ud800"
    with pytest.raises(ValidationError):
        analysis.SupportedProposal.model_validate(data)


def test_proposal_tuple_limits_fail_closed() -> None:
    supported = supported_proposal()
    supported["evidence_units"] = tuple(
        {
            "kind": "standalone",
            "bundle_fingerprint": f"{index:064x}",
            "citation_labels": ("E1",),
        }
        for index in range(201)
    )
    with pytest.raises(ValidationError):
        analysis.SupportedProposal.model_validate(supported)

    unreadable = unreadable_proposal()
    unreadable["coverage_blockers"] = tuple(
        {
            "page_id": f"page-{index}",
            "physical_page_index": index,
            "reason": "low",
        }
        for index in range(1001)
    )
    with pytest.raises(ValidationError):
        analysis.UnreadableProposal.model_validate(unreadable)


def test_objective_union_rejects_unsupported_as_a_sixth_status() -> None:
    data = supported_proposal()
    data["proposed_status"] = "unsupported"

    with pytest.raises(ValidationError):
        TypeAdapter(analysis.ObjectiveStatusProposal).validate_python(data)


def test_public_package_exports_only_the_ten_proposal_contract_names() -> None:
    assert set(analysis.__all__) == {
        "StandaloneAnalysisEvidenceUnit",
        "GroupedAnalysisEvidenceUnit",
        "AnalysisEvidenceUnit",
        "CoverageBlocker",
        "SupportedProposal",
        "InferredProposal",
        "NotReportedProposal",
        "ConflictingProposal",
        "UnreadableProposal",
        "ObjectiveStatusProposal",
        "SUPPORTED_OBJECTIVE_VERIFIER_POLICY_ID",
        "SupportedObjectiveVerificationRequest",
        "SupportedObjectiveVerified",
        "SupportedObjectiveAbstained",
        "SupportedObjectiveRejected",
        "SupportedObjectiveFailed",
        "SupportedObjectiveAbstentionReason",
        "SupportedObjectiveRejectionCode",
        "SupportedObjectiveFailureCode",
        "SupportedObjectiveVerificationResult",
        "SupportedObjectiveCitationVerifier",
    }
