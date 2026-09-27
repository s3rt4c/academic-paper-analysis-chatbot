"""Deterministic exact-text verification for caller-asserted objectives."""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal, NamedTuple

from pydantic import Field, ValidationError, computed_field

from academic_chatbot.analysis.models import (
    GroupedAnalysisEvidenceUnit,
    StandaloneAnalysisEvidenceUnit,
    SupportedProposal,
)
from academic_chatbot.evidence import groups as g
from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.resolver import EvidenceReadResolver, EvidenceResolutionError
from academic_chatbot.evidence.serialization import (
    EvidencePreparationError,
    _bounded_canonical_bytes,
    canonical_bundle_bytes,
)
from academic_chatbot.evidence.service import EvidenceBundleService

SUPPORTED_OBJECTIVE_VERIFIER_POLICY_ID = "supported-objective-exact-text-v1"
SUPPORTED_OBJECTIVE_REQUEST_SCHEMA = "supported-objective-verification-request-v1"
SUPPORTED_OBJECTIVE_RESULT_SCHEMA = "supported-objective-verification-result-v1"

AcceptedEvidenceBundle = m.EvidenceBundle | g.EvidenceGroupBundle


class SupportedObjectiveAbstentionReason(StrEnum):
    VALUE_NOT_EXACT_SOURCE_TEXT = "value_not_exact_source_text"
    EVIDENCE_GROUP_VALUE_PROJECTION_UNSUPPORTED = (
        "evidence_group_value_projection_unsupported"
    )


class SupportedObjectiveRejectionCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    POLICY_MISMATCH = "POLICY_MISMATCH"
    BUNDLE_FINGERPRINT_INVALID = "BUNDLE_FINGERPRINT_INVALID"
    SOURCE_NOT_CURRENT = "SOURCE_NOT_CURRENT"
    EVIDENCE_REFERENCE_INVALID = "EVIDENCE_REFERENCE_INVALID"
    EVIDENCE_GROUP_INCOMPLETE = "EVIDENCE_GROUP_INCOMPLETE"
    AUTHORITATIVE_EVIDENCE_INVALID = "AUTHORITATIVE_EVIDENCE_INVALID"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


class SupportedObjectiveFailureCode(StrEnum):
    AUTHORITY_UNAVAILABLE = "AUTHORITY_UNAVAILABLE"
    INTERNAL_INVARIANT_FAILURE = "INTERNAL_INVARIANT_FAILURE"


_REJECTION_MESSAGES = {
    SupportedObjectiveRejectionCode.INVALID_REQUEST: "Verification request is invalid",
    SupportedObjectiveRejectionCode.POLICY_MISMATCH: "Verification policy is not supported",
    SupportedObjectiveRejectionCode.BUNDLE_FINGERPRINT_INVALID: (
        "Evidence bundle fingerprint is invalid"
    ),
    SupportedObjectiveRejectionCode.SOURCE_NOT_CURRENT: "Evidence source is not current",
    SupportedObjectiveRejectionCode.EVIDENCE_REFERENCE_INVALID: (
        "Evidence reference is invalid"
    ),
    SupportedObjectiveRejectionCode.EVIDENCE_GROUP_INCOMPLETE: (
        "Evidence group is incomplete"
    ),
    SupportedObjectiveRejectionCode.AUTHORITATIVE_EVIDENCE_INVALID: (
        "Authoritative evidence is invalid"
    ),
    SupportedObjectiveRejectionCode.RESOURCE_LIMIT: (
        "Verification resource limit exceeded"
    ),
}

_FAILURE_MESSAGES = {
    SupportedObjectiveFailureCode.AUTHORITY_UNAVAILABLE: (
        "Evidence authority is unavailable"
    ),
    SupportedObjectiveFailureCode.INTERNAL_INVARIANT_FAILURE: (
        "Verification failed safely"
    ),
}


class SupportedObjectiveVerificationRequest(m.EvidenceValue):
    schema_version: Literal["supported-objective-verification-request-v1"] = (
        "supported-objective-verification-request-v1"
    )
    policy_id: m.Identifier
    proposal: SupportedProposal
    bundle: m.EvidenceBundle | g.EvidenceGroupBundle


class SupportedObjectiveVerified(m.EvidenceValue):
    schema_version: Literal["supported-objective-verification-result-v1"] = (
        "supported-objective-verification-result-v1"
    )
    outcome: Literal["verified"] = "verified"
    policy_id: Literal["supported-objective-exact-text-v1"] = (
        "supported-objective-exact-text-v1"
    )
    concern_profile_id: Literal["stated-study-objective-v1"] = (
        "stated-study-objective-v1"
    )
    proposal_fingerprint: m.Sha256
    bundle_fingerprint: m.Sha256


class SupportedObjectiveAbstained(m.EvidenceValue):
    schema_version: Literal["supported-objective-verification-result-v1"] = (
        "supported-objective-verification-result-v1"
    )
    outcome: Literal["abstained"] = "abstained"
    policy_id: Literal["supported-objective-exact-text-v1"] = (
        "supported-objective-exact-text-v1"
    )
    concern_profile_id: Literal["stated-study-objective-v1"] = (
        "stated-study-objective-v1"
    )
    proposal_fingerprint: m.Sha256
    bundle_fingerprint: m.Sha256
    reason: SupportedObjectiveAbstentionReason


class SupportedObjectiveRejected(m.EvidenceValue):
    schema_version: Literal["supported-objective-verification-result-v1"] = (
        "supported-objective-verification-result-v1"
    )
    outcome: Literal["rejected"] = "rejected"
    code: SupportedObjectiveRejectionCode

    @computed_field  # type: ignore[prop-decorator]
    @property
    def message(self) -> str:
        return _REJECTION_MESSAGES[self.code]


class SupportedObjectiveFailed(m.EvidenceValue):
    schema_version: Literal["supported-objective-verification-result-v1"] = (
        "supported-objective-verification-result-v1"
    )
    outcome: Literal["failed"] = "failed"
    code: SupportedObjectiveFailureCode

    @computed_field  # type: ignore[prop-decorator]
    @property
    def message(self) -> str:
        return _FAILURE_MESSAGES[self.code]


SupportedObjectiveVerificationResult = Annotated[
    SupportedObjectiveVerified
    | SupportedObjectiveAbstained
    | SupportedObjectiveRejected
    | SupportedObjectiveFailed,
    Field(discriminator="outcome"),
]


class _BoundStandalone(NamedTuple):
    proposal_unit: StandaloneAnalysisEvidenceUnit
    entries: tuple[m.EvidenceBundleEntry, ...]


class _BoundGroup(NamedTuple):
    proposal_unit: GroupedAnalysisEvidenceUnit
    logical_unit: g.EvidenceGroupLogicalUnit
    entries: tuple[m.EvidenceBundleEntry, m.EvidenceBundleEntry]


_BoundLogicalUnit = _BoundStandalone | _BoundGroup


def _proposal_fingerprint(proposal: SupportedProposal) -> str:
    return sha256(_bounded_canonical_bytes(proposal.model_dump(mode="json"))).hexdigest()


def _rejected(code: SupportedObjectiveRejectionCode) -> SupportedObjectiveRejected:
    return SupportedObjectiveRejected(code=code)


def _validate_bundle_fingerprint(
    bundle: AcceptedEvidenceBundle,
) -> SupportedObjectiveRejected | None:
    try:
        if isinstance(bundle, g.EvidenceGroupBundle):
            g.canonical_evidence_group_bytes(bundle)
        else:
            canonical_bundle_bytes(bundle)
    except EvidencePreparationError as error:
        if error.error.code == m.EvidenceErrorCode.RESOURCE_LIMIT:
            return _rejected(SupportedObjectiveRejectionCode.RESOURCE_LIMIT)
        return _rejected(SupportedObjectiveRejectionCode.BUNDLE_FINGERPRINT_INVALID)
    except g.EvidenceGroupPreparationError as error:
        if error.error.code == g.EvidenceGroupErrorCode.RESOURCE_LIMIT:
            return _rejected(SupportedObjectiveRejectionCode.RESOURCE_LIMIT)
        return _rejected(SupportedObjectiveRejectionCode.BUNDLE_FINGERPRINT_INVALID)
    return None


def _entries_for_labels(
    labels: tuple[str, ...], bundle: AcceptedEvidenceBundle
) -> tuple[m.EvidenceBundleEntry, ...] | None:
    entries = []
    for label in labels:
        matches = tuple(entry for entry in bundle.entries if entry.citation_label == label)
        if len(matches) != 1:
            return None
        entries.append(matches[0])
    return tuple(entries)


def _bind_logical_units(
    proposal: SupportedProposal,
    bundle: AcceptedEvidenceBundle,
) -> tuple[_BoundLogicalUnit, ...] | SupportedObjectiveRejected:
    seen_identities: set[tuple[object, ...]] = set()
    seen_labels: set[str] = set()
    bound: list[_BoundLogicalUnit] = []

    for unit in proposal.evidence_units:
        if unit.bundle_fingerprint != bundle.fingerprint:
            return _rejected(SupportedObjectiveRejectionCode.EVIDENCE_REFERENCE_INVALID)
        group_id = unit.group_id if isinstance(unit, GroupedAnalysisEvidenceUnit) else None
        identity = (unit.kind, unit.bundle_fingerprint, unit.citation_labels, group_id)
        if identity in seen_identities:
            return _rejected(SupportedObjectiveRejectionCode.EVIDENCE_REFERENCE_INVALID)
        seen_identities.add(identity)
        if len(unit.citation_labels) != len(set(unit.citation_labels)):
            return _rejected(SupportedObjectiveRejectionCode.EVIDENCE_REFERENCE_INVALID)
        if any(label in seen_labels for label in unit.citation_labels):
            return _rejected(SupportedObjectiveRejectionCode.EVIDENCE_REFERENCE_INVALID)
        seen_labels.update(unit.citation_labels)

        entries = _entries_for_labels(unit.citation_labels, bundle)
        if entries is None:
            if isinstance(bundle, g.EvidenceGroupBundle) and isinstance(
                unit, GroupedAnalysisEvidenceUnit
            ):
                return _rejected(SupportedObjectiveRejectionCode.EVIDENCE_GROUP_INCOMPLETE)
            return _rejected(SupportedObjectiveRejectionCode.EVIDENCE_REFERENCE_INVALID)

        if isinstance(bundle, m.EvidenceBundle):
            if isinstance(unit, GroupedAnalysisEvidenceUnit):
                return _rejected(SupportedObjectiveRejectionCode.EVIDENCE_REFERENCE_INVALID)
            disposition_matches = tuple(
                disposition
                for disposition in bundle.dispositions
                if disposition.omission is None
                and disposition.citation_labels
                and disposition.citation_labels == unit.citation_labels
            )
            if len(disposition_matches) != 1:
                return _rejected(SupportedObjectiveRejectionCode.EVIDENCE_REFERENCE_INVALID)
            bound.append(_BoundStandalone(unit, entries))
            continue

        if isinstance(unit, StandaloneAnalysisEvidenceUnit):
            standalone_matches = tuple(
                logical
                for logical in bundle.logical_units
                if isinstance(logical, g.StandaloneLogicalUnit)
                and logical.citation_labels == unit.citation_labels
            )
            if len(standalone_matches) != 1:
                group_labels = {
                    child.citation_label
                    for logical in bundle.logical_units
                    if isinstance(logical, g.EvidenceGroupLogicalUnit)
                    for child in logical.children
                }
                code = (
                    SupportedObjectiveRejectionCode.EVIDENCE_GROUP_INCOMPLETE
                    if any(label in group_labels for label in unit.citation_labels)
                    else SupportedObjectiveRejectionCode.EVIDENCE_REFERENCE_INVALID
                )
                return _rejected(code)
            bound.append(_BoundStandalone(unit, entries))
            continue

        group_matches = tuple(
            logical
            for logical in bundle.logical_units
            if isinstance(logical, g.EvidenceGroupLogicalUnit)
            and logical.group_id == unit.group_id
            and tuple(child.citation_label for child in logical.children)
            == unit.citation_labels
        )
        if len(group_matches) != 1 or len(entries) != 2:
            return _rejected(SupportedObjectiveRejectionCode.EVIDENCE_GROUP_INCOMPLETE)
        bound.append(_BoundGroup(unit, group_matches[0], (entries[0], entries[1])))

    return tuple(bound)


def _direct_support(
    proposal: SupportedProposal,
    bundle: AcceptedEvidenceBundle,
    bound_units: tuple[_BoundLogicalUnit, ...],
) -> SupportedObjectiveVerified | SupportedObjectiveAbstained:
    proposal_fingerprint = _proposal_fingerprint(proposal)
    if any(isinstance(unit, _BoundGroup) for unit in bound_units):
        return SupportedObjectiveAbstained(
            proposal_fingerprint=proposal_fingerprint,
            bundle_fingerprint=bundle.fingerprint,
            reason=(
                SupportedObjectiveAbstentionReason.EVIDENCE_GROUP_VALUE_PROJECTION_UNSUPPORTED
            ),
        )
    expected = proposal.value.encode("utf-8", errors="strict")
    for unit in bound_units:
        for entry in unit.entries:
            if expected != entry.text.encode("utf-8", errors="strict"):
                return SupportedObjectiveAbstained(
                    proposal_fingerprint=proposal_fingerprint,
                    bundle_fingerprint=bundle.fingerprint,
                    reason=SupportedObjectiveAbstentionReason.VALUE_NOT_EXACT_SOURCE_TEXT,
                )
    return SupportedObjectiveVerified(
        proposal_fingerprint=proposal_fingerprint,
        bundle_fingerprint=bundle.fingerprint,
    )


def _map_authority_error(
    code: m.EvidenceErrorCode,
) -> SupportedObjectiveRejected | SupportedObjectiveFailed:
    if code in {
        m.EvidenceErrorCode.GENERATION_NOT_CURRENT,
        m.EvidenceErrorCode.VECTOR_NOT_CURRENT,
    }:
        return _rejected(SupportedObjectiveRejectionCode.SOURCE_NOT_CURRENT)
    if code in {
        m.EvidenceErrorCode.STORAGE_UNAVAILABLE,
        m.EvidenceErrorCode.VECTOR_UNAVAILABLE,
    }:
        return SupportedObjectiveFailed(
            code=SupportedObjectiveFailureCode.AUTHORITY_UNAVAILABLE
        )
    if code == m.EvidenceErrorCode.RESOURCE_LIMIT:
        return _rejected(SupportedObjectiveRejectionCode.RESOURCE_LIMIT)
    return _rejected(SupportedObjectiveRejectionCode.AUTHORITATIVE_EVIDENCE_INVALID)


def _range_reference(
    entry: m.EvidenceBundleEntry,
) -> tuple[m.LexicalRangeRef | None, m.SemanticRangeRef | None] | None:
    source = entry.source
    lexical: m.LexicalRangeRef | None = None
    semantic: m.SemanticRangeRef | None = None
    for provenance in source.provenance:
        if provenance.channel == "lexical":
            if lexical is not None:
                return None
            lexical = m.LexicalRangeRef(
                start_offset=source.start_offset,
                end_offset=source.end_offset,
                expected_text_sha256=source.text_sha256,
                expected_anchor_ids=source.anchor_ids,
            )
        else:
            if semantic is not None or provenance.embedding_span_id is None:
                return None
            semantic = m.SemanticRangeRef(
                start_offset=source.start_offset,
                end_offset=source.end_offset,
                expected_text_sha256=source.text_sha256,
                expected_anchor_ids=source.anchor_ids,
                embedding_span_id=provenance.embedding_span_id,
            )
    if lexical is None and semantic is None:
        return None
    return lexical, semantic


def _candidate_for_entries(
    entries: tuple[m.EvidenceBundleEntry, ...], reported_rank: int
) -> m.EvidenceCandidateRef | None:
    if len(entries) == 1:
        ranges = _range_reference(entries[0])
        if ranges is None:
            return None
        return m.EvidenceCandidateRef(
            parent=entries[0].source.parent,
            reported_rank=reported_rank,
            lexical=ranges[0],
            semantic=ranges[1],
        )
    if len(entries) != 2 or entries[0].source.parent != entries[1].source.parent:
        return None
    first = _range_reference(entries[0])
    second = _range_reference(entries[1])
    if first is None or second is None:
        return None
    if (first[0] is not None, first[1] is not None) == (
        second[0] is not None,
        second[1] is not None,
    ):
        return None
    if sum(value is not None for value in (*first, *second)) != 2:
        return None
    lexical = first[0] if first[0] is not None else second[0]
    semantic = first[1] if first[1] is not None else second[1]
    if lexical is None or semantic is None:
        return None
    return m.EvidenceCandidateRef(
        parent=entries[0].source.parent,
        reported_rank=reported_rank,
        lexical=lexical,
        semantic=semantic,
    )


def _range_equals_entry(
    resolved: m.ResolvedEvidenceRange, entry: m.EvidenceBundleEntry
) -> bool:
    return resolved.model_dump(mode="json") == entry.model_dump(
        mode="json", exclude={"citation_label"}
    )


class SupportedObjectiveCitationVerifier:
    def __init__(self, *, resolver: EvidenceReadResolver) -> None:
        self._resolver = resolver

    def verify(self, request: object) -> SupportedObjectiveVerificationResult:
        try:
            admitted = SupportedObjectiveVerificationRequest.model_validate(request)
        except (ValidationError, TypeError, ValueError):
            return SupportedObjectiveRejected(
                code=SupportedObjectiveRejectionCode.INVALID_REQUEST
            )
        if admitted.policy_id != SUPPORTED_OBJECTIVE_VERIFIER_POLICY_ID:
            return SupportedObjectiveRejected(
                code=SupportedObjectiveRejectionCode.POLICY_MISMATCH
            )
        try:
            _proposal_fingerprint(admitted.proposal)
        except EvidencePreparationError:
            return _rejected(SupportedObjectiveRejectionCode.RESOURCE_LIMIT)
        try:
            invalid_fingerprint = _validate_bundle_fingerprint(admitted.bundle)
            if invalid_fingerprint is not None:
                return invalid_fingerprint
            bound_units = _bind_logical_units(admitted.proposal, admitted.bundle)
            if isinstance(bound_units, SupportedObjectiveRejected):
                return bound_units
            authority_result = self._revalidate_authority(admitted.bundle, bound_units)
            if authority_result is not None:
                return authority_result
            return _direct_support(admitted.proposal, admitted.bundle, bound_units)
        except Exception:
            return SupportedObjectiveFailed(
                code=SupportedObjectiveFailureCode.INTERNAL_INVARIANT_FAILURE
            )

    def _revalidate_authority(
        self,
        bundle: AcceptedEvidenceBundle,
        bound_units: tuple[_BoundLogicalUnit, ...],
    ) -> SupportedObjectiveRejected | SupportedObjectiveFailed | None:
        try:
            if isinstance(bundle, m.EvidenceBundle):
                dispositions = tuple(
                    sorted(bundle.dispositions, key=lambda value: value.input_position)
                )
                positions = tuple(value.input_position for value in dispositions)
                if positions != tuple(range(len(dispositions))):
                    return _rejected(
                        SupportedObjectiveRejectionCode.EVIDENCE_REFERENCE_INVALID
                    )
                request = m.EvidenceBundleRequest(
                    scope=m.BundleSourceScope(
                        **bundle.scope.model_dump(exclude={"source_pdf_sha256"})
                    ),
                    concern_profile_id=bundle.concern_profile_id,
                    origin=bundle.origin,
                    candidates=tuple(value.reference for value in dispositions),
                    preview_budget=m.PreviewBudget(
                        profile_id=bundle.budget.profile_id,
                        max_content_bytes=bundle.budget.max_content_bytes,
                        max_entries=bundle.budget.max_entries,
                    ),
                )
                rebuilt = EvidenceBundleService(resolver=self._resolver).build(request)
                if canonical_bundle_bytes(rebuilt) != canonical_bundle_bytes(bundle):
                    return _rejected(
                        SupportedObjectiveRejectionCode.AUTHORITATIVE_EVIDENCE_INVALID
                    )
                return None

            candidates: list[m.EvidenceCandidateRef] = []
            for bound in bound_units:
                if isinstance(bound, _BoundStandalone):
                    candidate = _candidate_for_entries(
                        bound.entries, len(candidates) + 1
                    )
                    if candidate is None:
                        return _rejected(
                            SupportedObjectiveRejectionCode.EVIDENCE_REFERENCE_INVALID
                        )
                    candidates.append(candidate)
                else:
                    for entry in bound.entries:
                        candidate = _candidate_for_entries(
                            (entry,), len(candidates) + 1
                        )
                        if candidate is None:
                            return _rejected(
                                SupportedObjectiveRejectionCode.EVIDENCE_GROUP_INCOMPLETE
                            )
                        candidates.append(candidate)
            if len(candidates) > m.MAX_CANDIDATE_GROUPS:
                return _rejected(SupportedObjectiveRejectionCode.RESOURCE_LIMIT)
            request = m.EvidenceBundleRequest(
                scope=m.BundleSourceScope(
                    **bundle.scope.model_dump(exclude={"source_pdf_sha256"})
                ),
                concern_profile_id=bundle.concern_profile_id,
                origin=bundle.origin,
                candidates=tuple(candidates),
                preview_budget=m.PreviewBudget(
                    profile_id=bundle.budget.profile_id,
                    max_content_bytes=bundle.budget.max_content_bytes,
                    max_entries=bundle.budget.max_entries,
                ),
            )
            resolved = self._resolver.resolve(request)
            if len(resolved.groups) != len(candidates):
                return _rejected(
                    SupportedObjectiveRejectionCode.AUTHORITATIVE_EVIDENCE_INVALID
                )
            resolved_index = 0
            for bound in bound_units:
                if isinstance(bound, _BoundStandalone):
                    group = resolved.groups[resolved_index]
                    if group.input_position != resolved_index or len(group.ranges) != len(
                        bound.entries
                    ):
                        return _rejected(
                            SupportedObjectiveRejectionCode.AUTHORITATIVE_EVIDENCE_INVALID
                        )
                    if any(
                        not _range_equals_entry(value, entry)
                        for value, entry in zip(
                            group.ranges, bound.entries, strict=True
                        )
                    ):
                        return _rejected(
                            SupportedObjectiveRejectionCode.AUTHORITATIVE_EVIDENCE_INVALID
                        )
                    resolved_index += 1
                    continue

                children: list[g.AuthoritativeChildIdentity] = []
                for entry in bound.entries:
                    group = resolved.groups[resolved_index]
                    if (
                        group.input_position != resolved_index
                        or len(group.ranges) != 1
                        or not _range_equals_entry(group.ranges[0], entry)
                    ):
                        return _rejected(
                            SupportedObjectiveRejectionCode.AUTHORITATIVE_EVIDENCE_INVALID
                        )
                    children.append(g.authoritative_child_identity(group.ranges[0]))
                    resolved_index += 1
                expected_children = tuple(
                    child.child_identity for child in bound.logical_unit.children
                )
                if tuple(children) != expected_children:
                    return _rejected(
                        SupportedObjectiveRejectionCode.AUTHORITATIVE_EVIDENCE_INVALID
                    )
                canonical_id = g.canonical_group_id(
                    bound.logical_unit.grouping_policy_id,
                    (children[0], children[1]),
                )
                if canonical_id != bound.logical_unit.group_id:
                    return _rejected(
                        SupportedObjectiveRejectionCode.AUTHORITATIVE_EVIDENCE_INVALID
                    )
            return None
        except EvidenceResolutionError as error:
            return _map_authority_error(error.error.code)
        except EvidencePreparationError as error:
            return _map_authority_error(error.error.code)
