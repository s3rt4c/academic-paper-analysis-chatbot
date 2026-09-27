"""Strict opt-in evidence-group contracts and canonical identity helpers."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, computed_field, model_validator

from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.serialization import (
    _bounded_canonical_bytes,
    _check_request_depth,
    _reject_request_constant,
    _unique_request_object,
    _validate_json,
)

EVIDENCE_GROUP_POLICY_ID: Literal["evidence-group-v1"] = "evidence-group-v1"
EVIDENCE_GROUP_REQUEST_SCHEMA: Literal["evidence-group-request-v1"] = (
    "evidence-group-request-v1"
)
EVIDENCE_GROUP_BUNDLE_SCHEMA: Literal["evidence-group-bundle-v1"] = (
    "evidence-group-bundle-v1"
)


def _validate_grouping_policy_id(value: str) -> str:
    import re

    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*-v[1-9][0-9]*", value) is None:
        raise ValueError("invalid grouping policy")
    return value


GroupingPolicyId = Annotated[
    str,
    Field(strict=True, min_length=3, max_length=128),
    AfterValidator(_validate_grouping_policy_id),
]
GroupId = Annotated[str, Field(strict=True, pattern=r"^sha256-[0-9a-f]{64}$")]


class EvidenceGroupErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_GROUP_IDENTITY = "INVALID_GROUP_IDENTITY"
    DUPLICATE_GROUP_MEMBER = "DUPLICATE_GROUP_MEMBER"
    DUPLICATE_GROUP = "DUPLICATE_GROUP"
    DUPLICATE_LOGICAL_UNIT = "DUPLICATE_LOGICAL_UNIT"
    GROUP_LINEAGE_MISMATCH = "GROUP_LINEAGE_MISMATCH"
    GROUP_MEMBER_RESOLUTION_FAILED = "GROUP_MEMBER_RESOLUTION_FAILED"
    GROUP_RANGE_OVERLAP = "GROUP_RANGE_OVERLAP"
    GROUP_PACKING_FAILED = "GROUP_PACKING_FAILED"


_SAFE_MESSAGES = {
    EvidenceGroupErrorCode.INVALID_REQUEST: "Evidence group request is invalid",
    EvidenceGroupErrorCode.RESOURCE_LIMIT: "Evidence group resource limit exceeded",
    EvidenceGroupErrorCode.INVALID_GROUP_IDENTITY: "Evidence group identity is invalid",
    EvidenceGroupErrorCode.DUPLICATE_GROUP_MEMBER: "Evidence group member is duplicated",
    EvidenceGroupErrorCode.DUPLICATE_GROUP: "Evidence group is duplicated",
    EvidenceGroupErrorCode.DUPLICATE_LOGICAL_UNIT: "Evidence logical unit is duplicated",
    EvidenceGroupErrorCode.GROUP_LINEAGE_MISMATCH: "Evidence group lineage does not match",
    EvidenceGroupErrorCode.GROUP_MEMBER_RESOLUTION_FAILED: (
        "Evidence group member resolution failed"
    ),
    EvidenceGroupErrorCode.GROUP_RANGE_OVERLAP: "Evidence group ranges overlap",
    EvidenceGroupErrorCode.GROUP_PACKING_FAILED: "Evidence group packing failed",
}


class EvidenceGroupError(m.EvidenceValue):
    code: EvidenceGroupErrorCode
    unit_index: m.InputPosition | None = None
    input_position: m.InputPosition | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def message(self) -> str:
        return _SAFE_MESSAGES[self.code]


class EvidenceGroupPreparationError(ValueError):
    def __init__(self, error: EvidenceGroupError) -> None:
        self.error = error
        super().__init__(error.message)


class EvidenceGroupMemberRequest(m.EvidenceValue):
    input_position: m.InputPosition
    child_identity: AuthoritativeChildIdentity


class StandaloneUnitRequest(m.EvidenceValue):
    kind: Literal["standalone"] = "standalone"
    input_position: m.InputPosition


class EvidenceGroupUnitRequest(m.EvidenceValue):
    kind: Literal["evidence_group"] = "evidence_group"
    grouping_policy_id: GroupingPolicyId
    requires_all: Literal[True] = True
    members: tuple[EvidenceGroupMemberRequest, EvidenceGroupMemberRequest]

    @model_validator(mode="after")
    def _distinct_members(self) -> EvidenceGroupUnitRequest:
        if self.members[0].input_position == self.members[1].input_position:
            raise ValueError("group member positions must be distinct")
        if self.members[0].child_identity == self.members[1].child_identity:
            raise ValueError("group member identities must be distinct")
        return self


type EvidenceGroupUnitRequestType = Annotated[
    StandaloneUnitRequest | EvidenceGroupUnitRequest, Field(discriminator="kind")
]


class EvidenceGroupRequest(m.EvidenceValue):
    schema_version: Literal["evidence-group-request-v1"] = "evidence-group-request-v1"
    policy_id: Literal["evidence-group-v1"] = "evidence-group-v1"
    base_request: m.EvidenceBundleRequest
    units: Annotated[
        tuple[EvidenceGroupUnitRequestType, ...],
        Field(min_length=1, max_length=m.MAX_CANDIDATE_GROUPS),
    ]

    @model_validator(mode="after")
    def _distinct_unit_positions(self) -> EvidenceGroupRequest:
        positions: list[int] = []
        for unit in self.units:
            if isinstance(unit, StandaloneUnitRequest):
                positions.append(unit.input_position)
            else:
                positions.extend(member.input_position for member in unit.members)
        if len(positions) != len(set(positions)):
            raise ValueError("logical unit positions must be distinct")
        return self


class AuthoritativeChildIdentity(m.EvidenceValue):
    scope: m.ResolvedBundleSource
    parent: m.ParentIdentity
    physical_page_index: m.Nonnegative
    start_offset: m.Nonnegative
    end_offset: m.Positive
    canonical_page_text_sha256: m.Sha256
    parser_profile_sha256: m.Sha256
    processing_profile_id: m.Identifier
    text_sha256: m.Sha256
    anchor_ids: m.AnchorIds


def authoritative_child_identity(value: m.ResolvedEvidenceRange) -> AuthoritativeChildIdentity:
    source = value.source
    return AuthoritativeChildIdentity(
        scope=source.scope,
        parent=source.parent,
        physical_page_index=source.physical_page_index,
        start_offset=source.start_offset,
        end_offset=source.end_offset,
        canonical_page_text_sha256=source.canonical_page_text_sha256,
        parser_profile_sha256=source.parser_profile_sha256,
        processing_profile_id=source.processing_profile_id,
        text_sha256=source.text_sha256,
        anchor_ids=source.anchor_ids,
    )


def canonical_child_sort_key(
    value: AuthoritativeChildIdentity,
) -> tuple[int, int, int, str, str, str, tuple[str, ...]]:
    return (
        value.physical_page_index,
        value.start_offset,
        value.end_offset,
        value.parent.page_id,
        value.parent.chunk_id,
        value.text_sha256,
        value.anchor_ids,
    )


def same_group_lineage(
    left: AuthoritativeChildIdentity, right: AuthoritativeChildIdentity
) -> bool:
    return (
        left.scope.project_id,
        left.scope.paper_id,
        left.scope.file_version_id,
        left.scope.document_generation_id,
        left.scope.source_pdf_sha256,
    ) == (
        right.scope.project_id,
        right.scope.paper_id,
        right.scope.file_version_id,
        right.scope.document_generation_id,
        right.scope.source_pdf_sha256,
    )


def canonical_range_overlap(
    left: AuthoritativeChildIdentity, right: AuthoritativeChildIdentity
) -> bool:
    if left.physical_page_index != right.physical_page_index:
        return False
    return max(left.start_offset, right.start_offset) < min(left.end_offset, right.end_offset)


def canonical_group_id(
    grouping_policy_id: GroupingPolicyId,
    children: tuple[AuthoritativeChildIdentity, AuthoritativeChildIdentity],
) -> GroupId:
    if children != tuple(sorted(children, key=canonical_child_sort_key)):
        raise ValueError("children are not in canonical order")
    identity_object = {
        "children": [child.model_dump(mode="json") for child in children],
        "grouping_policy_id": grouping_policy_id,
        "policy_id": EVIDENCE_GROUP_POLICY_ID,
    }
    digest = hashlib.sha256(
        json.dumps(
            identity_object,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return f"sha256-{digest}"


class ResolvedStandaloneUnit(m.EvidenceValue):
    kind: Literal["standalone"] = "standalone"
    input_position: m.InputPosition
    reference: m.EvidenceCandidateRef
    ranges: Annotated[tuple[m.ResolvedEvidenceRange, ...], Field(min_length=1, max_length=2)]
    context: m.ResolvedContext | None = None


class EvidenceGroupChild(m.EvidenceValue):
    kind: Literal["group_child"] = "group_child"
    input_position: m.InputPosition
    identity: AuthoritativeChildIdentity
    range: m.ResolvedEvidenceRange
    context: m.ResolvedContext | None = None


class ResolvedEvidenceGroup(m.EvidenceValue):
    kind: Literal["evidence_group"] = "evidence_group"
    group_id: GroupId
    grouping_policy_id: GroupingPolicyId
    requires_all: Literal[True] = True
    children: tuple[EvidenceGroupChild, EvidenceGroupChild]


type ResolvedEvidenceUnit = Annotated[
    ResolvedStandaloneUnit | ResolvedEvidenceGroup, Field(discriminator="kind")
]


class StandaloneLogicalUnit(m.EvidenceValue):
    kind: Literal["standalone"] = "standalone"
    input_position: m.InputPosition
    citation_labels: Annotated[tuple[m.CitationLabel, ...], Field(min_length=1, max_length=2)]


class GroupChildLogicalUnit(m.EvidenceValue):
    kind: Literal["group_child"] = "group_child"
    citation_label: m.CitationLabel
    child_identity: AuthoritativeChildIdentity


class EvidenceGroupLogicalUnit(m.EvidenceValue):
    kind: Literal["evidence_group"] = "evidence_group"
    group_id: GroupId
    grouping_policy_id: GroupingPolicyId
    requires_all: Literal[True] = True
    children: tuple[GroupChildLogicalUnit, GroupChildLogicalUnit]


type EvidenceGroupLogicalUnitType = Annotated[
    StandaloneLogicalUnit | EvidenceGroupLogicalUnit, Field(discriminator="kind")
]


class EvidenceGroupOmission(m.EvidenceValue):
    reason: Literal["preview_bytes_exceeded", "entry_limit_exceeded"]
    violated_constraints: Annotated[
        tuple[Literal["preview_bytes_exceeded", "entry_limit_exceeded"], ...],
        Field(min_length=1, max_length=2),
    ]


class StandaloneDisposition(m.EvidenceValue):
    kind: Literal["standalone"] = "standalone"
    input_position: m.InputPosition
    reference: m.EvidenceCandidateRef
    citation_labels: Annotated[tuple[m.CitationLabel, ...], Field(max_length=2)] = ()
    omission: m.EvidenceOmission | None = None


class EvidenceGroupDisposition(m.EvidenceValue):
    kind: Literal["evidence_group"] = "evidence_group"
    group_id: GroupId
    grouping_policy_id: GroupingPolicyId
    member_input_positions: tuple[m.InputPosition, m.InputPosition]
    citation_labels: Annotated[tuple[m.CitationLabel, ...], Field(max_length=2)] = ()
    omission: EvidenceGroupOmission | None = None


type DispositionType = Annotated[
    StandaloneDisposition | EvidenceGroupDisposition, Field(discriminator="kind")
]


class EvidenceGroupBundlePayload(m.EvidenceValue):
    schema_version: Literal["evidence-group-bundle-v1"] = "evidence-group-bundle-v1"
    policy_id: Literal["evidence-group-v1"] = "evidence-group-v1"
    scope: m.ResolvedBundleSource
    origin: m.CandidateOrigin
    concern_profile_id: Literal["stated-study-objective-v1"] = "stated-study-objective-v1"
    selection_mode: Literal["explicit_candidates"] = "explicit_candidates"
    selected_source_retrieval_completeness: Literal["not_assessed"] = "not_assessed"
    logical_units: Annotated[
        tuple[EvidenceGroupLogicalUnitType, ...], Field(max_length=m.MAX_CANDIDATE_GROUPS)
    ]
    entries: Annotated[tuple[m.EvidenceBundleEntry, ...], Field(max_length=m.MAX_ENTRIES)]
    context: Annotated[tuple[m.EvidenceContext, ...], Field(max_length=m.MAX_CANDIDATE_GROUPS)]
    coverage: m.EvidenceCoverage
    dispositions: Annotated[
        tuple[DispositionType, ...], Field(max_length=m.MAX_CANDIDATE_GROUPS)
    ]
    budget: m.EvidenceBudgetUsage
    state: m.PreviewState
    insufficient_reason: m.InsufficientReason | None = None
    generation_enabled: Literal[False] = False
    generation_readiness: Literal["not_assessed"] = "not_assessed"
    support_assessment: Literal["not_performed"] = "not_performed"
    generation_token_count: None = None
    content_preview: m.ContentPreview


class EvidenceGroupBundle(EvidenceGroupBundlePayload):
    fingerprint: m.Sha256


def parse_evidence_group_request(raw: bytes) -> EvidenceGroupRequest:
    if len(raw) > m.MAX_REQUEST_BYTES:
        raise EvidenceGroupPreparationError(
            EvidenceGroupError(code=EvidenceGroupErrorCode.RESOURCE_LIMIT)
        )
    try:
        text = raw.decode("utf-8", errors="strict")
        _check_request_depth(text)
        structural = json.loads(
            text,
            object_pairs_hook=_unique_request_object,
            parse_constant=_reject_request_constant,
        )
        _validate_json(structural)
        return EvidenceGroupRequest.model_validate_json(text)
    except EvidenceGroupPreparationError:
        raise
    except (UnicodeError, TypeError, ValueError):
        raise EvidenceGroupPreparationError(
            EvidenceGroupError(code=EvidenceGroupErrorCode.INVALID_REQUEST)
        ) from None


def _bounded_group_bytes(value: object) -> bytes:
    try:
        return _bounded_canonical_bytes(value)
    except Exception as error:
        if isinstance(error, EvidenceGroupPreparationError):
            raise
        raise EvidenceGroupPreparationError(
            EvidenceGroupError(code=EvidenceGroupErrorCode.RESOURCE_LIMIT)
        ) from None


def compute_evidence_group_fingerprint(payload: EvidenceGroupBundlePayload) -> str:
    return hashlib.sha256(_bounded_group_bytes(payload.model_dump(mode="json"))).hexdigest()


def seal_evidence_group_bundle(payload: EvidenceGroupBundlePayload) -> EvidenceGroupBundle:
    fingerprint = compute_evidence_group_fingerprint(payload)
    return EvidenceGroupBundle.model_validate(
        {**payload.model_dump(mode="python"), "fingerprint": fingerprint}
    )


def canonical_evidence_group_bytes(bundle: EvidenceGroupBundle) -> bytes:
    payload = EvidenceGroupBundlePayload.model_validate(
        {name: getattr(bundle, name) for name in EvidenceGroupBundlePayload.model_fields}
    )
    if bundle.fingerprint != compute_evidence_group_fingerprint(payload):
        raise EvidenceGroupPreparationError(
            EvidenceGroupError(code=EvidenceGroupErrorCode.GROUP_PACKING_FAILED)
        )
    return _bounded_group_bytes(bundle.model_dump(mode="json"))
