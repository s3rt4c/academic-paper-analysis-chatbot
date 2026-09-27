"""Caller assertions, not final decisions or authorization for factual output.

A separately authorized deterministic service must verify these proposal values
before any factual output may use them.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, Field, model_validator

from academic_chatbot.domain.enums import FinalFieldStatus
from academic_chatbot.evidence.models import (
    MAX_ENTRIES,
    MAX_SOURCE_PAGES,
    CitationLabel,
    Disabled,
    EvidenceValue,
    Identifier,
    Sha256,
)


class StandaloneAnalysisEvidenceUnit(EvidenceValue):
    kind: Literal["standalone"] = "standalone"
    bundle_fingerprint: Sha256
    citation_labels: Annotated[
        tuple[CitationLabel, ...], Field(min_length=1, max_length=2)
    ]


class GroupedAnalysisEvidenceUnit(EvidenceValue):
    kind: Literal["evidence_group"] = "evidence_group"
    bundle_fingerprint: Sha256
    group_id: Identifier
    citation_labels: tuple[CitationLabel, CitationLabel]
    requires_all: Literal[True] = True

    @model_validator(mode="after")
    def _distinct_labels(self) -> GroupedAnalysisEvidenceUnit:
        if self.citation_labels[0] == self.citation_labels[1]:
            raise ValueError("evidence group citation labels must be distinct")
        return self


AnalysisEvidenceUnit = Annotated[
    StandaloneAnalysisEvidenceUnit | GroupedAnalysisEvidenceUnit,
    Field(discriminator="kind"),
]


def _nonblank_utf8(value: str) -> str:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError("invalid Unicode") from error
    if not value.strip():
        raise ValueError("analysis text must be nonblank")
    return value


AnalysisText = Annotated[
    str,
    Field(strict=True, min_length=1),
    AfterValidator(_nonblank_utf8),
]

EvidenceUnits = Annotated[
    tuple[AnalysisEvidenceUnit, ...], Field(max_length=MAX_ENTRIES)
]

NegativeSearchChannels = tuple[
    Literal["section_headings"],
    Literal["alias_registry"],
    Literal["full_text_fts"],
    Literal["table_figure_captions"],
    Literal["metadata"],
]
NEGATIVE_SEARCH_CHANNELS: NegativeSearchChannels = (
    "section_headings",
    "alias_registry",
    "full_text_fts",
    "table_figure_captions",
    "metadata",
)


def _logical_unit_identity(
    unit: StandaloneAnalysisEvidenceUnit | GroupedAnalysisEvidenceUnit,
) -> tuple[object, ...]:
    group_id = unit.group_id if isinstance(unit, GroupedAnalysisEvidenceUnit) else None
    return (unit.kind, unit.bundle_fingerprint, unit.citation_labels, group_id)


def _require_distinct_logical_units(units: EvidenceUnits) -> None:
    identities = tuple(_logical_unit_identity(unit) for unit in units)
    if len(identities) != len(set(identities)):
        raise ValueError("analysis evidence units must be distinct")


class CoverageBlocker(EvidenceValue):
    page_id: Identifier
    physical_page_index: Annotated[int, Field(strict=True, ge=0)] | None
    reason: Literal["low", "empty", "unknown", "needs_ocr"]


class SupportedProposal(EvidenceValue):
    concern_profile_id: Literal["stated-study-objective-v1"] = (
        "stated-study-objective-v1"
    )
    proposed_status: Literal[FinalFieldStatus.SUPPORTED] = FinalFieldStatus.SUPPORTED
    value: AnalysisText
    evidence_units: Annotated[EvidenceUnits, Field(min_length=1)]
    generation_enabled: Disabled = False


class InferredProposal(EvidenceValue):
    concern_profile_id: Literal["stated-study-objective-v1"] = (
        "stated-study-objective-v1"
    )
    proposed_status: Literal[FinalFieldStatus.INFERRED] = FinalFieldStatus.INFERRED
    value: AnalysisText
    evidence_units: Annotated[EvidenceUnits, Field(min_length=2)]
    deterministic_inference_rule_id: Identifier
    generation_enabled: Disabled = False

    @model_validator(mode="after")
    def _distinct_evidence_units(self) -> InferredProposal:
        _require_distinct_logical_units(self.evidence_units)
        return self


class NotReportedProposal(EvidenceValue):
    concern_profile_id: Literal["stated-study-objective-v1"] = (
        "stated-study-objective-v1"
    )
    proposed_status: Literal[FinalFieldStatus.NOT_REPORTED] = (
        FinalFieldStatus.NOT_REPORTED
    )
    value: None = None
    evidence_units: tuple[()] = ()
    negative_search_channels: NegativeSearchChannels = NEGATIVE_SEARCH_CHANNELS
    negative_search_policy_id: Identifier
    coverage_proof_id: Identifier
    generation_enabled: Disabled = False

    @model_validator(mode="after")
    def _exact_negative_search_channels(self) -> NotReportedProposal:
        if self.negative_search_channels != NEGATIVE_SEARCH_CHANNELS:
            raise ValueError("negative search channels must match the frozen order")
        return self


class ConflictingProposal(EvidenceValue):
    concern_profile_id: Literal["stated-study-objective-v1"] = (
        "stated-study-objective-v1"
    )
    proposed_status: Literal[FinalFieldStatus.CONFLICTING] = (
        FinalFieldStatus.CONFLICTING
    )
    value: None = None
    evidence_units: Annotated[EvidenceUnits, Field(min_length=2)]
    deterministic_conflict_rule_id: Identifier
    conflict_dimension: AnalysisText
    generation_enabled: Disabled = False

    @model_validator(mode="after")
    def _distinct_evidence_units(self) -> ConflictingProposal:
        _require_distinct_logical_units(self.evidence_units)
        return self


class UnreadableProposal(EvidenceValue):
    concern_profile_id: Literal["stated-study-objective-v1"] = (
        "stated-study-objective-v1"
    )
    proposed_status: Literal[FinalFieldStatus.UNREADABLE] = FinalFieldStatus.UNREADABLE
    value: None = None
    evidence_units: EvidenceUnits = ()
    coverage_blockers: Annotated[
        tuple[CoverageBlocker, ...], Field(min_length=1, max_length=MAX_SOURCE_PAGES)
    ]
    generation_enabled: Disabled = False


ObjectiveStatusProposal = Annotated[
    SupportedProposal
    | InferredProposal
    | NotReportedProposal
    | ConflictingProposal
    | UnreadableProposal,
    Field(discriminator="proposed_status"),
]
