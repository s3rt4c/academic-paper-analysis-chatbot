"""Detached evidence values and strict wire contracts, without storage authority.

These values do not resolve references, assign citations, derive preview states,
pack candidates or seal fingerprints. Document text is always untrusted data.
"""

from __future__ import annotations

import re
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from academic_chatbot.ports.documents import PdfAnchorBox
from academic_chatbot.retrieval.hybrid_models import ChunkCandidateIdentity

MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_JSON_DEPTH = 16
MAX_CANDIDATE_GROUPS = 100
MAX_CONTRIBUTIONS = 2
MAX_IDENTIFIER_CHARS = 512
MAX_CHUNK_ANCHORS = 120
MAX_SOURCE_PAGES = 1000
MAX_PAGE_TEXT_BYTES = 1024 * 1024
MAX_RESOLVED_TEXT_BYTES = 16 * 1024 * 1024
DEFAULT_CONTENT_BYTES = 16 * 1024
MIN_CONTENT_BYTES = 77
MAX_CONTENT_BYTES = 64 * 1024
DEFAULT_ENTRIES = 8
MAX_ENTRIES = 200
MAX_AUDIT_BYTES = 4 * 1024 * 1024
MAX_VECTOR_ROWS = 100000
MAX_VECTOR_METADATA_BYTES = 16 * 1024 * 1024
MAX_VECTOR_FILE_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_PROFILE_BYTES = 64 * 1024
MAX_SOURCE_SNAPSHOT_ROWS = 10000


def _unicode(value: str) -> str:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError("invalid Unicode") from error
    return value


def _identifier(value: str) -> str:
    _unicode(value)
    if (
        not value.strip()
        or value in {".", ".."}
        or any(ord(char) < 32 or char in "/\\:" for char in value)
    ):
        raise ValueError("invalid identifier")
    return value


def _digest(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid SHA-256 digest")
    return value


def _label(value: str) -> str:
    if re.fullmatch(r"E[1-9][0-9]*", value) is None:
        raise ValueError("invalid citation label")
    return value


def _false(value: object) -> object:
    if value is not False:
        raise ValueError("value must be false")
    return value


Identifier = Annotated[
    str,
    Field(strict=True, min_length=1, max_length=MAX_IDENTIFIER_CHARS),
    AfterValidator(_identifier),
]
Sha256 = Annotated[str, Field(strict=True), AfterValidator(_digest)]
SourceText = Annotated[str, Field(strict=True), AfterValidator(_unicode)]
CitationLabel = Annotated[
    str, Field(strict=True, max_length=MAX_IDENTIFIER_CHARS), AfterValidator(_label)
]
Nonnegative = Annotated[int, Field(strict=True, ge=0)]
Positive = Annotated[int, Field(strict=True, gt=0)]
InputPosition = Annotated[int, Field(strict=True, ge=0, lt=MAX_CANDIDATE_GROUPS)]
Disabled = Annotated[Literal[False], BeforeValidator(_false)]
AnchorIds = Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=MAX_CHUNK_ANCHORS)]


class EvidenceValue(BaseModel):
    """Immutable typed values, including nested collections."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
    )


def _parent(value: object) -> object:
    # Existing identity models predate these admission limits. Validate their raw
    # strings too, without introducing a competing parent identity class.
    fields = ChunkCandidateIdentity.model_fields
    if isinstance(value, ChunkCandidateIdentity):
        values = {name: getattr(value, name) for name in fields}
    elif isinstance(value, dict):
        values = value
    else:
        raise ValueError("invalid parent identity")
    for name in fields:
        item = values.get(name)
        if type(item) is not str or not 1 <= len(item) <= MAX_IDENTIFIER_CHARS:
            raise ValueError("invalid parent identity field")
        _identifier(item)
    return value


ParentIdentity = Annotated[ChunkCandidateIdentity, BeforeValidator(_parent)]


class BundleSourceScope(EvidenceValue):
    project_id: Identifier
    paper_id: Identifier
    file_version_id: Identifier
    document_generation_id: Identifier


class CandidateOrigin(EvidenceValue):
    mode: Literal["lexical", "semantic", "hybrid"]
    ordering: Literal["caller_asserted_retrieval_order"] = "caller_asserted_retrieval_order"
    fusion_profile_id: Literal["rrf-v1"] | None = None
    embedding_profile_id: Identifier | None = None
    vector_generation_id: Identifier | None = None

    @model_validator(mode="after")
    def _mode_fields(self) -> Self:
        if self.mode == "lexical":
            if any(
                value is not None
                for value in (
                    self.fusion_profile_id,
                    self.embedding_profile_id,
                    self.vector_generation_id,
                )
            ):
                raise ValueError("lexical origin cannot carry semantic or fusion identities")
        else:
            if self.embedding_profile_id is None or self.vector_generation_id is None:
                raise ValueError("semantic origin identities are required")
            if (self.mode == "hybrid") != (self.fusion_profile_id == "rrf-v1"):
                raise ValueError("fusion identity must agree with origin mode")
        return self


class LexicalRangeRef(EvidenceValue):
    start_offset: Nonnegative
    end_offset: Positive
    expected_text_sha256: Sha256
    expected_anchor_ids: AnchorIds

    @model_validator(mode="after")
    def _range(self) -> Self:
        if self.end_offset <= self.start_offset:
            raise ValueError("range must be nonempty")
        return self


class SemanticRangeRef(LexicalRangeRef):
    embedding_span_id: Identifier


class EvidenceCandidateRef(EvidenceValue):
    parent: ParentIdentity
    reported_rank: Positive
    lexical: LexicalRangeRef | None = None
    semantic: SemanticRangeRef | None = None

    @model_validator(mode="after")
    def _contribution(self) -> Self:
        if self.lexical is None and self.semantic is None:
            raise ValueError("at least one contribution is required")
        return self


class PreviewBudget(EvidenceValue):
    profile_id: Literal["bundle-preview-bytes-v1"] = "bundle-preview-bytes-v1"
    max_content_bytes: Annotated[
        int, Field(strict=True, ge=MIN_CONTENT_BYTES, le=MAX_CONTENT_BYTES)
    ] = DEFAULT_CONTENT_BYTES
    max_entries: Annotated[int, Field(strict=True, ge=1, le=MAX_ENTRIES)] = DEFAULT_ENTRIES


class EvidenceBundleRequest(EvidenceValue):
    schema_version: Literal["evidence-bundle-request-v1"] = "evidence-bundle-request-v1"
    scope: BundleSourceScope
    concern_profile_id: Literal["stated-study-objective-v1"] = "stated-study-objective-v1"
    origin: CandidateOrigin
    candidates: Annotated[tuple[EvidenceCandidateRef, ...], Field(max_length=MAX_CANDIDATE_GROUPS)]
    preview_budget: PreviewBudget

    @model_validator(mode="after")
    def _channels(self) -> Self:
        for candidate in self.candidates:
            if self.origin.mode == "lexical" and (
                candidate.lexical is None or candidate.semantic is not None
            ):
                raise ValueError("candidate contributions do not match lexical origin")
            if self.origin.mode == "semantic" and (
                candidate.semantic is None or candidate.lexical is not None
            ):
                raise ValueError("candidate contributions do not match semantic origin")
        return self


class PreviewState(StrEnum):
    PREVIEW_READY = "preview_ready"
    PREVIEW_PARTIAL = "preview_partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class InsufficientReason(StrEnum):
    NO_CANDIDATES = "no_candidates"
    BUDGET_TOO_SMALL = "budget_too_small"


class OmissionReason(StrEnum):
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    PREVIEW_BYTES_EXCEEDED = "preview_bytes_exceeded"
    ENTRY_LIMIT_EXCEEDED = "entry_limit_exceeded"


class CoverageStatus(StrEnum):
    NO_FLAGGED_NATIVE_GAPS = "no_flagged_native_gaps"
    KNOWN_NATIVE_GAPS = "known_native_gaps"
    UNKNOWN_METADATA = "unknown_metadata"


class EvidenceErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    UNSAFE_PATH = "UNSAFE_PATH"
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
    STORAGE_INTEGRITY = "STORAGE_INTEGRITY"
    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    SOURCE_SCOPE_MISMATCH = "SOURCE_SCOPE_MISMATCH"
    GENERATION_NOT_CURRENT = "GENERATION_NOT_CURRENT"
    REFERENCE_NOT_FOUND = "REFERENCE_NOT_FOUND"
    RANGE_MISMATCH = "RANGE_MISMATCH"
    TEXT_DIGEST_MISMATCH = "TEXT_DIGEST_MISMATCH"
    ANCHOR_MISMATCH = "ANCHOR_MISMATCH"
    PROFILE_MISMATCH = "PROFILE_MISMATCH"
    VECTOR_UNAVAILABLE = "VECTOR_UNAVAILABLE"
    VECTOR_NOT_CURRENT = "VECTOR_NOT_CURRENT"
    VECTOR_INTEGRITY = "VECTOR_INTEGRITY"
    INVALID_ORDER = "INVALID_ORDER"
    CONTRADICTORY_DUPLICATE = "CONTRADICTORY_DUPLICATE"


_SAFE_MESSAGES = MappingProxyType(
    {
        EvidenceErrorCode.INVALID_REQUEST: "Request is invalid",
        EvidenceErrorCode.RESOURCE_LIMIT: "Resource limit exceeded",
        EvidenceErrorCode.UNSAFE_PATH: "Storage location is unsafe",
        EvidenceErrorCode.STORAGE_UNAVAILABLE: "Storage is unavailable",
        EvidenceErrorCode.STORAGE_INTEGRITY: "Stored evidence is inconsistent",
        EvidenceErrorCode.SOURCE_NOT_FOUND: "Source is unavailable",
        EvidenceErrorCode.SOURCE_SCOPE_MISMATCH: "Source scope does not match",
        EvidenceErrorCode.GENERATION_NOT_CURRENT: "Document generation is not current",
        EvidenceErrorCode.REFERENCE_NOT_FOUND: "Candidate reference is unavailable",
        EvidenceErrorCode.RANGE_MISMATCH: "Candidate range does not match",
        EvidenceErrorCode.TEXT_DIGEST_MISMATCH: "Candidate text digest does not match",
        EvidenceErrorCode.ANCHOR_MISMATCH: "Candidate anchors do not match",
        EvidenceErrorCode.PROFILE_MISMATCH: "Evidence profile does not match",
        EvidenceErrorCode.VECTOR_UNAVAILABLE: "Vector evidence is unavailable",
        EvidenceErrorCode.VECTOR_NOT_CURRENT: "Vector generation is not current",
        EvidenceErrorCode.VECTOR_INTEGRITY: "Vector evidence is inconsistent",
        EvidenceErrorCode.INVALID_ORDER: "Candidate order is invalid",
        EvidenceErrorCode.CONTRADICTORY_DUPLICATE: "Repeated candidate assertions conflict",
    }
)


class EvidenceBundleError(EvidenceValue):
    code: EvidenceErrorCode
    input_position: InputPosition | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def message(self) -> str:
        """Fixed safe text; raw storage/caller error strings are never accepted."""
        return _SAFE_MESSAGES[self.code]


class ResolvedBundleSource(BundleSourceScope):
    """Only authoritative stored identities; no invented domain fields."""

    source_pdf_sha256: Sha256


class ContributionProvenance(EvidenceValue):
    channel: Literal["lexical", "semantic"]
    embedding_profile_id: Identifier | None = None
    vector_generation_id: Identifier | None = None
    embedding_span_id: Identifier | None = None

    @model_validator(mode="after")
    def _semantic_identity(self) -> Self:
        values = (self.embedding_profile_id, self.vector_generation_id, self.embedding_span_id)
        if self.channel == "semantic" and any(value is None for value in values):
            raise ValueError("semantic provenance identities are required")
        if self.channel == "lexical" and any(value is not None for value in values):
            raise ValueError("lexical provenance cannot carry semantic identities")
        return self


class EvidenceSourceRef(EvidenceValue):
    scope: ResolvedBundleSource
    parent: ParentIdentity
    physical_page_index: Nonnegative
    display_page_number: Positive
    printed_page_label: SourceText | None
    canonical_page_text_sha256: Sha256
    parser_profile_sha256: Sha256
    processing_profile_id: Identifier
    page_width_points: Annotated[float, Field(gt=0)]
    page_height_points: Annotated[float, Field(gt=0)]
    source_page_rotation_degrees: int
    start_offset: Nonnegative
    end_offset: Positive
    text_sha256: Sha256
    anchor_ids: AnchorIds
    provenance: Annotated[tuple[ContributionProvenance, ...], Field(min_length=1, max_length=2)]

    @model_validator(mode="after")
    def _source_shape(self) -> Self:
        if self.end_offset <= self.start_offset:
            raise ValueError("source range must be nonempty")
        if self.display_page_number != self.physical_page_index + 1:
            raise ValueError("display page number must match physical index")
        if (
            self.parent.project_id != self.scope.project_id
            or self.parent.document_generation_id != self.scope.document_generation_id
        ):
            raise ValueError("source parent must agree with scope")
        return self


class BundleAnchor(EvidenceValue):
    """Compact native location projection; deliberately excludes canonical page text."""

    page_anchor_id: Identifier
    evidence_id: Annotated[str, Field(pattern=r"^ev-sha256-[0-9a-f]{64}$")]
    char_start: Nonnegative
    char_end: Positive
    anchor_text: SourceText
    anchor_text_sha256: Sha256
    boxes_sha256: Sha256
    boxes: Annotated[tuple[PdfAnchorBox, ...], Field(min_length=1, max_length=MAX_CHUNK_ANCHORS)]

    @field_validator("boxes", mode="before")
    @classmethod
    def _strict_boxes(cls, value: object) -> object:
        # Keep the established box type/semantics, but reject coercions before
        # that older nested model can erase the original scalar types.
        if isinstance(value, (tuple, list)):
            for box in value:
                if isinstance(box, PdfAnchorBox):
                    fields = box.model_dump()
                elif isinstance(box, dict):
                    fields = box
                else:
                    raise ValueError("invalid box value")
                for name in ("char_start", "char_end"):
                    if type(fields.get(name)) is not int:
                        raise ValueError("box offsets must be strict integers")
                for name in ("x0", "top", "x1", "bottom"):
                    if type(fields.get(name)) not in (int, float):
                        raise ValueError("box coordinates must be numeric")
        return value

    @model_validator(mode="after")
    def _location_shape(self) -> Self:
        if (
            self.char_end <= self.char_start
            or len(self.anchor_text) != self.char_end - self.char_start
        ):
            raise ValueError("anchor text length must agree with its range")
        for box in self.boxes:
            if box.char_start < self.char_start or box.char_end > self.char_end:
                raise ValueError("box must lie within anchor range")
        return self


Anchors = Annotated[tuple[BundleAnchor, ...], Field(min_length=1, max_length=MAX_CHUNK_ANCHORS)]


class ResolvedEvidenceRange(EvidenceValue):
    source: EvidenceSourceRef
    text: SourceText
    anchors: Anchors
    contributions: Annotated[tuple[ContributionProvenance, ...], Field(min_length=1, max_length=2)]
    trust: Literal["untrusted_source_data"] = "untrusted_source_data"


class ResolvedContext(EvidenceValue):
    """Whole parent material before packing, with no provisional citation label."""

    parent: ParentIdentity
    physical_page_index: Nonnegative
    start_offset: Nonnegative
    end_offset: Positive
    text: SourceText
    text_sha256: Sha256
    anchors: Anchors
    citable: Disabled = False
    trust: Literal["untrusted_source_data"] = "untrusted_source_data"

    @model_validator(mode="after")
    def _range(self) -> Self:
        if self.end_offset <= self.start_offset:
            raise ValueError("context range must be nonempty")
        return self


class ResolvedCandidateGroup(EvidenceValue):
    input_position: InputPosition
    reference: EvidenceCandidateRef
    ranges: Annotated[tuple[ResolvedEvidenceRange, ...], Field(min_length=1, max_length=2)]
    context: ResolvedContext | None = None


class CoveragePage(EvidenceValue):
    page_id: Identifier
    physical_page_index: Nonnegative | None
    display_page_number: Positive
    printed_page_label: SourceText | None
    extraction_category: Literal["adequate", "low", "empty", "unknown"]
    needs_ocr: bool | None
    has_chunks: bool


class EvidenceCoverage(EvidenceValue):
    status: CoverageStatus
    pages: Annotated[tuple[CoveragePage, ...], Field(max_length=MAX_SOURCE_PAGES)]
    candidate_page_ids: Annotated[tuple[Identifier, ...], Field(max_length=MAX_CANDIDATE_GROUPS)]
    packed_page_ids: Annotated[tuple[Identifier, ...], Field(max_length=MAX_CANDIDATE_GROUPS)]
    flagged_page_ids: Annotated[tuple[Identifier, ...], Field(max_length=MAX_SOURCE_PAGES)]
    excluded_page_ids: Annotated[tuple[Identifier, ...], Field(max_length=MAX_SOURCE_PAGES)]
    document_completeness: Literal["not_assessed"] = "not_assessed"
    quality_basis: Literal["native_text_presence_heuristic"] = "native_text_presence_heuristic"


class ResolvedEvidenceInput(EvidenceValue):
    request: EvidenceBundleRequest
    resolved_scope: ResolvedBundleSource
    coverage: EvidenceCoverage
    groups: Annotated[tuple[ResolvedCandidateGroup, ...], Field(max_length=MAX_CANDIDATE_GROUPS)]


class EvidenceBundleEntry(ResolvedEvidenceRange):
    citation_label: CitationLabel


class EvidenceContext(ResolvedContext):
    related_citation_labels: Annotated[tuple[CitationLabel, ...], Field(min_length=1, max_length=2)]


class EvidenceOmission(EvidenceValue):
    reason: OmissionReason
    duplicate_of_input_position: InputPosition | None = None
    violated_constraints: tuple[Literal["preview_bytes_exceeded", "entry_limit_exceeded"], ...] = ()


class CandidateDisposition(EvidenceValue):
    input_position: InputPosition
    reference: EvidenceCandidateRef
    citation_labels: Annotated[tuple[CitationLabel, ...], Field(max_length=2)] = ()
    omission: EvidenceOmission | None = None


class EvidenceBudgetUsage(PreviewBudget):
    used_content_bytes: Nonnegative
    used_entries: Annotated[int, Field(strict=True, ge=0, le=MAX_ENTRIES)]


class ContentEvidence(EvidenceValue):
    citation_label: CitationLabel
    physical_page_index: Nonnegative
    text: SourceText
    trust: Literal["untrusted_source_data"] = "untrusted_source_data"


class ContentContext(EvidenceValue):
    related_citation_labels: Annotated[tuple[CitationLabel, ...], Field(min_length=1, max_length=2)]
    physical_page_index: Nonnegative
    text: SourceText
    trust: Literal["untrusted_source_data"] = "untrusted_source_data"
    citable: Disabled = False


class ContentPreview(EvidenceValue):
    concern_profile_id: Literal["stated-study-objective-v1"] = "stated-study-objective-v1"
    context: Annotated[tuple[ContentContext, ...], Field(max_length=MAX_CANDIDATE_GROUPS)]
    evidence: Annotated[tuple[ContentEvidence, ...], Field(max_length=MAX_ENTRIES)]


class EvidenceBundlePayload(EvidenceValue):
    scope: ResolvedBundleSource
    origin: CandidateOrigin
    concern_profile_id: Literal["stated-study-objective-v1"] = "stated-study-objective-v1"
    selection_mode: Literal["explicit_candidates"] = "explicit_candidates"
    selected_source_retrieval_completeness: Literal["not_assessed"] = "not_assessed"
    entries: Annotated[tuple[EvidenceBundleEntry, ...], Field(max_length=MAX_ENTRIES)]
    context: Annotated[tuple[EvidenceContext, ...], Field(max_length=MAX_CANDIDATE_GROUPS)]
    coverage: EvidenceCoverage
    dispositions: Annotated[
        tuple[CandidateDisposition, ...], Field(max_length=MAX_CANDIDATE_GROUPS)
    ]
    budget: EvidenceBudgetUsage
    state: PreviewState
    insufficient_reason: InsufficientReason | None = None
    generation_enabled: Disabled = False
    generation_readiness: Literal["not_assessed"] = "not_assessed"
    support_assessment: Literal["not_performed"] = "not_performed"
    generation_token_count: None = None
    content_preview: ContentPreview

    @field_validator("generation_enabled", mode="before")
    @classmethod
    def _disabled(cls, value: object) -> object:
        return _false(value)


class EvidenceBundle(EvidenceBundlePayload):
    """A supplied fingerprint value, never a computed field or sealing operation."""

    fingerprint: Sha256
