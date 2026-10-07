"""Strict public values for bounded local generation."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, computed_field, field_validator, model_validator

from academic_chatbot.evidence import models as m

MVP_PROMPT_PROFILE_ID = "mvp-local-generation-prompt-v1"
MVP_OUTPUT_SCHEMA_ID = "mvp-cited-answer-v1"
MVP_RESULT_SCHEMA_ID = "mvp-local-generation-result-v1"
MVP_GENERATION_PROFILE_ID = "mvp-qwen3-8b-b10007-cuda-v1"
MVP_MODEL_PROFILE_ID = "qwen3-8b-q4-k-m"


class LocalGenerationAbstentionReason(StrEnum):
    EVIDENCE_PARTIAL = "evidence_partial"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    CANCELLED = "cancelled"


class LocalGenerationRejectionCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    SOURCE_SCOPE_AMBIGUOUS = "SOURCE_SCOPE_AMBIGUOUS"
    BUNDLE_FINGERPRINT_INVALID = "BUNDLE_FINGERPRINT_INVALID"
    SOURCE_NOT_CURRENT = "SOURCE_NOT_CURRENT"
    CITATION_REFERENCE_INVALID = "CITATION_REFERENCE_INVALID"
    CONTEXT_LIMIT_EXCEEDED = "CONTEXT_LIMIT_EXCEEDED"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


class LocalGenerationFailureCode(StrEnum):
    AUTHORITY_UNAVAILABLE = "AUTHORITY_UNAVAILABLE"
    RUNTIME_UNAVAILABLE = "RUNTIME_UNAVAILABLE"
    GENERATION_TIMEOUT = "GENERATION_TIMEOUT"
    MODEL_PROCESS_FAILURE = "MODEL_PROCESS_FAILURE"
    INTERNAL_INVARIANT_FAILURE = "INTERNAL_INVARIANT_FAILURE"


_REJECTION_MESSAGES = {
    LocalGenerationRejectionCode.INVALID_REQUEST: "Generation request is invalid",
    LocalGenerationRejectionCode.SOURCE_SCOPE_AMBIGUOUS: (
        "Selected paper source is ambiguous"
    ),
    LocalGenerationRejectionCode.BUNDLE_FINGERPRINT_INVALID: (
        "Evidence bundle fingerprint is invalid"
    ),
    LocalGenerationRejectionCode.SOURCE_NOT_CURRENT: "Evidence source is not current",
    LocalGenerationRejectionCode.CITATION_REFERENCE_INVALID: (
        "Answer citation is invalid"
    ),
    LocalGenerationRejectionCode.CONTEXT_LIMIT_EXCEEDED: (
        "Rendered request exceeds the context limit"
    ),
    LocalGenerationRejectionCode.MODEL_OUTPUT_INVALID: "Model output is invalid",
    LocalGenerationRejectionCode.RESOURCE_LIMIT: (
        "Generation resource limit exceeded"
    ),
}

_FAILURE_MESSAGES = {
    LocalGenerationFailureCode.AUTHORITY_UNAVAILABLE: (
        "Evidence authority is unavailable"
    ),
    LocalGenerationFailureCode.RUNTIME_UNAVAILABLE: (
        "Local generation runtime is unavailable"
    ),
    LocalGenerationFailureCode.GENERATION_TIMEOUT: "Local generation timed out",
    LocalGenerationFailureCode.MODEL_PROCESS_FAILURE: (
        "Local generation process failed"
    ),
    LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE: (
        "Local generation failed safely"
    ),
}


class MvpCitedAnswer(m.EvidenceValue):
    answer: m.SourceText
    citation_labels: Annotated[
        tuple[m.CitationLabel, ...], Field(min_length=1, max_length=m.MAX_ENTRIES)
    ]

    @field_validator("answer")
    @classmethod
    def _nonblank_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must be nonblank")
        return value

    @model_validator(mode="after")
    def _unique_labels(self) -> Self:
        if len(set(self.citation_labels)) != len(self.citation_labels):
            raise ValueError("citation labels must be unique")
        return self


class AnswerCitation(m.EvidenceValue):
    citation_label: m.CitationLabel
    paper_id: m.Identifier
    file_version_id: m.Identifier
    document_generation_id: m.Identifier
    physical_page_index: m.Nonnegative
    display_page_number: m.Positive
    printed_page_label: m.SourceText | None
    page_id: m.Identifier
    start_offset: m.Nonnegative
    end_offset: m.Positive
    text_sha256: m.Sha256
    anchor_ids: m.AnchorIds

    @model_validator(mode="after")
    def _valid_location(self) -> Self:
        if self.display_page_number != self.physical_page_index + 1:
            raise ValueError("display page number must match physical page index")
        if self.end_offset <= self.start_offset:
            raise ValueError("citation range must be nonempty")
        return self


class LocalGenerationRequest(m.EvidenceValue):
    question: m.SourceText
    bundle: m.EvidenceBundle

    @field_validator("question")
    @classmethod
    def _valid_question(cls, value: str) -> str:
        if not value.strip() or len(value.encode("utf-8")) > 4096:
            raise ValueError("question is invalid")
        return value


class LocalGenerationAnswered(m.EvidenceValue):
    schema_version: Literal["mvp-local-generation-result-v1"] = (
        "mvp-local-generation-result-v1"
    )
    outcome: Literal["answered"] = "answered"
    answer: m.SourceText
    citation_labels: Annotated[
        tuple[m.CitationLabel, ...], Field(min_length=1, max_length=m.MAX_ENTRIES)
    ]
    citations: Annotated[
        tuple[AnswerCitation, ...], Field(min_length=1, max_length=m.MAX_ENTRIES)
    ]
    bundle_fingerprint: m.Sha256
    generation_profile_id: Literal["mvp-qwen3-8b-b10007-cuda-v1"] = (
        "mvp-qwen3-8b-b10007-cuda-v1"
    )
    model_profile_id: Literal["qwen3-8b-q4-k-m"] = "qwen3-8b-q4-k-m"
    rendered_prompt_tokens: m.Positive
    completion_tokens: m.Positive

    @field_validator("answer")
    @classmethod
    def _nonblank_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must be nonblank")
        return value

    @model_validator(mode="after")
    def _citation_projection_matches(self) -> Self:
        if len(set(self.citation_labels)) != len(self.citation_labels):
            raise ValueError("citation labels must be unique")
        if tuple(value.citation_label for value in self.citations) != self.citation_labels:
            raise ValueError("citations must match citation labels in order")
        return self


class LocalGenerationAbstained(m.EvidenceValue):
    schema_version: Literal["mvp-local-generation-result-v1"] = (
        "mvp-local-generation-result-v1"
    )
    outcome: Literal["abstained"] = "abstained"
    reason: LocalGenerationAbstentionReason


class LocalGenerationRejected(m.EvidenceValue):
    schema_version: Literal["mvp-local-generation-result-v1"] = (
        "mvp-local-generation-result-v1"
    )
    outcome: Literal["rejected"] = "rejected"
    code: LocalGenerationRejectionCode

    @computed_field  # type: ignore[prop-decorator]
    @property
    def message(self) -> str:
        return _REJECTION_MESSAGES[self.code]


class LocalGenerationFailed(m.EvidenceValue):
    schema_version: Literal["mvp-local-generation-result-v1"] = (
        "mvp-local-generation-result-v1"
    )
    outcome: Literal["failed"] = "failed"
    code: LocalGenerationFailureCode

    @computed_field  # type: ignore[prop-decorator]
    @property
    def message(self) -> str:
        return _FAILURE_MESSAGES[self.code]


LocalGenerationResult = Annotated[
    LocalGenerationAnswered
    | LocalGenerationAbstained
    | LocalGenerationRejected
    | LocalGenerationFailed,
    Field(discriminator="outcome"),
]
