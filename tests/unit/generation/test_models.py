from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from academic_chatbot.generation.models import (
    AnswerCitation,
    LocalGenerationAbstained,
    LocalGenerationAbstentionReason,
    LocalGenerationAnswered,
    LocalGenerationFailed,
    LocalGenerationFailureCode,
    LocalGenerationRejected,
    LocalGenerationRejectionCode,
    LocalGenerationResult,
    MvpCitedAnswer,
)


def citation(label: str = "E1") -> AnswerCitation:
    return AnswerCitation(
        citation_label=label,
        paper_id="paper-1",
        file_version_id="file-1",
        document_generation_id="generation-1",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        page_id="page-1",
        start_offset=0,
        end_offset=10,
        text_sha256="a" * 64,
        anchor_ids=("page-anchor-sha256-" + "b" * 64,),
    )


def test_cited_answer_accepts_only_strict_nonblank_unique_labels() -> None:
    answer = MvpCitedAnswer(answer="English answer.", citation_labels=("E1", "E2"))
    assert answer.model_dump(mode="json") == {
        "answer": "English answer.",
        "citation_labels": ["E1", "E2"],
    }
    with pytest.raises(ValidationError):
        MvpCitedAnswer(answer=" ", citation_labels=("E1",))
    with pytest.raises(ValidationError):
        MvpCitedAnswer(answer="English answer.", citation_labels=("bad",))
    with pytest.raises(ValidationError):
        MvpCitedAnswer(answer="English answer.", citation_labels=("E1", "E1"))
    with pytest.raises(ValidationError):
        MvpCitedAnswer(answer="English answer.", citation_labels=())
    with pytest.raises(ValidationError):
        MvpCitedAnswer(
            answer="English answer.",
            citation_labels=tuple(f"E{index}" for index in range(1, 202)),
        )
    with pytest.raises(ValidationError):
        MvpCitedAnswer.model_validate(
            {"answer": "English answer.", "citation_labels": ("E1",), "extra": True}
        )


def test_cited_answer_is_frozen() -> None:
    answer = MvpCitedAnswer(answer="English answer.", citation_labels=("E1",))
    with pytest.raises(ValidationError):
        answer.answer = "Changed"  # type: ignore[misc]


def test_answered_result_requires_ordered_citation_projection() -> None:
    result = LocalGenerationAnswered(
        answer="English answer.",
        citation_labels=("E1",),
        citations=(citation(),),
        bundle_fingerprint="c" * 64,
        generation_profile_id="mvp-qwen3-8b-b10007-cuda-v1",
        model_profile_id="qwen3-8b-q4-k-m",
        rendered_prompt_tokens=100,
        completion_tokens=20,
    )
    assert result.outcome == "answered"
    assert result.model_dump(mode="json")["citations"][0]["citation_label"] == "E1"
    with pytest.raises(ValidationError):
        LocalGenerationAnswered(
            answer="English answer.",
            citation_labels=("E2",),
            citations=(citation("E1"),),
            bundle_fingerprint="c" * 64,
            generation_profile_id="mvp-qwen3-8b-b10007-cuda-v1",
            model_profile_id="qwen3-8b-q4-k-m",
            rendered_prompt_tokens=100,
            completion_tokens=20,
        )


def test_abstained_result_has_only_frozen_reason() -> None:
    result = LocalGenerationAbstained(
        reason=LocalGenerationAbstentionReason.EVIDENCE_PARTIAL
    )
    assert result.model_dump(mode="json") == {
        "schema_version": "mvp-local-generation-result-v1",
        "outcome": "abstained",
        "reason": "evidence_partial",
    }


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (LocalGenerationRejectionCode.INVALID_REQUEST, "Generation request is invalid"),
        (
            LocalGenerationRejectionCode.CONTEXT_LIMIT_EXCEEDED,
            "Rendered request exceeds the context limit",
        ),
        (
            LocalGenerationRejectionCode.MODEL_OUTPUT_INVALID,
            "Model output is invalid",
        ),
    ],
)
def test_rejected_messages_are_fixed_and_cannot_accept_error_text(
    code: LocalGenerationRejectionCode, message: str
) -> None:
    result = LocalGenerationRejected(code=code)
    assert result.message == message
    with pytest.raises(ValidationError):
        LocalGenerationRejected.model_validate(
            {"code": code, "message": "C:\\private\\model.gguf"}
        )


def test_failed_message_is_fixed_and_accepts_no_exception_text() -> None:
    result = LocalGenerationFailed(code=LocalGenerationFailureCode.GENERATION_TIMEOUT)
    assert result.model_dump(mode="json") == {
        "schema_version": "mvp-local-generation-result-v1",
        "outcome": "failed",
        "code": "GENERATION_TIMEOUT",
        "message": "Local generation timed out",
    }
    with pytest.raises(ValidationError):
        LocalGenerationFailed.model_validate(
            {"code": "GENERATION_TIMEOUT", "message": "secret exception"}
        )


def test_result_union_is_discriminated_without_analysis_status() -> None:
    value = TypeAdapter(LocalGenerationResult).validate_json(
        '{"outcome":"failed","code":"INTERNAL_INVARIANT_FAILURE"}'
    )
    assert isinstance(value, LocalGenerationFailed)
    serialized = value.model_dump(mode="json")
    assert set(serialized) == {"schema_version", "outcome", "code", "message"}
    assert not {"supported", "inferred", "not_reported", "conflicting", "unreadable"} & set(
        serialized
    )
