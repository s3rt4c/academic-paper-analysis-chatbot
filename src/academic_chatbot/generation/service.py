"""Fail-closed generation orchestration over an already-built evidence bundle."""

from __future__ import annotations

import json

from pydantic import ValidationError

from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.resolver import EvidenceReadResolver, EvidenceResolutionError
from academic_chatbot.evidence.serialization import (
    EvidencePreparationError,
    canonical_bundle_bytes,
    canonical_json_bytes,
)
from academic_chatbot.evidence.service import EvidenceBundleService
from academic_chatbot.generation.context import render_generation_context
from academic_chatbot.generation.llama_cpp import (
    ContextLimitExceeded,
    GenerationCancelled,
    GenerationInvariantFailure,
    GenerationProcessFailure,
    GenerationRuntimeUnavailable,
    GenerationTimeout,
)
from academic_chatbot.generation.models import (
    AnswerCitation,
    LocalGenerationAbstained,
    LocalGenerationAbstentionReason,
    LocalGenerationAnswered,
    LocalGenerationFailed,
    LocalGenerationFailureCode,
    LocalGenerationRejected,
    LocalGenerationRejectionCode,
    LocalGenerationRequest,
    LocalGenerationResult,
    MvpCitedAnswer,
)
from academic_chatbot.ports.model import CancellationSignal, StructuredLocalModel


def _rejected(code: LocalGenerationRejectionCode) -> LocalGenerationRejected:
    return LocalGenerationRejected(code=code)


def _authority_error(code: m.EvidenceErrorCode) -> LocalGenerationResult:
    if code in {
        m.EvidenceErrorCode.GENERATION_NOT_CURRENT,
        m.EvidenceErrorCode.VECTOR_NOT_CURRENT,
    }:
        return _rejected(LocalGenerationRejectionCode.SOURCE_NOT_CURRENT)
    if code in {
        m.EvidenceErrorCode.STORAGE_UNAVAILABLE,
        m.EvidenceErrorCode.VECTOR_UNAVAILABLE,
    }:
        return LocalGenerationFailed(
            code=LocalGenerationFailureCode.AUTHORITY_UNAVAILABLE
        )
    if code == m.EvidenceErrorCode.RESOURCE_LIMIT:
        return _rejected(LocalGenerationRejectionCode.RESOURCE_LIMIT)
    return _rejected(LocalGenerationRejectionCode.SOURCE_NOT_CURRENT)


def _rebuild_request(bundle: m.EvidenceBundle) -> m.EvidenceBundleRequest | None:
    dispositions = tuple(sorted(bundle.dispositions, key=lambda item: item.input_position))
    if tuple(item.input_position for item in dispositions) != tuple(range(len(dispositions))):
        return None
    return m.EvidenceBundleRequest(
        scope=m.BundleSourceScope(
            **bundle.scope.model_dump(exclude={"source_pdf_sha256"})
        ),
        concern_profile_id=bundle.concern_profile_id,
        origin=bundle.origin,
        candidates=tuple(item.reference for item in dispositions),
        preview_budget=m.PreviewBudget(
            profile_id=bundle.budget.profile_id,
            max_content_bytes=bundle.budget.max_content_bytes,
            max_entries=bundle.budget.max_entries,
        ),
    )


def _citation(entry: m.EvidenceBundleEntry) -> AnswerCitation:
    source = entry.source
    return AnswerCitation(
        citation_label=entry.citation_label,
        paper_id=source.scope.paper_id,
        file_version_id=source.scope.file_version_id,
        document_generation_id=source.scope.document_generation_id,
        physical_page_index=source.physical_page_index,
        display_page_number=source.display_page_number,
        printed_page_label=source.printed_page_label,
        page_id=source.parent.page_id,
        start_offset=source.start_offset,
        end_offset=source.end_offset,
        text_sha256=source.text_sha256,
        anchor_ids=source.anchor_ids,
    )


def _cancelled(cancel: CancellationSignal) -> bool | None:
    try:
        value = cancel.is_set()
    except Exception:
        return None
    return value if type(value) is bool else None


def _strict_model_output(content: str) -> MvpCitedAnswer:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    parsed = json.loads(content, object_pairs_hook=unique_object)
    if type(parsed) is not dict:
        raise ValueError("model output must be one JSON object")
    return MvpCitedAnswer.model_validate_json(canonical_json_bytes(parsed), strict=True)


class LocalGenerationService:
    """Generate only from a ready bundle and release only after authority rebuild."""

    def __init__(
        self, *, resolver: EvidenceReadResolver, model: StructuredLocalModel
    ) -> None:
        self._resolver = resolver
        self._model = model

    def generate(
        self,
        *,
        question: str,
        bundle: m.EvidenceBundle,
        cancel: CancellationSignal,
    ) -> LocalGenerationResult:
        cancellation_state = _cancelled(cancel)
        if cancellation_state is None:
            return LocalGenerationFailed(
                code=LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE
            )
        if cancellation_state:
            return LocalGenerationAbstained(
                reason=LocalGenerationAbstentionReason.CANCELLED
            )
        try:
            admitted = LocalGenerationRequest(question=question, bundle=bundle)
        except (ValidationError, TypeError, ValueError):
            return _rejected(LocalGenerationRejectionCode.INVALID_REQUEST)
        try:
            canonical_bundle_bytes(admitted.bundle)
        except EvidencePreparationError as error:
            code = (
                LocalGenerationRejectionCode.RESOURCE_LIMIT
                if error.error.code == m.EvidenceErrorCode.RESOURCE_LIMIT
                else LocalGenerationRejectionCode.BUNDLE_FINGERPRINT_INVALID
            )
            return _rejected(code)
        except Exception:
            return _rejected(LocalGenerationRejectionCode.BUNDLE_FINGERPRINT_INVALID)

        if admitted.bundle.state == m.PreviewState.PREVIEW_PARTIAL:
            return LocalGenerationAbstained(
                reason=LocalGenerationAbstentionReason.EVIDENCE_PARTIAL
            )
        if admitted.bundle.state == m.PreviewState.INSUFFICIENT_EVIDENCE:
            return LocalGenerationAbstained(
                reason=LocalGenerationAbstentionReason.EVIDENCE_INSUFFICIENT
            )
        if not admitted.bundle.entries:
            return _rejected(LocalGenerationRejectionCode.INVALID_REQUEST)

        try:
            rendered = render_generation_context(
                question=admitted.question, bundle=admitted.bundle
            )
            generated = self._model.generate(rendered.request, cancel=cancel)
        except GenerationCancelled:
            return LocalGenerationAbstained(
                reason=LocalGenerationAbstentionReason.CANCELLED
            )
        except ContextLimitExceeded:
            return _rejected(LocalGenerationRejectionCode.CONTEXT_LIMIT_EXCEEDED)
        except GenerationTimeout:
            return LocalGenerationFailed(
                code=LocalGenerationFailureCode.GENERATION_TIMEOUT
            )
        except GenerationRuntimeUnavailable:
            return LocalGenerationFailed(
                code=LocalGenerationFailureCode.RUNTIME_UNAVAILABLE
            )
        except GenerationProcessFailure:
            return LocalGenerationFailed(
                code=LocalGenerationFailureCode.MODEL_PROCESS_FAILURE
            )
        except (GenerationInvariantFailure, MemoryError):
            return LocalGenerationFailed(
                code=LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE
            )
        except Exception:
            return LocalGenerationFailed(
                code=LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE
            )
        cancellation_state = _cancelled(cancel)
        if cancellation_state is None:
            return LocalGenerationFailed(
                code=LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE
            )
        if cancellation_state:
            return LocalGenerationAbstained(
                reason=LocalGenerationAbstentionReason.CANCELLED
            )
        try:
            answer = _strict_model_output(generated.content)
        except (
            json.JSONDecodeError,
            RecursionError,
            ValidationError,
            ValueError,
            TypeError,
        ):
            return _rejected(LocalGenerationRejectionCode.MODEL_OUTPUT_INVALID)

        entries: list[m.EvidenceBundleEntry] = []
        for label in answer.citation_labels:
            matches = tuple(
                entry
                for entry in admitted.bundle.entries
                if entry.citation_label == label
            )
            if len(matches) != 1:
                return _rejected(LocalGenerationRejectionCode.CITATION_REFERENCE_INVALID)
            entries.append(matches[0])

        try:
            request = _rebuild_request(admitted.bundle)
            if request is None:
                return LocalGenerationFailed(
                    code=LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE
                )
            rebuilt = EvidenceBundleService(resolver=self._resolver).build(request)
            if (
                rebuilt.fingerprint != admitted.bundle.fingerprint
                or canonical_bundle_bytes(rebuilt) != canonical_bundle_bytes(admitted.bundle)
            ):
                return _rejected(LocalGenerationRejectionCode.SOURCE_NOT_CURRENT)
        except EvidenceResolutionError as error:
            return _authority_error(error.error.code)
        except EvidencePreparationError as error:
            return _authority_error(error.error.code)
        except Exception:
            return LocalGenerationFailed(
                code=LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE
            )

        cancellation_state = _cancelled(cancel)
        if cancellation_state is None:
            return LocalGenerationFailed(
                code=LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE
            )
        if cancellation_state:
            return LocalGenerationAbstained(
                reason=LocalGenerationAbstentionReason.CANCELLED
            )
        try:
            return LocalGenerationAnswered(
                answer=answer.answer,
                citation_labels=answer.citation_labels,
                citations=tuple(_citation(entry) for entry in entries),
                bundle_fingerprint=admitted.bundle.fingerprint,
                rendered_prompt_tokens=generated.prompt_tokens,
                completion_tokens=generated.completion_tokens,
            )
        except (ValidationError, ValueError, TypeError):
            return LocalGenerationFailed(
                code=LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE
            )
