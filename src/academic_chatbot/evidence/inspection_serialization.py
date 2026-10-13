"""Bounded canonical inspection JSON; output admission reserves one terminal LF."""

from __future__ import annotations

import json

from pydantic import TypeAdapter, ValidationError

from academic_chatbot.evidence.inspection_models import (
    EvidenceInspectionRequest,
    EvidenceInspectionResult,
)
from academic_chatbot.evidence.models import MAX_AUDIT_BYTES, MAX_REQUEST_BYTES
from academic_chatbot.retrieval.scoped_models import ScopedRetrievalError

MAX_RESULT_BYTES = MAX_AUDIT_BYTES


def _bounded_bytes(value: object, maximum: int) -> bytes:
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    result = bytearray()
    for token in encoder.iterencode(value):
        for start in range(0, len(token), 4096):
            block = token[start : start + 4096].encode("utf-8", errors="strict")
            if len(result) + len(block) > maximum:
                raise ScopedRetrievalError("RESOURCE_LIMIT")
            result.extend(block)
    return bytes(result)


def canonical_inspection_request_bytes(request: EvidenceInspectionRequest) -> bytes:
    try:
        admitted = EvidenceInspectionRequest.model_validate(request)
        return _bounded_bytes(admitted.model_dump(mode="json"), MAX_REQUEST_BYTES)
    except ScopedRetrievalError:
        raise
    except (ValidationError, ValueError, TypeError, UnicodeError):
        raise ScopedRetrievalError("INVALID_REQUEST") from None


def canonical_inspection_result_bytes(result: EvidenceInspectionResult) -> bytes:
    try:
        admitted: EvidenceInspectionResult = TypeAdapter(EvidenceInspectionResult).validate_python(
            result
        )
        return _bounded_bytes(admitted.model_dump(mode="json"), MAX_RESULT_BYTES - 1)
    except ScopedRetrievalError:
        raise
    except (ValidationError, ValueError, TypeError, UnicodeError):
        raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE") from None
