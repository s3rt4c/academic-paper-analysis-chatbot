"""Bounded canonical wire encoding, with one terminal CLI LF reserved."""

from __future__ import annotations

import json

from pydantic import TypeAdapter, ValidationError

from academic_chatbot.evidence.models import MAX_AUDIT_BYTES
from academic_chatbot.retrieval.scoped_models import (
    ScopedRetrievalError,
    ScopedRetrievalResult,
)

MAX_OUTPUT_BYTES = MAX_AUDIT_BYTES


def canonical_scoped_result_bytes(result: ScopedRetrievalResult) -> bytes:
    try:
        value: ScopedRetrievalResult = TypeAdapter(ScopedRetrievalResult).validate_python(result)
        encoder = json.JSONEncoder(
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        output = bytearray()
        for fragment in encoder.iterencode(value.model_dump(mode="json")):
            for start in range(0, len(fragment), 4096):
                block = fragment[start : start + 4096].encode("utf-8", errors="strict")
                if len(output) + len(block) + 1 > MAX_OUTPUT_BYTES:
                    raise ScopedRetrievalError("RESOURCE_LIMIT")
                output.extend(block)
        return bytes(output)
    except (ValidationError, UnicodeError, ValueError) as error:
        if isinstance(error, ScopedRetrievalError):
            raise
        raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE") from None


def canonical_scoped_error_bytes(error: ScopedRetrievalError) -> bytes:
    return json.dumps(
        {
            "schema_version": "source-scoped-search-error-v1",
            "error": {"code": error.code.value, "message": error.message},
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
