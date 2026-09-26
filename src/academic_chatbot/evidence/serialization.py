"""Canonical JSON primitives and bounded in-memory evidence bundle sealing."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from typing import BinaryIO

from academic_chatbot.evidence import models as m


class EvidencePreparationError(ValueError):
    """Safe structured preparation failure, independent of storage resolution."""

    def __init__(self, code: m.EvidenceErrorCode, input_position: int | None = None) -> None:
        self.error = m.EvidenceBundleError(code=code, input_position=input_position)
        super().__init__(self.error.message)


def read_request_bytes(stream: BinaryIO) -> bytes:
    """Read bounded chunks, consuming at most one byte beyond the request cap."""
    result = bytearray()
    while len(result) <= m.MAX_REQUEST_BYTES:
        fragment = stream.read(min(65536, m.MAX_REQUEST_BYTES + 1 - len(result)))
        if not fragment:
            return bytes(result)
        result.extend(fragment)
    raise EvidencePreparationError(m.EvidenceErrorCode.RESOURCE_LIMIT)


def _check_request_depth(text: str) -> None:
    """Admit container depth before JSON allocation, ignoring quoted delimiters."""
    stack: list[str] = []
    quoted = False
    escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            if len(stack) == m.MAX_JSON_DEPTH:
                raise EvidencePreparationError(m.EvidenceErrorCode.RESOURCE_LIMIT)
            stack.append(char)
        elif char in "]}":
            if not stack or stack.pop() != ("[" if char == "]" else "{"):
                raise EvidencePreparationError(m.EvidenceErrorCode.INVALID_REQUEST)


def _unique_request_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_request_constant(value: str) -> object:
    raise ValueError("non-finite JSON number")


def parse_bundle_request(raw: bytes) -> m.EvidenceBundleRequest:
    """Admit bounded strict JSON, then delegate all request semantics to the model."""
    if len(raw) > m.MAX_REQUEST_BYTES:
        raise EvidencePreparationError(m.EvidenceErrorCode.RESOURCE_LIMIT)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError:
        raise EvidencePreparationError(m.EvidenceErrorCode.INVALID_REQUEST) from None
    _check_request_depth(text)
    try:
        structural = json.loads(
            text,
            object_pairs_hook=_unique_request_object,
            parse_constant=_reject_request_constant,
        )
        _validate_json(structural)
        # JSON validation preserves strict scalar types while admitting JSON arrays
        # for the immutable tuple fields; Python-mode validation rejects those lists.
        return m.EvidenceBundleRequest.model_validate_json(text)
    except (ValueError, TypeError):
        raise EvidencePreparationError(m.EvidenceErrorCode.INVALID_REQUEST) from None


def _validate_json(value: object) -> None:
    if value is None or type(value) in (bool, int):
        return
    if isinstance(value, str):
        value.encode("utf-8", errors="strict")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            _validate_json(key)
            _validate_json(item)
        return
    raise TypeError("unsupported JSON value")


def canonical_json_bytes(payload: object) -> bytes:
    """Encode explicit JSON values without normalization, coercion, hashing or LF.

    Callers project typed models with model_dump(mode='json'); arbitrary objects
    do not acquire implicit serializers. Array order and explicit nulls survive.
    """
    _validate_json(payload)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _bounded_canonical_bytes(value: object) -> bytes:
    """Collect at most cap-minus-LF bytes; never encode an unlimited string.

    iterencode may yield one large string token. Slice every token before UTF-8
    encoding so even that case allocates only a small bounded byte fragment.
    Nothing is exposed until the entire canonical object passes admission.
    """
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    result = bytearray()
    limit = m.MAX_AUDIT_BYTES - 1
    for token in encoder.iterencode(value):
        for offset in range(0, len(token), 4096):
            fragment = token[offset : offset + 4096].encode("utf-8", errors="strict")
            if len(result) + len(fragment) > limit:
                raise EvidencePreparationError(m.EvidenceErrorCode.RESOURCE_LIMIT)
            result.extend(fragment)
    return bytes(result)


def canonical_payload_bytes(payload: m.EvidenceBundlePayload) -> bytes:
    """Canonical audit payload, excluding fingerprint and terminal LF."""
    return _bounded_canonical_bytes(payload.model_dump(mode="json", exclude={"fingerprint"}))


def compute_bundle_fingerprint(payload: m.EvidenceBundlePayload) -> str:
    """Hash the complete payload, never the sealed bundle or a terminal LF."""
    return sha256(canonical_payload_bytes(payload)).hexdigest()


def seal_bundle(payload: m.EvidenceBundlePayload) -> m.EvidenceBundle:
    """Add the deterministic payload fingerprint without recursive serialization."""
    fingerprint = compute_bundle_fingerprint(payload)
    return m.EvidenceBundle.model_validate(
        {**payload.model_dump(mode="python", exclude={"fingerprint"}), "fingerprint": fingerprint}
    )


def canonical_bundle_bytes(bundle: m.EvidenceBundle) -> bytes:
    """Validate fingerprint and admit full audit JSON with one future LF reserved."""
    payload = m.EvidenceBundlePayload.model_validate(
        {name: getattr(bundle, name) for name in m.EvidenceBundlePayload.model_fields}
    )
    if bundle.fingerprint != compute_bundle_fingerprint(payload):
        raise EvidencePreparationError(m.EvidenceErrorCode.STORAGE_INTEGRITY)
    return _bounded_canonical_bytes(bundle.model_dump(mode="json"))
