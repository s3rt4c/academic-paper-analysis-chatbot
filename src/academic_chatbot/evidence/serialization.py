"""Canonical JSON primitives only; no request I/O, packing or bundle sealing."""

from __future__ import annotations

import json
import math


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
