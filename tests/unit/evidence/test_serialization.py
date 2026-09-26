"""Independent canonical byte/hash oracles; no bundle sealing here."""

from hashlib import sha256
from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "value,expected,digest",
    [
        ({}, b"{}", "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"),
        (
            {"b": [2, 1], "a": None},
            b'{"a":null,"b":[2,1]}',
            "64befe554fb7858d5aedd620f047ac02bca15dfe6f2f13dc790b1d18d532ef8c",
        ),
        (
            {"evidence": [], "context": [], "concern_profile_id": "stated-study-objective-v1"},
            b'{"concern_profile_id":"stated-study-objective-v1","context":[],"evidence":[]}',
            "a741f66f6c5920488b51ee561db23ade275b51c5c938b6e2d7e0cab737101ed6",
        ),
    ],
)
def test_canonical_json_exact_bytes(value, expected, digest):
    encode = import_module("academic_chatbot.evidence.serialization").canonical_json_bytes
    assert encode(value) == expected
    assert sha256(encode(value)).hexdigest() == digest
    assert not encode(value).endswith(b"\n")
    assert encode(value) == encode(value)


def test_canonical_json_preserves_unicode_and_array_order():
    encode = import_module("academic_chatbot.evidence.serialization").canonical_json_bytes
    assert encode({"é": ["e\u0301", "é", '\n"']}) == '{"é":["é","é","\\n\\""]}'.encode()
    assert encode([1, 2]) != encode([2, 1])


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), {"bad": "\ud800"}, {1: "non-string-key"}, object()],
)
def test_noncanonical_values_rejected(value):
    encode = import_module("academic_chatbot.evidence.serialization").canonical_json_bytes
    with pytest.raises((ValueError, TypeError, UnicodeError)):
        encode(value)
