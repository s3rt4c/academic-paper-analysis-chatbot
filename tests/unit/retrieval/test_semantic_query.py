from __future__ import annotations

import unicodedata
from dataclasses import FrozenInstanceError

import pytest

from academic_chatbot.retrieval.semantic_query import (
    CANONICAL_CONCERN_TEXT,
    CONCERN_REGISTRY,
    MAX_CANONICAL_CONCERN_TEXT_BYTES,
    MAX_CONCERN_ID_BYTES,
    MAX_CONCERN_ID_CODEPOINTS,
    MAX_REGISTRY_ENTRIES,
    SEMANTIC_CONCERN_QUERY_POLICY_ID,
    SemanticQueryPolicyError,
    SemanticQuerySelection,
    resolve_semantic_query,
)


def test_supported_concern_has_exact_frozen_registry_identity_and_text() -> None:
    assert len(CONCERN_REGISTRY) == 1
    assert len(CONCERN_REGISTRY) <= MAX_REGISTRY_ENTRIES
    entry = CONCERN_REGISTRY[0]

    assert entry.concern_id == "stated-study-objective-v1"
    assert entry.policy_id == SEMANTIC_CONCERN_QUERY_POLICY_ID
    assert entry.policy_id == "semantic-concern-query-v1"
    assert entry.canonical_text == (
        "A study objective states what the study aims to investigate, evaluate, "
        "establish, or develop."
    )
    assert entry.canonical_text == CANONICAL_CONCERN_TEXT
    assert entry.canonical_text.encode("utf-8") == (
        b"A study objective states what the study aims to investigate, evaluate, "
        b"establish, or develop."
    )
    assert len(entry.canonical_text.encode("utf-8")) == 93
    assert len(entry.canonical_text.encode("utf-8")) <= MAX_CANONICAL_CONCERN_TEXT_BYTES
    assert unicodedata.normalize("NFC", entry.canonical_text) == entry.canonical_text


def test_repeated_concern_resolution_is_byte_identical_and_immutable() -> None:
    selection = SemanticQuerySelection(
        mode="concern", concern_id="stated-study-objective-v1"
    )

    first = resolve_semantic_query("Which result is reported?", selection)
    second = resolve_semantic_query("What does the article say?", selection)

    assert first == second
    assert first.policy_id == "semantic-concern-query-v1"
    assert first.concern_id == "stated-study-objective-v1"
    assert first.text.encode("utf-8") == second.text.encode("utf-8")
    with pytest.raises(FrozenInstanceError):
        first.text = "changed"  # type: ignore[misc]


def test_generic_selection_preserves_the_exact_user_query() -> None:
    user_query = "  Which Ω value is reported?  "

    implicit = resolve_semantic_query(user_query)
    explicit = resolve_semantic_query(
        user_query, SemanticQuerySelection(mode="user_query")
    )

    assert implicit.policy_id is None
    assert implicit.concern_id is None
    assert implicit.text == user_query
    assert explicit == implicit


def test_user_query_cannot_change_the_concern_representation() -> None:
    selection = SemanticQuerySelection(
        mode="concern", concern_id="stated-study-objective-v1"
    )

    first = resolve_semantic_query("short question", selection)
    second = resolve_semantic_query(
        "A completely different question with punctuation: Ç?", selection
    )

    assert first.text == second.text == CANONICAL_CONCERN_TEXT
    assert first.concern_id == second.concern_id
    assert first.policy_id == second.policy_id


@pytest.mark.parametrize(
    "selection",
    (
        SemanticQuerySelection(mode="concern"),
        SemanticQuerySelection(mode="user_query", concern_id="stated-study-objective-v1"),
        SemanticQuerySelection(mode="unsupported"),  # type: ignore[arg-type]
    ),
)
def test_malformed_selection_fails_closed(selection: SemanticQuerySelection) -> None:
    with pytest.raises(SemanticQueryPolicyError):
        resolve_semantic_query("valid user query", selection)


def test_unknown_concern_fails_closed_without_fallback() -> None:
    with pytest.raises(SemanticQueryPolicyError):
        resolve_semantic_query(
            "valid user query",
            SemanticQuerySelection(mode="concern", concern_id="unknown-concern-v1"),
        )


def test_oversized_concern_id_fails_before_registry_lookup() -> None:
    oversized = "x" * (MAX_CONCERN_ID_CODEPOINTS + 1)

    with pytest.raises(SemanticQueryPolicyError):
        resolve_semantic_query(
            "valid user query",
            SemanticQuerySelection(mode="concern", concern_id=oversized),
        )


def test_concern_id_limits_are_explicit_and_utf8_bounded() -> None:
    assert MAX_CONCERN_ID_CODEPOINTS == 128
    assert MAX_CONCERN_ID_BYTES == 256
    assert len("x" * MAX_CONCERN_ID_CODEPOINTS) == MAX_CONCERN_ID_CODEPOINTS
    assert len(("x" * MAX_CONCERN_ID_CODEPOINTS).encode("utf-8")) <= MAX_CONCERN_ID_BYTES


def test_canonical_representation_has_no_branch_or_fusion_surface() -> None:
    result = resolve_semantic_query(
        "valid user query",
        SemanticQuerySelection(mode="concern", concern_id="stated-study-objective-v1"),
    )

    assert not hasattr(result, "branches")
    assert not hasattr(result, "aliases")
    assert not hasattr(result, "fusion_score")
