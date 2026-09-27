"""Bounded, explicit concern-native semantic query representations."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

SEMANTIC_CONCERN_QUERY_POLICY_ID = "semantic-concern-query-v1"
SUPPORTED_CONCERN_ID = "stated-study-objective-v1"
CANONICAL_CONCERN_TEXT = (
    "A study objective states what the study aims to investigate, evaluate, "
    "establish, or develop."
)

MAX_CONCERN_ID_CODEPOINTS = 128
MAX_CONCERN_ID_BYTES = 256
MAX_CANONICAL_CONCERN_TEXT_BYTES = 256
MAX_REGISTRY_ENTRIES = 16


class SemanticQueryPolicyError(ValueError):
    """Raised when an explicit semantic query selection is invalid."""


@dataclass(frozen=True, slots=True)
class SemanticConcernPolicy:
    """One immutable concern-to-canonical-query registry entry."""

    concern_id: str
    policy_id: str
    canonical_text: str


@dataclass(frozen=True, slots=True)
class SemanticQuerySelection:
    """Explicitly select generic user-query or concern-native semantic mode."""

    mode: Literal["user_query", "concern"] = "user_query"
    concern_id: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticQueryRepresentation:
    """The one query-role string selected for one semantic search."""

    policy_id: str | None
    concern_id: str | None
    text: str


CONCERN_REGISTRY = (
    SemanticConcernPolicy(
        concern_id=SUPPORTED_CONCERN_ID,
        policy_id=SEMANTIC_CONCERN_QUERY_POLICY_ID,
        canonical_text=CANONICAL_CONCERN_TEXT,
    ),
)


def resolve_semantic_query(
    user_query: str, selection: SemanticQuerySelection | None = None
) -> SemanticQueryRepresentation:
    """Resolve one explicit selection without rewriting the caller's query."""

    if not isinstance(user_query, str) or not user_query.strip():
        raise SemanticQueryPolicyError("semantic query must not be empty or whitespace-only")
    if selection is None:
        return SemanticQueryRepresentation(policy_id=None, concern_id=None, text=user_query)
    if not isinstance(selection, SemanticQuerySelection):
        raise SemanticQueryPolicyError("semantic query selection is invalid")

    if selection.mode == "user_query":
        if selection.concern_id is not None:
            raise SemanticQueryPolicyError("user-query mode cannot carry a concern")
        return SemanticQueryRepresentation(policy_id=None, concern_id=None, text=user_query)
    if selection.mode != "concern":
        raise SemanticQueryPolicyError("semantic query selection mode is invalid")
    if selection.concern_id is None:
        raise SemanticQueryPolicyError("concern mode requires a concern")

    _validate_concern_id(selection.concern_id)
    entry = next(
        (item for item in CONCERN_REGISTRY if item.concern_id == selection.concern_id),
        None,
    )
    if entry is None:
        raise SemanticQueryPolicyError("semantic concern is unsupported")
    return SemanticQueryRepresentation(
        policy_id=entry.policy_id,
        concern_id=entry.concern_id,
        text=entry.canonical_text,
    )


def _validate_concern_id(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise SemanticQueryPolicyError("semantic concern is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise SemanticQueryPolicyError("semantic concern is invalid") from error
    if (
        len(value) > MAX_CONCERN_ID_CODEPOINTS
        or len(encoded) > MAX_CONCERN_ID_BYTES
    ):
        raise SemanticQueryPolicyError("semantic concern exceeds its resource limit")


def _validate_registry() -> None:
    if len(CONCERN_REGISTRY) > MAX_REGISTRY_ENTRIES:
        raise SemanticQueryPolicyError("semantic concern registry exceeds its resource limit")
    seen: set[str] = set()
    for entry in CONCERN_REGISTRY:
        _validate_concern_id(entry.concern_id)
        if entry.concern_id in seen:
            raise SemanticQueryPolicyError("semantic concern registry contains a duplicate")
        seen.add(entry.concern_id)
        if entry.policy_id != SEMANTIC_CONCERN_QUERY_POLICY_ID:
            raise SemanticQueryPolicyError("semantic concern policy identity is invalid")
        if not isinstance(entry.canonical_text, str) or not entry.canonical_text:
            raise SemanticQueryPolicyError("semantic concern text is invalid")
        try:
            encoded = entry.canonical_text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise SemanticQueryPolicyError("semantic concern text is invalid") from error
        if len(encoded) > MAX_CANONICAL_CONCERN_TEXT_BYTES:
            raise SemanticQueryPolicyError("semantic concern text exceeds its resource limit")
        if unicodedata.normalize("NFC", entry.canonical_text) != entry.canonical_text:
            raise SemanticQueryPolicyError("semantic concern text must be NFC-normalized")


_validate_registry()
