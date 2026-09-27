"""Deterministic natural-language planning for ordinary lexical search."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from academic_chatbot.retrieval.fts import RetrievalQueryError

LEXICAL_POLICY_ID = "lexical-natural-language-v1"
_MAX_QUERY_BYTES = 4096
_MAX_RAW_SPANS = 128
_MAX_TOKENS = 256
_MAX_TERM_CODE_POINTS = 64
_MAX_TERM_BYTES = 128
_MAX_SELECTED_TERMS = 32
_MAX_EXPRESSION_BYTES = 8192
_ENGLISH_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "this",
        "that",
        "these",
        "those",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "and",
        "or",
    }
)


@dataclass(frozen=True, slots=True)
class LexicalQueryPlan:
    """Immutable, short-lived state for one safe lexical query."""

    policy_id: str
    normalized_terms: tuple[str, ...] = field(repr=False)
    selected_terms: tuple[str, ...] = field(repr=False)
    match_policy: str
    expression: str = field(repr=False)


def plan_natural_language_query(query: str) -> LexicalQueryPlan:
    """Plan one ordinary user query as a bounded, literal FTS OR expression."""

    if type(query) is not str:
        raise RetrievalQueryError("lexical query must be text")
    if len(query) > _MAX_QUERY_BYTES:
        raise RetrievalQueryError("lexical query exceeds resource limits")
    try:
        raw_bytes = query.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise RetrievalQueryError("lexical query contains unsupported characters") from error
    if len(raw_bytes) > _MAX_QUERY_BYTES:
        raise RetrievalQueryError("lexical query exceeds resource limits")
    if len(query.split()) > _MAX_RAW_SPANS:
        raise RetrievalQueryError("lexical query exceeds resource limits")

    for character in query:
        category = unicodedata.category(character)
        if category in {"Cc", "Cf"} and character not in "\t\n\r":
            raise RetrievalQueryError("lexical query contains unsupported characters")

    normalized = _ascii_lower(unicodedata.normalize("NFC", query))
    try:
        normalized_bytes = normalized.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise RetrievalQueryError("lexical query contains unsupported characters") from error
    if len(normalized_bytes) > _MAX_QUERY_BYTES:
        raise RetrievalQueryError("lexical query exceeds resource limits")

    normalized_terms = _scan_terms(normalized)
    selected_terms = tuple(
        term
        for term in normalized_terms
        if term not in _ENGLISH_FUNCTION_WORDS
    )
    selected_terms = tuple(dict.fromkeys(selected_terms))
    if not selected_terms:
        raise RetrievalQueryError("lexical query has no searchable terms")
    if len(selected_terms) > _MAX_SELECTED_TERMS:
        raise RetrievalQueryError("lexical query exceeds resource limits")

    literals = tuple(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in selected_terms)
    expression = " OR ".join(literals)
    if len(expression.encode("utf-8")) > _MAX_EXPRESSION_BYTES:
        raise RetrievalQueryError("lexical query exceeds resource limits")
    return LexicalQueryPlan(
        policy_id=LEXICAL_POLICY_ID,
        normalized_terms=normalized_terms,
        selected_terms=selected_terms,
        match_policy="any_term",
        expression=expression,
    )


def _ascii_lower(value: str) -> str:
    return "".join(
        chr(ord(character) + (ord("a") - ord("A")))
        if "A" <= character <= "Z"
        else character
        for character in value
    )


def _scan_terms(value: str) -> tuple[str, ...]:
    terms: list[str] = []
    current: list[str] = []

    def finish() -> None:
        if not current:
            return
        if len(terms) >= _MAX_TOKENS:
            raise RetrievalQueryError("lexical query exceeds resource limits")
        term = "".join(current)
        if len(term) > _MAX_TERM_CODE_POINTS or len(term.encode("utf-8")) > _MAX_TERM_BYTES:
            raise RetrievalQueryError("lexical query exceeds resource limits")
        terms.append(term)
        current.clear()

    for character in value:
        category = unicodedata.category(character)
        if category[0] in {"L", "N"} or (category[0] == "M" and current):
            current.append(character)
        else:
            finish()
    finish()
    return tuple(terms)
