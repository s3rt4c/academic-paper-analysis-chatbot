from __future__ import annotations

import sqlite3

import pytest

from academic_chatbot.retrieval.fts import RetrievalQueryError
from academic_chatbot.retrieval.query_plan import plan_natural_language_query


def test_question_and_duplicates_have_a_deterministic_content_plan() -> None:
    plan = plan_natural_language_query("Why does THERMAL thermal conductivity change?")

    assert plan.policy_id == "lexical-natural-language-v1"
    assert plan.normalized_terms == (
        "why",
        "does",
        "thermal",
        "thermal",
        "conductivity",
        "change",
    )
    assert plan.selected_terms == ("thermal", "conductivity", "change")
    assert plan.match_policy == "any_term"
    assert plan.expression == '"thermal" OR "conductivity" OR "change"'
    assert plan == plan_natural_language_query(
        "Why does THERMAL thermal conductivity change?"
    )


def test_operators_are_data_and_cannot_match_an_unrelated_row() -> None:
    plan = plan_natural_language_query('alpha NOT NEAR(beta)* "gamma"')

    assert plan.expression == '"alpha" OR "not" OR "near" OR "beta" OR "gamma"'
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE VIRTUAL TABLE evidence USING fts5("
            "text, tokenize='unicode61 remove_diacritics 2')"
        )
        connection.executemany(
            "INSERT INTO evidence(text) VALUES (?)", [("alpha",), ("unrelated",)]
        )
        rows = connection.execute(
            "SELECT rowid FROM evidence WHERE evidence MATCH ?", (plan.expression,)
        ).fetchall()

    assert rows == [(1,)]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            'alpha AND OR NOT NEAR* (beta):gamma-"delta"',
            '"alpha" OR "not" OR "near" OR "beta" OR "gamma" OR "delta"',
        ),
        ('embedded "quote" punctuation!!!', '"embedded" OR "quote" OR "punctuation"'),
    ],
)
def test_special_input_is_structurally_literal(query: str, expected: str) -> None:
    assert plan_natural_language_query(query).expression == expected


@pytest.mark.parametrize("query", ["", " \t\n", "what is the", "OR", "* ()"])
def test_no_information_is_an_error(query: str) -> None:
    with pytest.raises(RetrievalQueryError, match="no searchable terms"):
        plan_natural_language_query(query)


def test_normalization_preserves_nfc_and_non_ascii_terms() -> None:
    plan = plan_natural_language_query("CAFÉ cafe\u0301 \u0130stanbul çal\u0131şma")

    assert plan.normalized_terms == ("cafÉ", "café", "\u0130stanbul", "çal\u0131şma")
    assert plan.selected_terms == plan.normalized_terms


def test_pruning_is_exact_and_does_not_remove_domain_terms() -> None:
    query = (
        "a an the what which who whom whose when where why how is are was were "
        "be been being do does did this that these those of to in on at for and or "
        "objective aim purpose study evaluate investigate"
    )

    plan = plan_natural_language_query(query)

    assert plan.selected_terms == (
        "objective",
        "aim",
        "purpose",
        "study",
        "evaluate",
        "investigate",
    )


def test_first_occurrence_deduplication_is_preserved() -> None:
    plan = plan_natural_language_query("Beta alpha BETA alpha")

    assert plan.normalized_terms == ("beta", "alpha", "beta", "alpha")
    assert plan.selected_terms == ("beta", "alpha")
    assert plan.expression == '"beta" OR "alpha"'


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("thermal", ('"thermal"',)),
        ("thermal conductivity", ('"thermal" OR "conductivity"',)),
        ("alpha-beta, gamma_delta", ('"alpha" OR "beta" OR "gamma" OR "delta"',)),
    ],
)
def test_term_cardinality_and_punctuation(query: str, expected: tuple[str, ...]) -> None:
    assert plan_natural_language_query(query).expression == expected[0]


def test_selected_term_limit_rejects_without_truncation() -> None:
    accepted = plan_natural_language_query(" ".join(f"term{index}" for index in range(32)))
    assert len(accepted.selected_terms) == 32

    query = " ".join(f"term{index}" for index in range(33))

    with pytest.raises(RetrievalQueryError, match="resource limits"):
        plan_natural_language_query(query)


def test_exact_term_code_point_and_utf8_limits() -> None:
    assert len(plan_natural_language_query("x" * 64).selected_terms) == 1
    with pytest.raises(RetrievalQueryError, match="resource limits"):
        plan_natural_language_query("x" * 65)
    assert len(plan_natural_language_query("é" * 64).selected_terms) == 1
    with pytest.raises(RetrievalQueryError, match="resource limits"):
        plan_natural_language_query("界" * 43)


def test_query_and_whitespace_limits_are_enforced() -> None:
    assert plan_natural_language_query(("the " * 127) + "thermal").selected_terms == ("thermal",)
    with pytest.raises(RetrievalQueryError, match="resource limits"):
        plan_natural_language_query(("the " * 128) + "thermal")
    with pytest.raises(RetrievalQueryError, match="resource limits"):
        plan_natural_language_query(" ".join("the-" for _ in range(257)))

    accepted = plan_natural_language_query(("the-" * 255) + "thermal")
    assert accepted.selected_terms == ("thermal",)


def test_raw_query_byte_limit_is_exact() -> None:
    accepted = plan_natural_language_query("thermal" + (" " * (4096 - len("thermal"))))
    assert accepted.selected_terms == ("thermal",)
    with pytest.raises(RetrievalQueryError, match="resource limits"):
        plan_natural_language_query("thermal" + (" " * (4097 - len("thermal"))))


@pytest.mark.parametrize("query", [None, b"thermal", 42])
def test_query_must_be_text(query: object) -> None:
    with pytest.raises(RetrievalQueryError, match="must be text"):
        plan_natural_language_query(query)  # type: ignore[arg-type]


@pytest.mark.parametrize("query", ["control\x00thermal", "zero\u200bwidth", "bad\u202eformat"])
def test_unsupported_controls_fail_before_search(query: str) -> None:
    with pytest.raises(RetrievalQueryError, match="unsupported characters"):
        plan_natural_language_query(query)


def test_lone_surrogate_is_an_unsupported_character() -> None:
    with pytest.raises(RetrievalQueryError, match="unsupported characters"):
        plan_natural_language_query("bad\ud800")


def test_repr_does_not_expose_query_terms_or_expression() -> None:
    plan = plan_natural_language_query("sentinel thermal")

    assert "sentinel" not in repr(plan)
    assert "thermal" not in repr(plan)


def test_repeated_planning_is_deterministic() -> None:
    query = "Why does thermal conductivity change?"

    assert plan_natural_language_query(query) == plan_natural_language_query(query)
