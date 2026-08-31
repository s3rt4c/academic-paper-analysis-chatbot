from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from academic_chatbot.db.connection import DatabasePathError, open_read_only_connection
from academic_chatbot.retrieval.fts import RetrievalQueryError, build_literal_match_expression


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("OR", '"OR"'),
        ("AND", '"AND"'),
        ("NOT", '"NOT"'),
        ("NEAR(foo bar)", '"NEAR(foo" "bar)"'),
        ("foo*", '"foo*"'),
        ("(foo)", '"(foo)"'),
        ('"quoted"', '"""quoted"""'),
        ("foo NOT bar", '"foo" "NOT" "bar"'),
        ("' OR 1=1 --", "\"'\" \"OR\" \"1=1\" \"--\""),
        ("  caf\u00e9\tcontrol  ", '"caf\u00e9" "control"'),
    ],
)
def test_plain_lexical_queries_are_quoted_before_they_reach_fts5(
    query: str, expected: str
) -> None:
    """Would fail if user FTS operators or punctuation became active MATCH syntax."""
    assert build_literal_match_expression(query) == expected


@pytest.mark.parametrize("query", ("", " \t\n ", "---", "()"))
def test_empty_or_nonlexical_query_fails_cleanly(query: str) -> None:
    """Would fail if malformed plain input leaked FTS parser errors."""
    with pytest.raises(RetrievalQueryError, match="meaningful lexical"):
        build_literal_match_expression(query)


def test_read_only_connection_requires_an_existing_contained_database(tmp_path: Path) -> None:
    """Would fail if retrieval could initialize missing project state during search."""
    with pytest.raises(DatabasePathError, match="does not exist"):
        open_read_only_connection(tmp_path / "missing.sqlite3", data_root=tmp_path)


def test_read_only_connection_maps_open_failures_to_database_path_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if a read-connection problem leaked a SQLite implementation error."""
    database_path = tmp_path / "project.sqlite3"
    database_path.touch()

    def fail_open(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("read connection unavailable")

    monkeypatch.setattr("academic_chatbot.db.connection.sqlite3.connect", fail_open)
    with pytest.raises(DatabasePathError, match="could not be opened"):
        open_read_only_connection(database_path, data_root=tmp_path)
