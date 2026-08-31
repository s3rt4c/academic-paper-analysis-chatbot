"""Literal FTS5 query construction and active-generation lexical selection."""

from __future__ import annotations

import sqlite3


class RetrievalQueryError(ValueError):
    """Raised when a plain lexical query cannot be searched safely."""


_ACTIVE_SEARCH_SQL = """
    SELECT
        projects.project_id, papers.paper_id, file_versions.file_version_id,
        file_versions.sha256 AS source_pdf_sha256,
        document_generations.document_generation_id,
        pages.page_id, pages.physical_page_index,
        pages.page_number AS display_page_number, pages.printed_page_label,
        pages.canonical_text, pages.canonical_text_sha256,
        pages.parser_profile_sha256, pages.page_width_points,
        pages.page_height_points, pages.source_page_rotation_degrees,
        chunks.chunk_id, chunks.ordinal AS chunk_ordinal,
        chunks.start_offset, chunks.end_offset, chunks.chunk_text,
        chunk_fts.chunk_text AS indexed_chunk_text,
        bm25(chunk_fts) AS raw_bm25_score
    FROM chunk_fts
    JOIN chunks ON chunks.chunk_id = chunk_fts.chunk_id
    JOIN generation_publications
      ON generation_publications.document_generation_id = chunks.document_generation_id
    JOIN document_generations
      ON document_generations.document_generation_id = chunks.document_generation_id
     AND document_generations.file_version_id = generation_publications.file_version_id
    JOIN file_versions
      ON file_versions.file_version_id = document_generations.file_version_id
    JOIN papers ON papers.paper_id = file_versions.paper_id
    JOIN projects ON projects.project_id = papers.project_id
    JOIN pages
      ON pages.page_id = chunks.page_id
     AND pages.document_generation_id = chunks.document_generation_id
    WHERE chunk_fts MATCH ? AND projects.project_id = ?
    ORDER BY bm25(chunk_fts) ASC, papers.paper_id ASC, file_versions.file_version_id ASC,
             pages.physical_page_index ASC, chunks.ordinal ASC, chunks.chunk_id ASC
    LIMIT ?
"""

def build_literal_match_expression(query: str) -> str:
    """Quote every user term so FTS5 operators cannot become syntax."""

    terms = query.split()
    if not terms or not any(character.isalnum() for term in terms for character in term):
        raise RetrievalQueryError("query must contain meaningful lexical content")
    return " ".join(f'"{term.replace("\"", "\"\"")}"' for term in terms)


def search_active_chunks(
    connection: sqlite3.Connection, *, project_id: str, match_expression: str, limit: int
) -> tuple[sqlite3.Row, ...]:
    """Return bounded active project chunks in deterministic BM25 order."""

    connection.execute(
        _ACTIVE_SEARCH_SQL, ('"task5-schema-probe"', project_id, 0)
    ).fetchall()
    try:
        rows = connection.execute(
            _ACTIVE_SEARCH_SQL, (match_expression, project_id, limit)
        ).fetchall()
    except sqlite3.OperationalError as error:
        raise RetrievalQueryError("plain lexical query could not be searched") from error
    return tuple(rows)
