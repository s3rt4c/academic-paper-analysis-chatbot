"""Read-only project-local retrieval with persisted evidence validation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from academic_chatbot.retrieval.scoped_models import ScopedRetrievalRequest

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from academic_chatbot.db.connection import DatabasePathError, open_read_only_connection
from academic_chatbot.domain.library import Project
from academic_chatbot.ports.documents import NativePdfAnchor, PdfAnchorBox
from academic_chatbot.retrieval.fts import (
    RetrievalQueryError,
    search_active_chunks,
)
from academic_chatbot.retrieval.query_plan import plan_natural_language_query
from academic_chatbot.storage.paths import ProjectPaths


class RetrievalStorageError(RuntimeError):
    """Raised when a project database cannot be safely searched."""


class RetrievalIntegrityError(RuntimeError):
    """Raised when persisted search evidence cannot be reconstructed exactly."""


class RetrievalHit(BaseModel):
    """One active lexical hit with occurrence-specific persisted evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_id: str
    paper_id: str
    file_version_id: str
    document_generation_id: str
    page_id: str
    physical_page_index: int = Field(ge=0)
    display_page_number: int = Field(ge=1)
    printed_page_label: str | None
    chunk_id: str
    chunk_ordinal: int = Field(ge=0)
    chunk_text: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    rank: int = Field(ge=1)
    raw_bm25_score: float
    anchors: tuple[NativePdfAnchor, ...] = Field(min_length=1)


class RetrievalResults(BaseModel):
    """Immutable ordered lexical results for one requested project."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_id: str
    query: str
    hits: tuple[RetrievalHit, ...]


class RetrievalService:
    """Search persisted active chunks without opening the original PDF."""

    def __init__(self, *, data_root: Path) -> None:
        self._scoped_data_root = Path(data_root)
        self._data_root = data_root.resolve(strict=False)

    def search_scoped(self, request: ScopedRetrievalRequest) -> RetrievalResults:
        """Select eligible chunks before the channel cap in one read snapshot."""
        from academic_chatbot.library.repository import guard_scoped_database
        from academic_chatbot.retrieval.fts import search_active_chunks_scoped
        from academic_chatbot.retrieval.scoped_models import (
            ScopedRetrievalError,
            ScopedRetrievalRequest,
            validate_hit_scope,
        )

        try:
            request = ScopedRetrievalRequest.model_validate(request)
        except ValidationError:
            raise ScopedRetrievalError("INVALID_REQUEST") from None
        if request.mode != "lexical":
            raise ScopedRetrievalError("INVALID_REQUEST")
        paths = guard_scoped_database(self._scoped_data_root, request.scope.project_id)
        try:
            expression = plan_natural_language_query(request.query).expression
            connection = open_read_only_connection(paths.database_path, data_root=self._data_root)
        except RetrievalQueryError:
            raise ScopedRetrievalError("INVALID_REQUEST") from None
        except (DatabasePathError, OSError):
            raise ScopedRetrievalError("STORAGE_UNAVAILABLE") from None
        try:
            connection.execute("BEGIN")
            rows = search_active_chunks_scoped(
                connection, scope=request.scope, match_expression=expression, limit=request.limit
            )
            hits = tuple(
                _hit_from_row(connection, row=row, rank=rank) for rank, row in enumerate(rows, 1)
            )
            for hit in hits:
                validate_hit_scope(hit, request.scope, None)
            return RetrievalResults(
                project_id=request.scope.project_id, query=request.query, hits=hits
            )
        except (sqlite3.DatabaseError, RetrievalIntegrityError, ValueError) as error:
            if isinstance(error, ScopedRetrievalError):
                raise
            raise ScopedRetrievalError("STORAGE_INTEGRITY") from None
        finally:
            connection.rollback()
            connection.close()

    def search(self, project: Project, query: str, limit: int = 10) -> RetrievalResults:
        if type(limit) is not int or limit <= 0:
            raise RetrievalQueryError("limit must be a positive integer")
        expression = plan_natural_language_query(query).expression
        paths = ProjectPaths.create(self._data_root, project_id=project.project_id)
        try:
            connection = open_read_only_connection(paths.database_path, data_root=self._data_root)
        except DatabasePathError as error:
            raise RetrievalStorageError(str(error)) from error
        try:
            rows = search_active_chunks(
                connection,
                project_id=project.project_id,
                match_expression=expression,
                limit=limit,
            )
            hits = tuple(
                _hit_from_row(connection, row=row, rank=index)
                for index, row in enumerate(rows, start=1)
            )
        except sqlite3.DatabaseError as error:
            raise RetrievalStorageError("project database could not be searched") from error
        finally:
            connection.close()
        return RetrievalResults(project_id=project.project_id, query=query, hits=hits)


def _hit_from_row(connection: sqlite3.Connection, *, row: sqlite3.Row, rank: int) -> RetrievalHit:
    page_text = _required_text(row, "canonical_text")
    start_offset = int(row["start_offset"])
    end_offset = int(row["end_offset"])
    chunk_text = _required_text(row, "chunk_text")
    if _required_text(row, "indexed_chunk_text") != chunk_text:
        raise RetrievalIntegrityError("persisted FTS text does not equal its chunk text")
    fts_count = connection.execute(
        "SELECT count(*) FROM chunk_fts WHERE chunk_id = ?", (str(row["chunk_id"]),)
    ).fetchone()
    if fts_count is None or fts_count[0] != 1:
        raise RetrievalIntegrityError("persisted FTS rows must contain exactly one chunk row")
    if (
        not 0 <= start_offset < end_offset <= len(page_text)
        or page_text[start_offset:end_offset] != chunk_text
    ):
        raise RetrievalIntegrityError("persisted chunk does not equal its canonical page range")
    anchor_rows = connection.execute(
        """
        SELECT page_anchor_id, evidence_id, char_start, char_end, anchor_text,
               anchor_text_sha256, boxes_sha256, x0, top, x1, bottom
        FROM page_anchors
        WHERE page_id = ? AND char_start < ? AND char_end > ?
        ORDER BY char_start, char_end, page_anchor_id
        """,
        (str(row["page_id"]), end_offset, start_offset),
    ).fetchall()
    if not anchor_rows:
        raise RetrievalIntegrityError("persisted chunk has no range-scoped anchors")
    try:
        if any(
            int(anchor_row["char_start"]) < start_offset or int(anchor_row["char_end"]) > end_offset
            for anchor_row in anchor_rows
        ):
            raise RetrievalIntegrityError("persisted anchor extends outside its chunk range")
        anchors = tuple(anchor_from_row(row, anchor_row) for anchor_row in anchor_rows)
        return RetrievalHit(
            project_id=str(row["project_id"]),
            paper_id=str(row["paper_id"]),
            file_version_id=str(row["file_version_id"]),
            document_generation_id=str(row["document_generation_id"]),
            page_id=str(row["page_id"]),
            physical_page_index=int(row["physical_page_index"]),
            display_page_number=int(row["display_page_number"]),
            printed_page_label=row["printed_page_label"],
            chunk_id=str(row["chunk_id"]),
            chunk_ordinal=int(row["chunk_ordinal"]),
            start_offset=start_offset,
            end_offset=end_offset,
            chunk_text=chunk_text,
            rank=rank,
            raw_bm25_score=float(row["raw_bm25_score"]),
            anchors=anchors,
        )
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise RetrievalIntegrityError("persisted evidence is malformed") from error


def anchor_from_row(row: sqlite3.Row, anchor_row: sqlite3.Row) -> NativePdfAnchor:
    """Reconstruct one validated native anchor from persisted canonical evidence."""
    box = PdfAnchorBox(
        char_start=int(anchor_row["char_start"]),
        char_end=int(anchor_row["char_end"]),
        x0=float(anchor_row["x0"]),
        top=float(anchor_row["top"]),
        x1=float(anchor_row["x1"]),
        bottom=float(anchor_row["bottom"]),
    )
    file_version_id = str(row["file_version_id"])
    source_pdf_sha256 = str(row["source_pdf_sha256"])
    binding = hashlib.sha256(
        json.dumps(
            {"file_version_id": file_version_id, "pdf_sha256": source_pdf_sha256},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return NativePdfAnchor(
        evidence_id=str(anchor_row["evidence_id"]),
        file_version_id=file_version_id,
        file_version_binding_sha256=binding,
        source_pdf_sha256=source_pdf_sha256,
        parser_profile_sha256=_required_text(row, "parser_profile_sha256"),
        physical_page_index=int(row["physical_page_index"]),
        display_page_number=int(row["display_page_number"]),
        printed_page_label=row["printed_page_label"],
        page_width_points=float(row["page_width_points"]),
        page_height_points=float(row["page_height_points"]),
        source_page_rotation_degrees=int(row["source_page_rotation_degrees"]),
        char_start=int(anchor_row["char_start"]),
        char_end=int(anchor_row["char_end"]),
        canonical_page_text=_required_text(row, "canonical_text"),
        canonical_page_text_sha256=_required_text(row, "canonical_text_sha256"),
        anchor_text=_required_text(anchor_row, "anchor_text"),
        anchor_text_sha256=_required_text(anchor_row, "anchor_text_sha256"),
        boxes=(box,),
        boxes_sha256=_required_text(anchor_row, "boxes_sha256"),
    )


def _required_text(row: sqlite3.Row, name: str) -> str:
    value = row[name]
    if not isinstance(value, str) or not value:
        raise RetrievalIntegrityError(f"persisted {name} is missing")
    return value


def preflight_scoped_evidence(
    connection: sqlite3.Connection,
    *,
    page_id: str,
    chunk_id: str,
    start: int,
    end: int,
) -> int:
    """Inspect selected row metadata before materializing any text or anchor payload."""
    from academic_chatbot.evidence.models import (
        MAX_CHUNK_ANCHORS,
        MAX_IDENTIFIER_CHARS,
        MAX_PAGE_TEXT_BYTES,
        MAX_RESOLVED_TEXT_BYTES,
    )
    from academic_chatbot.retrieval.scoped_models import ScopedRetrievalError

    row = connection.execute(
        """SELECT length(CAST(p.canonical_text AS BLOB)),
                  length(CAST(c.chunk_text AS BLOB)),
                  length(p.printed_page_label),
                  (SELECT count(*) FROM page_anchors a
                   WHERE a.page_id=p.page_id AND a.char_start < ? AND a.char_end > ?),
                  (SELECT coalesce(sum(length(CAST(a.anchor_text AS BLOB))),0)
                   FROM page_anchors a
                   WHERE a.page_id=p.page_id AND a.char_start < ? AND a.char_end > ?),
                  (SELECT coalesce(max(max(length(a.page_anchor_id),length(a.evidence_id))),0)
                   FROM page_anchors a
                   WHERE a.page_id=p.page_id AND a.char_start < ? AND a.char_end > ?)
           FROM pages p JOIN chunks c ON c.page_id=p.page_id
           WHERE p.page_id=? AND c.chunk_id=?""",
        (end, start, end, start, end, start, page_id, chunk_id),
    ).fetchone()
    if row is None:
        raise ScopedRetrievalError("STORAGE_INTEGRITY")
    page_bytes, chunk_bytes, label_length, count, anchor_bytes, id_length = row
    if page_bytes is None or chunk_bytes is None:
        raise ScopedRetrievalError("STORAGE_INTEGRITY")
    # Each returned anchor repeats canonical page text in the existing wire value.
    total = page_bytes * count + chunk_bytes + anchor_bytes
    if (
        page_bytes > MAX_PAGE_TEXT_BYTES
        or chunk_bytes > MAX_PAGE_TEXT_BYTES
        or count > MAX_CHUNK_ANCHORS
        or (label_length or 0) > MAX_IDENTIFIER_CHARS
        or id_length > MAX_IDENTIFIER_CHARS
        or total > MAX_RESOLVED_TEXT_BYTES
    ):
        raise ScopedRetrievalError("RESOURCE_LIMIT")
    return int(total)
