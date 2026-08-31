"""Parameterized SQLite persistence for the minimal local library."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import cast

from academic_chatbot.db.connection import connect_project_database, immediate_transaction
from academic_chatbot.documents.chunking import LexicalChunk, chunk_canonical_page
from academic_chatbot.documents.models import (
    NativePdfDocument,
    NativePdfPage,
    PublishedDocumentGeneration,
)
from academic_chatbot.domain.library import FileVersion, Paper, Project
from academic_chatbot.storage.paths import ProjectPaths


class DuplicateProjectError(ValueError):
    """Raised when an existing project ID is requested again."""


class UnknownProjectError(ValueError):
    """Raised when a paper is assigned to a project that does not exist."""


class UnknownPaperError(ValueError):
    """Raised when admission is requested for an unknown project-owned paper."""


class DocumentGenerationPublicationError(ValueError):
    """Raised when an immutable document generation cannot be published safely."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ProjectRepository:
    """Persistence operations only; filesystem publication belongs to admission."""

    def __init__(self, paths: ProjectPaths) -> None:
        self._paths = paths

    def project_exists(self, project_id: str) -> bool:
        return self._one("SELECT 1 FROM projects WHERE project_id = ?", (project_id,)) is not None

    def create_project(self, project: Project) -> None:
        try:
            with self._connection() as connection, immediate_transaction(connection):
                connection.execute(
                    "INSERT INTO projects (project_id, created_at) VALUES (?, ?)",
                    (project.project_id, _timestamp()),
                )
        except sqlite3.IntegrityError as error:
            raise DuplicateProjectError(f"project already exists: {project.project_id}") from error

    def paper_exists(self, paper_id: str, project_id: str) -> bool:
        return self._one(
            "SELECT 1 FROM papers WHERE paper_id = ? AND project_id = ?", (paper_id, project_id)
        ) is not None

    def create_paper(self, paper: Paper) -> None:
        if not self.project_exists(paper.project_id):
            raise UnknownProjectError(f"unknown project: {paper.project_id}")
        try:
            with self._connection() as connection, immediate_transaction(connection):
                connection.execute(
                    "INSERT INTO papers (paper_id, project_id, created_at) VALUES (?, ?, ?)",
                    (paper.paper_id, paper.project_id, _timestamp()),
                )
        except sqlite3.IntegrityError as error:
            raise UnknownPaperError(f"paper already exists: {paper.paper_id}") from error

    def ensure_paper_belongs_to_project(self, paper_id: str, project_id: str) -> None:
        if not self.paper_exists(paper_id, project_id):
            message = f"paper {paper_id} does not belong to project {project_id}"
            raise UnknownPaperError(message)

    def register_file_version(self, *, file_version: FileVersion) -> FileVersion:
        with self._connection() as connection, immediate_transaction(connection):
            return self.register_file_version_in_transaction(
                connection=connection, file_version=file_version
            )

    def register_file_version_in_transaction(
        self, *, connection: sqlite3.Connection, file_version: FileVersion
    ) -> FileVersion:
        row = connection.execute(
            """
            SELECT file_version_id, paper_id, sha256, original_relative_path
            FROM file_versions WHERE paper_id = ? AND sha256 = ?
            """,
            (file_version.paper_id, file_version.sha256),
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO file_versions
                    (file_version_id, paper_id, sha256, original_relative_path, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    file_version.file_version_id,
                    file_version.paper_id,
                    file_version.sha256,
                    file_version.stored_relative_path,
                    _timestamp(),
                ),
            )
            return file_version
        return FileVersion(
            file_version_id=str(row[0]),
            paper_id=str(row[1]),
            sha256=str(row[2]),
            byte_length=file_version.byte_length,
            stored_relative_path=str(row[3]),
        )

    def any_file_version_references_sha256(
        self, sha256: str, *, connection: sqlite3.Connection | None = None
    ) -> bool:
        if connection is not None:
            return connection.execute(
                "SELECT 1 FROM file_versions WHERE sha256 = ?", (sha256,)
            ).fetchone() is not None
        return self._one("SELECT 1 FROM file_versions WHERE sha256 = ?", (sha256,)) is not None

    @contextmanager
    def file_version_admission_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection, immediate_transaction(connection):
            yield connection

    def file_version_count(self) -> int:
        row = self._one("SELECT count(*) FROM file_versions", ())
        assert row is not None
        return int(row[0])

    def publish_native_document(
        self,
        *,
        parsed: NativePdfDocument,
        document_generation_id: str,
        processing_profile_id: str,
        lexical_chunk_profile_id: str,
    ) -> PublishedDocumentGeneration:
        """Atomically publish one complete generation or retain the existing active one."""

        with self._connection() as connection, immediate_transaction(connection):
            self._validate_parsed_file_version(connection, parsed.file_version)
            existing = connection.execute(
                """
                SELECT document_generation_id FROM document_generations
                WHERE file_version_id = ? AND pipeline_version = ?
                """,
                (parsed.file_version.file_version_id, processing_profile_id),
            ).fetchone()
            if existing is not None:
                existing_id = str(existing[0])
                if existing_id != document_generation_id:
                    raise DocumentGenerationPublicationError(
                        "existing generation identity disagrees with the processing profile"
                    )
                result = self._published_generation_result(
                    connection,
                    file_version_id=parsed.file_version.file_version_id,
                    document_generation_id=existing_id,
                    processing_profile_id=processing_profile_id,
                    reused=True,
                )
                self._validate_candidate_generation(connection, result)
                self._validate_active_generation(connection, result)
                return result

            connection.execute(
                """
                INSERT INTO document_generations
                    (document_generation_id, file_version_id, pipeline_version, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    document_generation_id,
                    parsed.file_version.file_version_id,
                    processing_profile_id,
                    _timestamp(),
                ),
            )
            chunk_count = 0
            for page in parsed.pages:
                page_id = _page_id_for(
                    document_generation_id=document_generation_id,
                    physical_page_index=page.physical_page_index,
                )
                connection.execute(
                    """
                    INSERT INTO pages (
                        page_id, document_generation_id, page_number, text_relative_path,
                        physical_page_index, printed_page_label, canonical_text,
                        canonical_text_sha256, parser_profile_sha256, page_width_points,
                        page_height_points, source_page_rotation_degrees, extraction_quality,
                        needs_ocr
                    ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        page_id,
                        document_generation_id,
                        page.display_page_number,
                        page.physical_page_index,
                        page.printed_page_label,
                        page.canonical_text,
                        page.canonical_text_sha256,
                        page.parser_profile_sha256,
                        page.page_width_points,
                        page.page_height_points,
                        page.source_page_rotation_degrees,
                        page.quality,
                        int(page.needs_ocr),
                    ),
                )
                self._persist_page_anchors(connection, page_id=page_id, parsed_page=page)
                if not page.needs_ocr:
                    chunks = chunk_canonical_page(
                        document_generation_id=document_generation_id,
                        page_id=page_id,
                        canonical_text=page.canonical_text,
                        canonical_words=page.words,
                        processing_profile_id=lexical_chunk_profile_id,
                    )
                    self._persist_chunks(
                        connection,
                        chunks=chunks,
                        lexical_chunk_profile_id=lexical_chunk_profile_id,
                    )
                    self._persist_chunk_fts(connection, chunks=chunks)
                    chunk_count += len(chunks)

            candidate = PublishedDocumentGeneration(
                file_version_id=parsed.file_version.file_version_id,
                document_generation_id=document_generation_id,
                processing_profile_id=processing_profile_id,
                page_count=len(parsed.pages),
                chunk_count=chunk_count,
                reused=False,
            )
            self._validate_candidate_generation(connection, candidate)
            self._activate_generation(connection, candidate)
            self._validate_active_generation(connection, candidate)
            return candidate

    @staticmethod
    def _validate_parsed_file_version(
        connection: sqlite3.Connection, file_version: FileVersion
    ) -> None:
        persisted = connection.execute(
            """
            SELECT paper_id, sha256, original_relative_path
            FROM file_versions WHERE file_version_id = ?
            """,
            (file_version.file_version_id,),
        ).fetchone()
        if persisted is None or tuple(persisted) != (
            file_version.paper_id,
            file_version.sha256,
            file_version.stored_relative_path,
        ):
            raise DocumentGenerationPublicationError(
                "parsed FileVersion does not match project record"
            )

    @staticmethod
    def _persist_page_anchors(
        connection: sqlite3.Connection, *, page_id: str, parsed_page: NativePdfPage
    ) -> None:
        for anchor in parsed_page.anchors:
            if len(anchor.boxes) != 1:
                raise DocumentGenerationPublicationError(
                    "Task 4 requires one persisted source box per Task 3 word anchor"
                )
            box = anchor.boxes[0]
            connection.execute(
                """
                INSERT INTO page_anchors (
                    page_anchor_id, evidence_id, page_id, char_start, char_end, anchor_text,
                    anchor_text_sha256, boxes_sha256, x0, top, x1, bottom
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _page_anchor_id_for(page_id=page_id, evidence_id=anchor.evidence_id),
                    anchor.evidence_id,
                    page_id,
                    anchor.char_start,
                    anchor.char_end,
                    anchor.anchor_text,
                    anchor.anchor_text_sha256,
                    anchor.boxes_sha256,
                    box.x0,
                    box.top,
                    box.x1,
                    box.bottom,
                ),
            )

    @staticmethod
    def _persist_chunks(
        connection: sqlite3.Connection,
        *,
        chunks: tuple[LexicalChunk, ...],
        lexical_chunk_profile_id: str,
    ) -> None:
        for chunk in chunks:
            if chunk.processing_profile_id != lexical_chunk_profile_id:
                raise DocumentGenerationPublicationError("chunk profile changed during publication")
            connection.execute(
                """
                INSERT INTO chunks (
                    chunk_id, document_generation_id, page_id, ordinal, start_offset,
                    end_offset, chunk_text, lexical_word_count, processing_profile_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.document_generation_id,
                    chunk.page_id,
                    chunk.ordinal,
                    chunk.start_offset,
                    chunk.end_offset,
                    chunk.text,
                    chunk.lexical_word_count,
                    chunk.processing_profile_id,
                ),
            )

    @staticmethod
    def _persist_chunk_fts(
        connection: sqlite3.Connection, *, chunks: tuple[LexicalChunk, ...]
    ) -> None:
        for chunk in chunks:
            connection.execute(
                "INSERT INTO chunk_fts (chunk_id, chunk_text) VALUES (?, ?)",
                (chunk.chunk_id, chunk.text),
            )

    @staticmethod
    def _validate_candidate_generation(
        connection: sqlite3.Connection, candidate: PublishedDocumentGeneration
    ) -> None:
        page_count = connection.execute(
            "SELECT count(*) FROM pages WHERE document_generation_id = ?",
            (candidate.document_generation_id,),
        ).fetchone()
        chunk_count = connection.execute(
            "SELECT count(*) FROM chunks WHERE document_generation_id = ?",
            (candidate.document_generation_id,),
        ).fetchone()
        invalid_chunks = connection.execute(
            """
            SELECT 1 FROM chunks AS c JOIN pages AS p ON p.page_id = c.page_id
            WHERE c.document_generation_id = ?
              AND c.chunk_text <> substr(p.canonical_text, c.start_offset + 1,
                                          c.end_offset - c.start_offset)
            LIMIT 1
            """,
            (candidate.document_generation_id,),
        ).fetchone()
        missing_anchor = connection.execute(
            """
            SELECT 1 FROM chunks AS c
            WHERE c.document_generation_id = ? AND NOT EXISTS (
                SELECT 1 FROM page_anchors AS a
                WHERE a.page_id = c.page_id
                  AND a.char_start < c.end_offset AND a.char_end > c.start_offset
            ) LIMIT 1
            """,
            (candidate.document_generation_id,),
        ).fetchone()
        fts_count = connection.execute(
            """
            SELECT count(*) FROM chunk_fts
            WHERE chunk_id IN (
                SELECT chunk_id FROM chunks WHERE document_generation_id = ?
            )
            """,
            (candidate.document_generation_id,),
        ).fetchone()
        invalid_fts_entry = connection.execute(
            """
            SELECT 1 FROM chunks AS c
            LEFT JOIN chunk_fts AS f ON f.chunk_id = c.chunk_id
            WHERE c.document_generation_id = ?
            GROUP BY c.chunk_id, c.chunk_text
            HAVING count(f.rowid) <> 1 OR max(f.chunk_text) IS NOT c.chunk_text
            LIMIT 1
            """,
            (candidate.document_generation_id,),
        ).fetchone()
        if (
            page_count is None
            or chunk_count is None
            or fts_count is None
            or int(page_count[0]) != candidate.page_count
            or int(chunk_count[0]) != candidate.chunk_count
            or int(fts_count[0]) != candidate.chunk_count
            or invalid_chunks is not None
            or missing_anchor is not None
            or invalid_fts_entry is not None
        ):
            raise DocumentGenerationPublicationError(
                "candidate generation failed evidence validation"
            )
        connection.execute("INSERT INTO chunk_fts(chunk_fts) VALUES ('integrity-check')")

    @staticmethod
    def _activate_generation(
        connection: sqlite3.Connection, candidate: PublishedDocumentGeneration
    ) -> None:
        connection.execute(
            """
            DELETE FROM chunk_fts WHERE chunk_id IN (
                SELECT c.chunk_id FROM chunks AS c
                JOIN generation_publications AS active
                  ON active.document_generation_id = c.document_generation_id
                WHERE active.file_version_id = ?
            )
            """,
            (candidate.file_version_id,),
        )
        connection.execute(
            """
            INSERT INTO generation_publications (file_version_id, document_generation_id)
            VALUES (?, ?)
            ON CONFLICT(file_version_id) DO UPDATE SET
                document_generation_id = excluded.document_generation_id
            """,
            (candidate.file_version_id, candidate.document_generation_id),
        )

    @staticmethod
    def _validate_active_generation(
        connection: sqlite3.Connection, candidate: PublishedDocumentGeneration
    ) -> None:
        active = connection.execute(
            """
            SELECT document_generation_id FROM generation_publications
            WHERE file_version_id = ?
            """,
            (candidate.file_version_id,),
        ).fetchone()
        invalid_fts = connection.execute(
            """
            SELECT 1 FROM chunk_fts AS f
            LEFT JOIN chunks AS c ON c.chunk_id = f.chunk_id
            LEFT JOIN generation_publications AS active
              ON active.document_generation_id = c.document_generation_id
            WHERE c.chunk_id IS NULL OR active.document_generation_id IS NULL
            LIMIT 1
            """
        ).fetchone()
        if active is None or str(active[0]) != candidate.document_generation_id or invalid_fts:
            raise DocumentGenerationPublicationError(
                "active generation failed publication validation"
            )

    @staticmethod
    def _published_generation_result(
        connection: sqlite3.Connection,
        *,
        file_version_id: str,
        document_generation_id: str,
        processing_profile_id: str,
        reused: bool,
    ) -> PublishedDocumentGeneration:
        page_count = connection.execute(
            "SELECT count(*) FROM pages WHERE document_generation_id = ?",
            (document_generation_id,),
        ).fetchone()
        chunk_count = connection.execute(
            "SELECT count(*) FROM chunks WHERE document_generation_id = ?",
            (document_generation_id,),
        ).fetchone()
        assert page_count is not None and chunk_count is not None
        return PublishedDocumentGeneration(
            file_version_id=file_version_id,
            document_generation_id=document_generation_id,
            processing_profile_id=processing_profile_id,
            page_count=int(page_count[0]),
            chunk_count=int(chunk_count[0]),
            reused=reused,
        )

    def _one(self, statement: str, parameters: tuple[str, ...]) -> sqlite3.Row | None:
        with self._connection() as connection:
            row = connection.execute(statement, parameters).fetchone()
            return cast(sqlite3.Row | None, row)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = connect_project_database(
            self._paths.database_path, data_root=self._paths.data_root
        )
        try:
            yield connection
        finally:
            connection.close()


def _page_id_for(*, document_generation_id: str, physical_page_index: int) -> str:
    identity = (
        len(document_generation_id.encode("utf-8")).to_bytes(8, "big")
        + document_generation_id.encode("utf-8")
        + physical_page_index.to_bytes(8, "big", signed=False)
    )
    return "page-sha256-" + hashlib.sha256(identity).hexdigest()


def _page_anchor_id_for(*, page_id: str, evidence_id: str) -> str:
    identity = (
        len(page_id.encode("utf-8")).to_bytes(8, "big")
        + page_id.encode("utf-8")
        + len(evidence_id.encode("ascii")).to_bytes(8, "big")
        + evidence_id.encode("ascii")
    )
    return "page-anchor-sha256-" + hashlib.sha256(identity).hexdigest()
