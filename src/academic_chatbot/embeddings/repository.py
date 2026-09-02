"""Narrow SQLite metadata persistence for future offline semantic retrieval."""

from __future__ import annotations

# ruff: noqa: E501
import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from academic_chatbot.db.connection import connect_project_database, immediate_transaction
from academic_chatbot.embeddings.models import (
    EmbeddingProfile,
    EmbeddingSpanIdentity,
    canonical_json_bytes,
    validate_embedding_profile_id,
)
from academic_chatbot.storage.paths import PathEscapeError, ProjectPaths


class EmbeddingPersistenceError(ValueError):
    """Raised when immutable embedding/vector metadata is inconsistent."""


class EmbeddingSpanStatus(StrEnum):
    EMBEDDABLE = "EMBEDDABLE"
    EXCLUDED_UNEMBEDDABLE = "EXCLUDED_UNEMBEDDABLE"


class VectorGenerationState(StrEnum):
    DB_CANDIDATE = "DB_CANDIDATE"
    FILES_FINALIZED = "FILES_FINALIZED"
    STALE = "STALE"


@dataclass(frozen=True, slots=True)
class EmbeddingSpan:
    identity: EmbeddingSpanIdentity
    status: EmbeddingSpanStatus

    @property
    def embedding_span_id(self) -> str:
        return self.identity.embedding_span_id


@dataclass(frozen=True, slots=True)
class SourceMembership:
    file_version_id: str
    document_generation_id: str
    eligible_native_chunk_count: int
    needs_ocr_page_count: int


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    project_id: str
    embedding_profile_id: str
    source_snapshot_sha256: str
    sources: tuple[SourceMembership, ...]


@dataclass(frozen=True, slots=True)
class CanonicalSourceWord:
    """One durable canonical-word anchor used for exact semantic span construction."""

    text: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class ActiveChunkSource:
    """One active lexical chunk with its page-local canonical word ranges."""

    file_version_id: str
    document_generation_id: str
    page_id: str
    physical_page_index: int
    chunk_id: str
    chunk_ordinal: int
    page_text: str
    chunk_text: str
    chunk_start_offset: int
    chunk_end_offset: int
    words: tuple[CanonicalSourceWord, ...]


@dataclass(frozen=True, slots=True)
class VectorGenerationCoverage:
    eligible_native_chunks: int
    embeddable_spans: int
    excluded_unembeddable_spans: int
    needs_ocr_pages: int
    indexed_documents: int
    unindexed_documents: int

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.eligible_native_chunks,
                self.embeddable_spans,
                self.excluded_unembeddable_spans,
                self.needs_ocr_pages,
                self.indexed_documents,
                self.unindexed_documents,
            )
        ):
            raise EmbeddingPersistenceError("coverage values must be non-negative")


@dataclass(frozen=True, slots=True)
class VectorGeneration:
    vector_generation_id: str
    project_id: str
    embedding_profile_id: str
    source_snapshot_sha256: str
    artifact_relative_dir: str
    state: VectorGenerationState
    vector_store_manifest_sha256: str | None
    coverage: VectorGenerationCoverage


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _generation_id(
    *, project_id: str, embedding_profile_id: str, source_snapshot_sha256: str
) -> str:
    payload = {
        "schema_version": "vector-generation-v1",
        "project_id": project_id,
        "embedding_profile_id": embedding_profile_id,
        "source_snapshot_sha256": source_snapshot_sha256,
    }
    return "vector-generation-sha256-" + _sha256(canonical_json_bytes(payload))


class EmbeddingRepository:
    """Project-local persistence; it neither loads models nor opens vector files."""

    def __init__(self, paths: ProjectPaths) -> None:
        self._paths = paths

    def register_profile(self, profile: EmbeddingProfile, *, artifact_manifest_sha256: str) -> None:
        _require_sha256(artifact_manifest_sha256, "artifact manifest hash")
        profile_id = profile.embedding_profile_id
        canonical = canonical_json_bytes(
            profile.model_dump(mode="json", exclude={"embedding_profile_id"})
        )
        with self._connection() as connection, immediate_transaction(connection):
            existing = connection.execute(
                "SELECT canonical_profile_json, artifact_manifest_sha256 FROM embedding_profiles WHERE embedding_profile_id = ?",
                (profile_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing[0]).encode("utf-8") != canonical
                    or str(existing[1]) != artifact_manifest_sha256
                ):
                    raise EmbeddingPersistenceError(
                        "embedding profile ID conflicts with immutable content"
                    )
                return
            connection.execute(
                """INSERT INTO embedding_profiles (
                    embedding_profile_id, canonical_profile_json, canonical_profile_sha256,
                    artifact_manifest_sha256, dimension, span_policy_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile_id,
                    canonical.decode("utf-8"),
                    _sha256(canonical),
                    artifact_manifest_sha256,
                    profile.dimension,
                    profile.span_policy,
                    _timestamp(),
                ),
            )

    def get_profile(self, embedding_profile_id: str) -> EmbeddingProfile | None:
        validate_embedding_profile_id(embedding_profile_id)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT canonical_profile_json FROM embedding_profiles WHERE embedding_profile_id = ?",
                (embedding_profile_id,),
            ).fetchone()
        if row is None:
            return None
        profile = EmbeddingProfile.model_validate(json.loads(str(row[0])))
        if profile.embedding_profile_id != embedding_profile_id:
            raise EmbeddingPersistenceError(
                "persisted embedding profile does not match its identity"
            )
        return profile

    def persist_span(self, span: EmbeddingSpan) -> None:
        identity = span.identity
        if self.get_profile(identity.embedding_profile_id) is None:
            raise EmbeddingPersistenceError("embedding span references an unknown profile")
        with self._connection() as connection, immediate_transaction(connection):
            existing = connection.execute(
                "SELECT embedding_profile_id, document_generation_id, chunk_id, page_id, start_offset, end_offset, coverage_status FROM embedding_spans WHERE embedding_span_id = ?",
                (span.embedding_span_id,),
            ).fetchone()
            expected = (
                identity.embedding_profile_id,
                identity.document_generation_id,
                identity.chunk_id,
                identity.page_id,
                identity.start_offset,
                identity.end_offset,
                span.status.value,
            )
            if existing is not None:
                if tuple(existing) != expected:
                    raise EmbeddingPersistenceError(
                        "embedding span ID conflicts with immutable occurrence"
                    )
                return
            connection.execute(
                """INSERT INTO embedding_spans VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (span.embedding_span_id, *expected),
            )

    def current_source_snapshot(
        self, *, project_id: str, embedding_profile_id: str
    ) -> SourceSnapshot:
        self._require_repository_project(project_id)
        validate_embedding_profile_id(embedding_profile_id)
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT fv.file_version_id, active.document_generation_id,
                    count(DISTINCT chunks.chunk_id) AS eligible_native_chunk_count,
                    count(DISTINCT CASE WHEN pages.needs_ocr = 1 THEN pages.page_id END) AS needs_ocr_page_count
                FROM file_versions AS fv
                JOIN papers AS paper ON paper.paper_id = fv.paper_id
                JOIN generation_publications AS active ON active.file_version_id = fv.file_version_id
                LEFT JOIN pages ON pages.document_generation_id = active.document_generation_id
                LEFT JOIN chunks ON chunks.document_generation_id = active.document_generation_id
                WHERE paper.project_id = ?
                GROUP BY fv.file_version_id, active.document_generation_id
                ORDER BY fv.file_version_id ASC, active.document_generation_id ASC""",
                (project_id,),
            ).fetchall()
        sources = tuple(
            SourceMembership(str(row[0]), str(row[1]), int(row[2]), int(row[3])) for row in rows
        )
        payload = {
            "schema_version": "vector-source-snapshot-v1",
            "project_id": project_id,
            "embedding_profile_id": embedding_profile_id,
            "sources": [
                {
                    "file_version_id": item.file_version_id,
                    "document_generation_id": item.document_generation_id,
                    "eligible_native_chunk_count": item.eligible_native_chunk_count,
                    "needs_ocr_page_count": item.needs_ocr_page_count,
                }
                for item in sources
            ],
        }
        return SourceSnapshot(
            project_id, embedding_profile_id, _sha256(canonical_json_bytes(payload)), sources
        )

    def active_chunk_sources(
        self, *, project_id: str, embedding_profile_id: str
    ) -> tuple[ActiveChunkSource, ...]:
        """Read active chunks with construction-derived persisted word anchors.

        The query deliberately reads page anchors rather than token offsets or
        searching canonical text.  A malformed historic generation is rejected
        before it can produce a semantic span.
        """

        self._require_repository_project(project_id)
        validate_embedding_profile_id(embedding_profile_id)
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT fv.file_version_id, active.document_generation_id, p.page_id,
                    p.physical_page_index, c.chunk_id, c.ordinal, p.canonical_text,
                    c.chunk_text, c.start_offset AS chunk_start_offset,
                    c.end_offset AS chunk_end_offset, c.lexical_word_count,
                    anchor.char_start AS word_start_offset,
                    anchor.char_end AS word_end_offset, anchor.anchor_text AS word_text
                FROM file_versions AS fv
                JOIN papers AS paper ON paper.paper_id = fv.paper_id
                JOIN generation_publications AS active
                  ON active.file_version_id = fv.file_version_id
                JOIN pages AS p ON p.document_generation_id = active.document_generation_id
                JOIN chunks AS c
                  ON c.document_generation_id = active.document_generation_id
                 AND c.page_id = p.page_id
                LEFT JOIN page_anchors AS anchor
                  ON anchor.page_id = c.page_id
                 AND anchor.char_start >= c.start_offset
                 AND anchor.char_end <= c.end_offset
                WHERE paper.project_id = ?
                ORDER BY fv.file_version_id, active.document_generation_id,
                    p.physical_page_index, c.ordinal, anchor.char_start""",
                (project_id,),
            ).fetchall()

        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(str(row["chunk_id"]), []).append(row)
        sources: list[ActiveChunkSource] = []
        for chunk_rows in grouped.values():
            first = chunk_rows[0]
            words: list[CanonicalSourceWord] = []
            for row in chunk_rows:
                if row["word_start_offset"] is None:
                    continue
                words.append(
                    CanonicalSourceWord(
                        text=str(row["word_text"]),
                        start_offset=int(row["word_start_offset"]),
                        end_offset=int(row["word_end_offset"]),
                    )
                )
            source = ActiveChunkSource(
                file_version_id=str(first["file_version_id"]),
                document_generation_id=str(first["document_generation_id"]),
                page_id=str(first["page_id"]),
                physical_page_index=int(first["physical_page_index"]),
                chunk_id=str(first["chunk_id"]),
                chunk_ordinal=int(first["ordinal"]),
                page_text=str(first["canonical_text"]),
                chunk_text=str(first["chunk_text"]),
                chunk_start_offset=int(first["chunk_start_offset"]),
                chunk_end_offset=int(first["chunk_end_offset"]),
                words=tuple(words),
            )
            _validate_active_chunk_source(
                source, expected_word_count=int(first["lexical_word_count"])
            )
            sources.append(source)
        return tuple(sources)

    def generation_for_snapshot(
        self,
        *,
        project_id: str,
        embedding_profile_id: str,
        source_snapshot_sha256: str,
    ) -> VectorGeneration | None:
        """Return the deterministic candidate identity, if it already exists."""

        self._require_repository_project(project_id)
        validate_embedding_profile_id(embedding_profile_id)
        _require_sha256(source_snapshot_sha256, "source snapshot hash")
        generation_id = _generation_id(
            project_id=project_id,
            embedding_profile_id=embedding_profile_id,
            source_snapshot_sha256=source_snapshot_sha256,
        )
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM vector_generations WHERE vector_generation_id = ?",
                (generation_id,),
            ).fetchone()
        return None if row is None else _generation_from_row(row)

    def candidate_generations(
        self, *, project_id: str, embedding_profile_id: str
    ) -> tuple[VectorGeneration, ...]:
        """Return incomplete candidates for bounded, controlled reconciliation."""

        self._require_repository_project(project_id)
        validate_embedding_profile_id(embedding_profile_id)
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM vector_generations
                WHERE project_id = ? AND embedding_profile_id = ? AND state = ?
                ORDER BY vector_generation_id""",
                (project_id, embedding_profile_id, VectorGenerationState.DB_CANDIDATE),
            ).fetchall()
        return tuple(_generation_from_row(row) for row in rows)

    def finalized_unpublished_generations(
        self, *, project_id: str, embedding_profile_id: str
    ) -> tuple[VectorGeneration, ...]:
        """Return finalized candidates that have no authoritative pointer yet."""

        self._require_repository_project(project_id)
        validate_embedding_profile_id(embedding_profile_id)
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT generation.* FROM vector_generations AS generation
                LEFT JOIN vector_generation_publications AS publication
                  ON publication.vector_generation_id = generation.vector_generation_id
                WHERE generation.project_id = ? AND generation.embedding_profile_id = ?
                  AND generation.state = ? AND publication.vector_generation_id IS NULL
                ORDER BY generation.vector_generation_id""",
                (project_id, embedding_profile_id, VectorGenerationState.FILES_FINALIZED),
            ).fetchall()
        return tuple(_generation_from_row(row) for row in rows)

    def mark_stale(self, vector_generation_id: str) -> None:
        """Permanently retire an unpublished candidate whose source changed."""

        with self._connection() as connection, immediate_transaction(connection):
            generation = self._generation_in_transaction(connection, vector_generation_id)
            if generation.state is not VectorGenerationState.FILES_FINALIZED:
                raise EmbeddingPersistenceError("only finalized unpublished generations can become stale")
            published = connection.execute(
                "SELECT 1 FROM vector_generation_publications WHERE vector_generation_id = ?",
                (vector_generation_id,),
            ).fetchone()
            if published is not None:
                raise EmbeddingPersistenceError("published vector generations cannot become stale")
            current = self._snapshot_in_transaction(
                connection, generation.project_id, generation.embedding_profile_id
            )
            if current.source_snapshot_sha256 == generation.source_snapshot_sha256:
                raise EmbeddingPersistenceError("current vector generation cannot become stale")
            connection.execute(
                "UPDATE vector_generations SET state = 'STALE' WHERE vector_generation_id = ?",
                (vector_generation_id,),
            )

    def discard_candidate(self, vector_generation_id: str) -> None:
        """Remove only incomplete DB metadata; finalized files stay forensic orphans."""

        with self._connection() as connection, immediate_transaction(connection):
            generation = self._generation_in_transaction(connection, vector_generation_id)
            if generation.state is not VectorGenerationState.DB_CANDIDATE:
                raise EmbeddingPersistenceError(
                    "only an incomplete database candidate can be discarded"
                )
            connection.execute(
                "DELETE FROM vector_generation_spans WHERE vector_generation_id = ?",
                (vector_generation_id,),
            )
            connection.execute(
                "DELETE FROM vector_generation_sources WHERE vector_generation_id = ?",
                (vector_generation_id,),
            )
            connection.execute(
                "DELETE FROM vector_generations WHERE vector_generation_id = ?",
                (vector_generation_id,),
            )

    def create_candidate(
        self,
        *,
        project_id: str,
        embedding_profile_id: str,
        artifact_relative_dir: str,
        coverage: VectorGenerationCoverage,
    ) -> VectorGeneration:
        self._require_repository_project(project_id)
        validate_embedding_profile_id(embedding_profile_id)
        self._validate_artifact_dir(artifact_relative_dir)
        if self.get_profile(embedding_profile_id) is None:
            raise EmbeddingPersistenceError("vector generation references an unknown profile")
        snapshot = self.current_source_snapshot(
            project_id=project_id, embedding_profile_id=embedding_profile_id
        )
        generation_id = _generation_id(
            project_id=project_id,
            embedding_profile_id=embedding_profile_id,
            source_snapshot_sha256=snapshot.source_snapshot_sha256,
        )
        with self._connection() as connection, immediate_transaction(connection):
            existing = connection.execute(
                "SELECT * FROM vector_generations WHERE vector_generation_id = ?", (generation_id,)
            ).fetchone()
            if existing is not None:
                result = _generation_from_row(existing)
                if (
                    result.artifact_relative_dir != artifact_relative_dir
                    or result.coverage != coverage
                ):
                    raise EmbeddingPersistenceError(
                        "vector generation identity conflicts with immutable metadata"
                    )
                return result
            connection.execute(
                """INSERT INTO vector_generations VALUES (?, ?, ?, ?, ?, 'DB_CANDIDATE', NULL,
                   ?, ?, ?, ?, ?, ?, ?)""",
                (
                    generation_id,
                    project_id,
                    embedding_profile_id,
                    snapshot.source_snapshot_sha256,
                    artifact_relative_dir,
                    coverage.eligible_native_chunks,
                    coverage.embeddable_spans,
                    coverage.excluded_unembeddable_spans,
                    coverage.needs_ocr_pages,
                    coverage.indexed_documents,
                    coverage.unindexed_documents,
                    _timestamp(),
                ),
            )
            for source in snapshot.sources:
                connection.execute(
                    "INSERT INTO vector_generation_sources VALUES (?, ?, ?, ?, ?)",
                    (
                        generation_id,
                        source.file_version_id,
                        source.document_generation_id,
                        source.eligible_native_chunk_count,
                        source.needs_ocr_page_count,
                    ),
                )
        return VectorGeneration(
            generation_id,
            project_id,
            embedding_profile_id,
            snapshot.source_snapshot_sha256,
            artifact_relative_dir,
            VectorGenerationState.DB_CANDIDATE,
            None,
            coverage,
        )

    def attach_vector_row(
        self, *, vector_generation_id: str, vector_row: int, embedding_span_id: str
    ) -> None:
        if vector_row < 0:
            raise EmbeddingPersistenceError("vector row must be non-negative")
        with self._connection() as connection, immediate_transaction(connection):
            generation = connection.execute(
                "SELECT state FROM vector_generations WHERE vector_generation_id = ?",
                (vector_generation_id,),
            ).fetchone()
            if generation is None or str(generation[0]) != VectorGenerationState.DB_CANDIDATE:
                raise EmbeddingPersistenceError(
                    "only a candidate vector generation accepts row mappings"
                )
            span = connection.execute(
                "SELECT coverage_status FROM embedding_spans WHERE embedding_span_id = ?",
                (embedding_span_id,),
            ).fetchone()
            if span is not None and str(span[0]) == EmbeddingSpanStatus.EXCLUDED_UNEMBEDDABLE:
                raise EmbeddingPersistenceError(
                    "excluded unembeddable spans cannot obtain vector rows"
                )
            try:
                connection.execute(
                    "INSERT INTO vector_generation_spans VALUES (?, ?, ?)",
                    (vector_generation_id, vector_row, embedding_span_id),
                )
            except sqlite3.IntegrityError as error:
                raise EmbeddingPersistenceError(
                    "invalid or duplicate vector row mapping"
                ) from error

    def finalize_candidate(
        self, vector_generation_id: str, *, vector_store_manifest_sha256: str
    ) -> VectorGeneration:
        _require_sha256(vector_store_manifest_sha256, "vector store manifest hash")
        with self._connection() as connection, immediate_transaction(connection):
            generation = self._generation_in_transaction(connection, vector_generation_id)
            if generation.state is not VectorGenerationState.DB_CANDIDATE:
                raise EmbeddingPersistenceError("only a database candidate can be finalized")
            self._require_current(connection, generation)
            self._validate_candidate(connection, generation)
            connection.execute(
                "UPDATE vector_generations SET state = 'FILES_FINALIZED', vector_store_manifest_sha256 = ? WHERE vector_generation_id = ?",
                (vector_store_manifest_sha256, vector_generation_id),
            )
            return replace(
                generation,
                state=VectorGenerationState.FILES_FINALIZED,
                vector_store_manifest_sha256=vector_store_manifest_sha256,
            )

    def publish(self, vector_generation_id: str) -> None:
        with self._connection() as connection, immediate_transaction(connection):
            generation = self._generation_in_transaction(connection, vector_generation_id)
            if generation.state is not VectorGenerationState.FILES_FINALIZED:
                raise EmbeddingPersistenceError("only a finalized vector generation can publish")
            self._require_current(connection, generation)
            self._validate_candidate(connection, generation)
            connection.execute(
                """INSERT INTO vector_generation_publications VALUES (?, ?, ?)
                ON CONFLICT(project_id, embedding_profile_id) DO UPDATE SET vector_generation_id = excluded.vector_generation_id""",
                (
                    generation.project_id,
                    generation.embedding_profile_id,
                    generation.vector_generation_id,
                ),
            )

    def active_generation(
        self, *, project_id: str, embedding_profile_id: str
    ) -> VectorGeneration | None:
        self._require_repository_project(project_id)
        with self._connection() as connection:
            connection.execute("BEGIN")
            try:
                row = connection.execute(
                    """SELECT vg.* FROM vector_generation_publications AS publication
                    JOIN vector_generations AS vg ON vg.vector_generation_id = publication.vector_generation_id
                    WHERE publication.project_id = ? AND publication.embedding_profile_id = ?""",
                    (project_id, embedding_profile_id),
                ).fetchone()
                if row is None:
                    return None
                generation = _generation_from_row(row)
                snapshot = self._snapshot_in_transaction(
                    connection, project_id, embedding_profile_id
                )
                if snapshot.source_snapshot_sha256 != generation.source_snapshot_sha256:
                    return None
                return generation
            finally:
                connection.rollback()

    def vector_row_mapping(self, vector_generation_id: str) -> tuple[tuple[int, str], ...]:
        """Return the persisted ExactVectorStore row identity in explicit row order."""

        with self._connection() as connection:
            rows = connection.execute(
                """SELECT vector_row, embedding_span_id FROM vector_generation_spans
                WHERE vector_generation_id = ? ORDER BY vector_row ASC""",
                (vector_generation_id,),
            ).fetchall()
        return tuple((int(row[0]), str(row[1])) for row in rows)

    def is_generation_current(self, vector_generation_id: str) -> bool:
        with self._connection() as connection:
            generation = self._generation_in_transaction(connection, vector_generation_id)
        snapshot = self.current_source_snapshot(
            project_id=generation.project_id, embedding_profile_id=generation.embedding_profile_id
        )
        return snapshot.source_snapshot_sha256 == generation.source_snapshot_sha256

    def _require_repository_project(self, project_id: str) -> None:
        if project_id != self._paths.project_id:
            raise EmbeddingPersistenceError("embedding repository cannot access another project")

    def _require_current(
        self, connection: sqlite3.Connection, generation: VectorGeneration
    ) -> None:
        snapshot = self._snapshot_in_transaction(
            connection, generation.project_id, generation.embedding_profile_id
        )
        stored = tuple(
            SourceMembership(str(row[0]), str(row[1]), int(row[2]), int(row[3]))
            for row in connection.execute(
                """SELECT file_version_id, document_generation_id, eligible_native_chunk_count,
                needs_ocr_page_count FROM vector_generation_sources
                WHERE vector_generation_id = ? ORDER BY file_version_id, document_generation_id""",
                (generation.vector_generation_id,),
            )
        )
        if (
            snapshot.source_snapshot_sha256 != generation.source_snapshot_sha256
            or stored != snapshot.sources
        ):
            raise EmbeddingPersistenceError(
                "stale vector generation cannot be finalized or published"
            )

    def _snapshot_in_transaction(
        self, connection: sqlite3.Connection, project_id: str, embedding_profile_id: str
    ) -> SourceSnapshot:
        # Read through the same connection so finalization/publication is atomic with its current check.
        rows = connection.execute(
            """SELECT fv.file_version_id, active.document_generation_id,
            count(DISTINCT chunks.chunk_id), count(DISTINCT CASE WHEN pages.needs_ocr = 1 THEN pages.page_id END)
            FROM file_versions AS fv JOIN papers AS paper ON paper.paper_id = fv.paper_id
            JOIN generation_publications AS active ON active.file_version_id = fv.file_version_id
            LEFT JOIN pages ON pages.document_generation_id = active.document_generation_id
            LEFT JOIN chunks ON chunks.document_generation_id = active.document_generation_id
            WHERE paper.project_id = ? GROUP BY fv.file_version_id, active.document_generation_id
            ORDER BY fv.file_version_id, active.document_generation_id""",
            (project_id,),
        ).fetchall()
        sources = tuple(SourceMembership(str(r[0]), str(r[1]), int(r[2]), int(r[3])) for r in rows)
        payload = {
            "schema_version": "vector-source-snapshot-v1",
            "project_id": project_id,
            "embedding_profile_id": embedding_profile_id,
            "sources": [
                {
                    "file_version_id": s.file_version_id,
                    "document_generation_id": s.document_generation_id,
                    "eligible_native_chunk_count": s.eligible_native_chunk_count,
                    "needs_ocr_page_count": s.needs_ocr_page_count,
                }
                for s in sources
            ],
        }
        return SourceSnapshot(
            project_id, embedding_profile_id, _sha256(canonical_json_bytes(payload)), sources
        )

    @staticmethod
    def _validate_artifact_dir(relative_dir: str) -> None:
        try:
            # Existing ProjectPaths is the authoritative Windows-aware path policy.
            ProjectPaths.create(Path("."), project_id="validation").resolve_relative(relative_dir)
        except (PathEscapeError, ValueError) as error:
            raise EmbeddingPersistenceError(
                "artifact directory must be a safe project-relative path"
            ) from error

    def _validate_candidate(
        self, connection: sqlite3.Connection, generation: VectorGeneration
    ) -> None:
        source = connection.execute(
            "SELECT coalesce(sum(eligible_native_chunk_count), 0), coalesce(sum(needs_ocr_page_count), 0), count(*) FROM vector_generation_sources WHERE vector_generation_id = ?",
            (generation.vector_generation_id,),
        ).fetchone()
        mapped = connection.execute(
            """SELECT count(*), count(DISTINCT span.document_generation_id),
            min(mapping.vector_row), max(mapping.vector_row)
            FROM vector_generation_spans AS mapping
            JOIN embedding_spans AS span ON span.embedding_span_id = mapping.embedding_span_id
            WHERE mapping.vector_generation_id = ?""",
            (generation.vector_generation_id,),
        ).fetchone()
        excluded = connection.execute(
            """SELECT count(*) FROM embedding_spans AS span JOIN vector_generation_sources AS source
            ON source.document_generation_id = span.document_generation_id
            WHERE source.vector_generation_id = ? AND span.embedding_profile_id = ?
              AND span.coverage_status = 'EXCLUDED_UNEMBEDDABLE'""",
            (generation.vector_generation_id, generation.embedding_profile_id),
        ).fetchone()
        uncovered = connection.execute(
            """SELECT 1 FROM chunks AS chunk JOIN vector_generation_sources AS source
            ON source.document_generation_id = chunk.document_generation_id
            WHERE source.vector_generation_id = ? AND NOT EXISTS (
                SELECT 1 FROM embedding_spans AS span WHERE span.embedding_profile_id = ? AND span.chunk_id = chunk.chunk_id
            ) LIMIT 1""",
            (generation.vector_generation_id, generation.embedding_profile_id),
        ).fetchone()
        unmapped_embeddable = connection.execute(
            """SELECT 1 FROM embedding_spans AS span
            JOIN vector_generation_sources AS source
              ON source.document_generation_id = span.document_generation_id
            LEFT JOIN vector_generation_spans AS mapping
              ON mapping.vector_generation_id = source.vector_generation_id
             AND mapping.embedding_span_id = span.embedding_span_id
            WHERE source.vector_generation_id = ?
              AND span.embedding_profile_id = ?
              AND span.coverage_status = 'EMBEDDABLE'
              AND mapping.embedding_span_id IS NULL
            LIMIT 1""",
            (generation.vector_generation_id, generation.embedding_profile_id),
        ).fetchone()
        assert source is not None and mapped is not None and excluded is not None
        expected = generation.coverage
        actual = (
            int(source[0]),
            int(mapped[0]),
            int(excluded[0]),
            int(source[1]),
            int(mapped[1]),
            int(source[2]) - int(mapped[1]),
        )
        if (
            actual
            != (
                expected.eligible_native_chunks,
                expected.embeddable_spans,
                expected.excluded_unembeddable_spans,
                expected.needs_ocr_pages,
                expected.indexed_documents,
                expected.unindexed_documents,
            )
            or uncovered is not None
            or unmapped_embeddable is not None
            or (
                int(mapped[0]) > 0 and (int(mapped[2]) != 0 or int(mapped[3]) != int(mapped[0]) - 1)
            )
        ):
            raise EmbeddingPersistenceError(
                "vector generation coverage or source-span mapping is inconsistent"
            )

    @staticmethod
    def _generation_in_transaction(
        connection: sqlite3.Connection, vector_generation_id: str
    ) -> VectorGeneration:
        row = connection.execute(
            "SELECT * FROM vector_generations WHERE vector_generation_id = ?",
            (vector_generation_id,),
        ).fetchone()
        if row is None:
            raise EmbeddingPersistenceError("unknown vector generation")
        return _generation_from_row(row)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = connect_project_database(
            self._paths.database_path, data_root=self._paths.data_root
        )
        try:
            yield connection
        finally:
            connection.close()


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise EmbeddingPersistenceError(f"{label} must be a lowercase SHA-256 digest")


def _generation_from_row(row: sqlite3.Row) -> VectorGeneration:
    return VectorGeneration(
        str(row[0]),
        str(row[1]),
        str(row[2]),
        str(row[3]),
        str(row[4]),
        VectorGenerationState(str(row[5])),
        None if row[6] is None else str(row[6]),
        VectorGenerationCoverage(*(int(row[index]) for index in range(7, 13))),
    )


def _validate_active_chunk_source(source: ActiveChunkSource, *, expected_word_count: int) -> None:
    if (
        not source.words
        or source.chunk_start_offset < 0
        or source.chunk_end_offset <= source.chunk_start_offset
        or source.chunk_end_offset > len(source.page_text)
        or len(source.words) != expected_word_count
    ):
        raise EmbeddingPersistenceError("active chunk lacks complete canonical word evidence")
    cursor = source.chunk_start_offset
    for index, word in enumerate(source.words):
        if (
            not word.text
            or word.start_offset != cursor
            or word.end_offset != word.start_offset + len(word.text)
            or word.end_offset > source.chunk_end_offset
            or source.page_text[word.start_offset : word.end_offset] != word.text
        ):
            raise EmbeddingPersistenceError("active chunk canonical word evidence is invalid")
        cursor = word.end_offset + (1 if index < len(source.words) - 1 else 0)
    if (
        source.words[-1].end_offset != source.chunk_end_offset
        or source.page_text[source.chunk_start_offset : source.chunk_end_offset]
        != source.chunk_text
        or source.chunk_text != " ".join(word.text for word in source.words)
    ):
        raise EmbeddingPersistenceError("active chunk source slice is not canonical")
