"""Read-only semantic retrieval over one current published vector generation."""

from __future__ import annotations

# ruff: noqa: E501
import hashlib
import json
import sqlite3
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from academic_chatbot.db.connection import DatabasePathError, open_read_only_connection
from academic_chatbot.domain.library import Project
from academic_chatbot.embeddings.artifacts import EmbeddingArtifactError, load_verified_artifacts
from academic_chatbot.embeddings.embedder import OfflineEmbedder, OfflineEmbedderError
from academic_chatbot.embeddings.models import (
    EmbeddingProfile,
    canonical_json_bytes,
    validate_embedding_profile_id,
)
from academic_chatbot.embeddings.tokenizer import EmbeddingInputError, EmbeddingInputTooLongError
from academic_chatbot.embeddings.vector_build import _profile_sha256, _verify_empty_artifact
from academic_chatbot.ports.documents import NativePdfAnchor
from academic_chatbot.retrieval.exact_memmap import ExactVectorStore, VectorHit
from academic_chatbot.retrieval.service import _required_text, anchor_from_row
from academic_chatbot.storage.paths import PathEscapeError, ProjectPaths


class SemanticRetrievalError(RuntimeError):
    """Base class for stable semantic-search domain failures."""


class SemanticQueryError(SemanticRetrievalError):
    """The query is not valid for the frozen QUERY embedding role."""


class SemanticQueryTooLongError(SemanticQueryError):
    """The query would exceed the frozen source-token budget."""


class SemanticIndexUnavailableError(SemanticRetrievalError):
    """No published semantic index exists for this project/profile."""


class SemanticIndexStaleError(SemanticRetrievalError):
    """The published semantic index no longer describes current evidence."""


class SemanticArtifactIntegrityError(SemanticRetrievalError):
    """The authoritative artifact cannot be safely opened as published."""


class SemanticRetrievalIntegrityError(SemanticRetrievalError):
    """Persisted vector-to-evidence lineage does not prove one result."""


class SemanticProfileError(SemanticRetrievalError):
    """The requested local profile/artifact boundary is inconsistent."""


class _QueryEmbedder(Protocol):
    @property
    def profile(self) -> EmbeddingProfile: ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...


class SemanticRetrievalHit(BaseModel):
    """One current, occurrence-specific semantic evidence hit."""

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
    embedding_span_id: str
    embedding_profile_id: str
    vector_generation_id: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    embedding_span_text: str
    rank: int = Field(ge=1)
    raw_semantic_score: float
    anchors: tuple[NativePdfAnchor, ...] = Field(min_length=1)


class SemanticRetrievalResults(BaseModel):
    """Immutable semantic result channel; scores remain raw cosine values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_id: str
    query: str
    embedding_profile_id: str
    vector_generation_id: str
    hits: tuple[SemanticRetrievalHit, ...]


@dataclass(frozen=True, slots=True)
class _Source:
    file_version_id: str
    document_generation_id: str
    eligible_native_chunk_count: int
    needs_ocr_page_count: int


@dataclass(frozen=True, slots=True)
class _Generation:
    vector_generation_id: str
    project_id: str
    embedding_profile_id: str
    source_snapshot_sha256: str
    artifact_relative_dir: str
    vector_store_manifest_sha256: str
    embeddable_spans: int


class SemanticRetrievalService:
    """Search only the requested project's current published semantic evidence."""

    def __init__(
        self, *, data_root: Path, profile: EmbeddingProfile, embedder: _QueryEmbedder
    ) -> None:
        if embedder.profile != profile:
            raise SemanticProfileError("query embedder must use the requested embedding profile")
        self._data_root = Path(data_root).resolve(strict=False)
        self.profile = profile
        self._embedder = embedder

    @classmethod
    def open_from_model_root(
        cls, *, data_root: Path, project_id: str, profile_id: str, model_root: Path
    ) -> SemanticRetrievalService:
        paths = ProjectPaths.create(data_root, project_id=project_id)
        profile, registered_manifest_sha256 = _load_registered_profile_with_manifest_hash(
            paths, profile_id
        )
        try:
            artifacts = load_verified_artifacts(model_root, profile=profile)
            artifact_manifest_sha256 = hashlib.sha256(
                canonical_json_bytes(artifacts.manifest.canonical_payload())
            ).hexdigest()
            if artifact_manifest_sha256 != registered_manifest_sha256:
                raise SemanticProfileError(
                    "verified embedding artifact manifest does not match project registration"
                )
            embedder = OfflineEmbedder.open(model_root, profile)
        except (EmbeddingArtifactError, OfflineEmbedderError, OSError, ValueError) as error:
            raise SemanticProfileError(
                "verified local embedding artifact is unavailable"
            ) from error
        return cls(data_root=data_root, profile=profile, embedder=embedder)

    def search(self, project: Project, query: str, limit: int = 10) -> SemanticRetrievalResults:
        if type(limit) is not int or limit <= 0:
            raise SemanticQueryError("limit must be a positive integer")
        if not isinstance(query, str) or not query.strip():
            raise SemanticQueryError("semantic query must not be empty or whitespace-only")
        paths = ProjectPaths.create(self._data_root, project_id=project.project_id)
        try:
            connection = open_read_only_connection(paths.database_path, data_root=self._data_root)
        except DatabasePathError as error:
            raise SemanticIndexUnavailableError(
                "semantic index is not built for this project"
            ) from error
        try:
            connection.execute("BEGIN")
            generation = _active_generation(
                connection, project_id=project.project_id, profile=self.profile
            )
            current_sources, current_snapshot = _current_snapshot(
                connection,
                project_id=project.project_id,
                embedding_profile_id=self.profile.embedding_profile_id,
            )
            if generation.source_snapshot_sha256 != current_snapshot:
                raise SemanticIndexStaleError("published semantic index is stale")
            if _generation_sources(connection, generation.vector_generation_id) != current_sources:
                raise SemanticIndexStaleError("published semantic index source lineage is stale")
            return self._search_active(
                connection=connection,
                paths=paths,
                generation=generation,
                project=project,
                query=query,
                limit=limit,
            )
        except sqlite3.DatabaseError as error:
            raise SemanticRetrievalIntegrityError(
                "semantic retrieval database state is invalid"
            ) from error
        finally:
            connection.rollback()
            connection.close()

    def _search_active(
        self,
        *,
        connection: sqlite3.Connection,
        paths: ProjectPaths,
        generation: _Generation,
        project: Project,
        query: str,
        limit: int,
    ) -> SemanticRetrievalResults:
        artifact = _artifact_path(
            paths,
            generation.artifact_relative_dir,
            profile_id=generation.embedding_profile_id,
            source_snapshot_sha256=generation.source_snapshot_sha256,
        )
        empty_manifest = artifact / "empty-generation.json"
        if empty_manifest.exists():
            _require_ordinary_artifact_files(artifact, empty=True)
            if generation.embeddable_spans != 0 or _mapping(
                connection, generation.vector_generation_id
            ):
                raise SemanticRetrievalIntegrityError(
                    "empty semantic generation has vector mappings"
                )
            try:
                _verify_empty_artifact(
                    artifact,
                    expected_manifest_sha256=generation.vector_store_manifest_sha256,
                    profile_sha256=_profile_sha256(self.profile),
                    source_snapshot_sha256=generation.source_snapshot_sha256,
                )
            except (OSError, ValueError) as error:
                raise SemanticArtifactIntegrityError(
                    "empty semantic artifact is corrupt"
                ) from error
            return SemanticRetrievalResults(
                project_id=project.project_id,
                query=query,
                embedding_profile_id=self.profile.embedding_profile_id,
                vector_generation_id=generation.vector_generation_id,
                hits=(),
            )

        mapping = _mapping(connection, generation.vector_generation_id)
        _require_ordinary_artifact_files(artifact, empty=False)
        if generation.embeddable_spans != len(mapping) or tuple(row for row, _ in mapping) != tuple(
            range(len(mapping))
        ):
            raise SemanticRetrievalIntegrityError(
                "semantic vector row mapping is incomplete or unordered"
            )
        try:
            store = ExactVectorStore.open(artifact)
        except (OSError, ValueError) as error:
            raise SemanticArtifactIntegrityError(
                "published semantic vector artifact is missing or corrupt"
            ) from error
        try:
            expected_ids = tuple(span_id for _, span_id in mapping)
            if (
                store.manifest.manifest_sha256 != generation.vector_store_manifest_sha256
                or store.manifest.profile_sha256 != _profile_sha256(self.profile)
                or store.manifest.dimension != self.profile.dimension
                or tuple(store.row_ids) != expected_ids
            ):
                raise SemanticArtifactIntegrityError(
                    "published vector artifact does not match its metadata"
                )
            query_vector = self._embed_query(query)
            try:
                vector_hits = store.search(query_vector, limit=limit, block_rows=4096)
            except (TypeError, ValueError) as error:
                raise SemanticRetrievalIntegrityError(
                    "semantic vector search failed validation"
                ) from error
            hits = tuple(
                _semantic_hit(
                    connection=connection,
                    project_id=project.project_id,
                    generation=generation,
                    profile=self.profile,
                    vector_hit=vector_hit,
                    rank=rank,
                )
                for rank, vector_hit in enumerate(vector_hits, start=1)
            )
        finally:
            store.close()
        return SemanticRetrievalResults(
            project_id=project.project_id,
            query=query,
            embedding_profile_id=self.profile.embedding_profile_id,
            vector_generation_id=generation.vector_generation_id,
            hits=hits,
        )

    def _embed_query(self, query: str) -> np.ndarray:
        try:
            vectors = self._embedder.embed_queries((query,))
        except EmbeddingInputTooLongError as error:
            raise SemanticQueryTooLongError(
                "semantic query exceeds the embedding token budget"
            ) from error
        except EmbeddingInputError as error:
            raise SemanticQueryError("semantic query is invalid") from error
        except Exception as error:
            raise SemanticProfileError("local semantic query embedding failed") from error
        if (
            not isinstance(vectors, np.ndarray)
            or vectors.dtype != np.dtype(np.float32)
            or vectors.shape != (1, self.profile.dimension)
            or not np.isfinite(vectors).all()
        ):
            raise SemanticRetrievalIntegrityError("query embedder returned an invalid vector")
        query_vector = np.asarray(vectors[0], dtype=np.float32)
        norm = np.sqrt(np.sum(query_vector * query_vector, dtype=np.float32), dtype=np.float32)
        if not np.isfinite(norm) or not np.isclose(norm, np.float32(1.0), rtol=1e-5, atol=1e-5):
            raise SemanticRetrievalIntegrityError("query embedder returned a non-unit vector")
        return query_vector


def _load_registered_profile_with_manifest_hash(
    paths: ProjectPaths, profile_id: str
) -> tuple[EmbeddingProfile, str]:
    validate_embedding_profile_id(profile_id)
    connection = open_read_only_connection(paths.database_path, data_root=paths.data_root)
    try:
        row = connection.execute(
            """SELECT canonical_profile_json, canonical_profile_sha256, artifact_manifest_sha256
            FROM embedding_profiles WHERE embedding_profile_id = ?""",
            (profile_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise SemanticIndexUnavailableError(
            "semantic embedding profile is not registered for this project"
        )
    try:
        raw_profile = str(row[0]).encode("utf-8")
        profile = EmbeddingProfile.model_validate(json.loads(raw_profile))
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        raise SemanticProfileError("persisted embedding profile is invalid") from error
    if profile.embedding_profile_id != profile_id:
        raise SemanticProfileError("persisted embedding profile identity is invalid")
    if hashlib.sha256(raw_profile).hexdigest() != str(row[1]):
        raise SemanticProfileError("persisted embedding profile content hash is invalid")
    registered_manifest_sha256 = str(row[2])
    if len(registered_manifest_sha256) != 64:
        raise SemanticProfileError("persisted embedding artifact manifest hash is invalid")
    return profile, registered_manifest_sha256


def _active_generation(
    connection: sqlite3.Connection, *, project_id: str, profile: EmbeddingProfile
) -> _Generation:
    row = connection.execute(
        """SELECT vg.* FROM vector_generation_publications AS publication
        JOIN vector_generations AS vg ON vg.vector_generation_id = publication.vector_generation_id
        WHERE publication.project_id = ? AND publication.embedding_profile_id = ?""",
        (project_id, profile.embedding_profile_id),
    ).fetchone()
    if row is None:
        raise SemanticIndexUnavailableError("semantic index is not built for this project/profile")
    if (
        str(row["project_id"]) != project_id
        or str(row["embedding_profile_id"]) != profile.embedding_profile_id
        or str(row["state"]) != "FILES_FINALIZED"
        or row["vector_store_manifest_sha256"] is None
    ):
        raise SemanticRetrievalIntegrityError("published semantic generation is invalid")
    return _Generation(
        vector_generation_id=str(row["vector_generation_id"]),
        project_id=project_id,
        embedding_profile_id=profile.embedding_profile_id,
        source_snapshot_sha256=str(row["source_snapshot_sha256"]),
        artifact_relative_dir=str(row["artifact_relative_dir"]),
        vector_store_manifest_sha256=str(row["vector_store_manifest_sha256"]),
        embeddable_spans=int(row["embeddable_spans"]),
    )


def _current_snapshot(
    connection: sqlite3.Connection, *, project_id: str, embedding_profile_id: str
) -> tuple[tuple[_Source, ...], str]:
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
    sources = tuple(_Source(str(r[0]), str(r[1]), int(r[2]), int(r[3])) for r in rows)
    payload = {
        "schema_version": "vector-source-snapshot-v1",
        "project_id": project_id,
        "embedding_profile_id": embedding_profile_id,
        "sources": [
            {
                "file_version_id": source.file_version_id,
                "document_generation_id": source.document_generation_id,
                "eligible_native_chunk_count": source.eligible_native_chunk_count,
                "needs_ocr_page_count": source.needs_ocr_page_count,
            }
            for source in sources
        ],
    }
    return sources, hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _generation_sources(connection: sqlite3.Connection, generation_id: str) -> tuple[_Source, ...]:
    rows = connection.execute(
        """SELECT file_version_id, document_generation_id, eligible_native_chunk_count, needs_ocr_page_count
        FROM vector_generation_sources WHERE vector_generation_id = ?
        ORDER BY file_version_id, document_generation_id""",
        (generation_id,),
    ).fetchall()
    return tuple(_Source(str(r[0]), str(r[1]), int(r[2]), int(r[3])) for r in rows)


def _mapping(connection: sqlite3.Connection, generation_id: str) -> tuple[tuple[int, str], ...]:
    rows = connection.execute(
        """SELECT vector_row, embedding_span_id FROM vector_generation_spans
        WHERE vector_generation_id = ? ORDER BY vector_row""",
        (generation_id,),
    ).fetchall()
    return tuple((int(row[0]), str(row[1])) for row in rows)


def _artifact_path(
    paths: ProjectPaths,
    relative: str,
    *,
    profile_id: str,
    source_snapshot_sha256: str,
) -> Path:
    parts = relative.split("/")
    expected_prefix = ("indexes", "semantic", profile_id, source_snapshot_sha256)
    if (
        len(parts) != 6
        or tuple(parts[:4]) != expected_prefix
        or parts[4] not in {"generations", "empty-generations"}
        or not parts[5]
    ):
        raise SemanticArtifactIntegrityError(
            "semantic artifact must remain below its project semantic index root"
        )
    try:
        raw_path = paths.project_root.joinpath(*parts)
        current = paths.project_root
        for part in parts:
            current = current / part
            if not current.exists():
                continue
            metadata = current.lstat()
            reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if (
                current.is_symlink()
                or (reparse and metadata.st_file_attributes & reparse)
                or not stat.S_ISDIR(metadata.st_mode)
            ):
                raise SemanticArtifactIntegrityError(
                    "semantic artifact path contains a symlink or reparse entry"
                )
        resolved = paths.resolve_relative(relative)
        if resolved != raw_path.resolve(strict=False):
            raise SemanticArtifactIntegrityError("semantic artifact path resolved unexpectedly")
        return resolved
    except SemanticArtifactIntegrityError:
        raise
    except (OSError, PathEscapeError, ValueError) as error:
        raise SemanticArtifactIntegrityError(
            "semantic artifact path escapes the project"
        ) from error


def _require_ordinary_artifact_files(artifact: Path, *, empty: bool) -> None:
    filenames = (
        ("empty-generation.json",)
        if empty
        else ("manifest.json", "vectors.npy", "vectors.meta.json")
    )
    for filename in filenames:
        candidate = artifact / filename
        try:
            metadata = candidate.lstat()
        except OSError as error:
            raise SemanticArtifactIntegrityError(
                "semantic artifact file is missing or inaccessible"
            ) from error
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if (
            candidate.is_symlink()
            or (reparse and metadata.st_file_attributes & reparse)
            or not stat.S_ISREG(metadata.st_mode)
        ):
            raise SemanticArtifactIntegrityError(
                "semantic artifact file must not be a symlink or reparse entry"
            )


def _semantic_hit(
    *,
    connection: sqlite3.Connection,
    project_id: str,
    generation: _Generation,
    profile: EmbeddingProfile,
    vector_hit: VectorHit,
    rank: int,
) -> SemanticRetrievalHit:
    row = connection.execute(
        """SELECT project.project_id, paper.paper_id, fv.file_version_id, fv.sha256 AS source_pdf_sha256,
        dg.document_generation_id, page.page_id, page.physical_page_index,
        page.page_number AS display_page_number, page.printed_page_label, page.canonical_text,
        page.canonical_text_sha256, page.parser_profile_sha256, page.page_width_points,
        page.page_height_points, page.source_page_rotation_degrees, chunk.chunk_id,
        chunk.ordinal AS chunk_ordinal, chunk.start_offset AS chunk_start_offset,
        chunk.end_offset AS chunk_end_offset, span.embedding_span_id, span.embedding_profile_id,
        span.start_offset, span.end_offset, span.coverage_status
        FROM vector_generation_spans AS mapping
        JOIN embedding_spans AS span ON span.embedding_span_id = mapping.embedding_span_id
        JOIN chunks AS chunk ON chunk.chunk_id = span.chunk_id
        JOIN pages AS page ON page.page_id = span.page_id AND page.document_generation_id = span.document_generation_id
        JOIN document_generations AS dg ON dg.document_generation_id = span.document_generation_id
        JOIN file_versions AS fv ON fv.file_version_id = dg.file_version_id
        JOIN papers AS paper ON paper.paper_id = fv.paper_id
        JOIN projects AS project ON project.project_id = paper.project_id
        JOIN vector_generation_sources AS source
          ON source.vector_generation_id = mapping.vector_generation_id
         AND source.document_generation_id = span.document_generation_id
        WHERE mapping.vector_generation_id = ? AND mapping.vector_row = ?
          AND mapping.embedding_span_id = ?""",
        (generation.vector_generation_id, vector_hit.vector_row, vector_hit.row_id),
    ).fetchall()
    if len(row) != 1:
        raise SemanticRetrievalIntegrityError(
            "semantic vector row mapping is missing or duplicated"
        )
    persisted = row[0]
    if (
        str(persisted["project_id"]) != project_id
        or str(persisted["embedding_span_id"]) != vector_hit.row_id
        or str(persisted["embedding_profile_id"]) != profile.embedding_profile_id
        or str(persisted["coverage_status"]) != "EMBEDDABLE"
    ):
        raise SemanticRetrievalIntegrityError(
            "semantic span lineage does not match the requested project/profile"
        )
    page_text = _required_text(persisted, "canonical_text")
    start, end = int(persisted["start_offset"]), int(persisted["end_offset"])
    chunk_start, chunk_end = (
        int(persisted["chunk_start_offset"]),
        int(persisted["chunk_end_offset"]),
    )
    if not chunk_start <= start < end <= chunk_end <= len(page_text):
        raise SemanticRetrievalIntegrityError(
            "semantic span range is outside its chunk/page evidence"
        )
    span_text = page_text[start:end]
    anchor_rows = connection.execute(
        """SELECT page_anchor_id, evidence_id, char_start, char_end, anchor_text,
        anchor_text_sha256, boxes_sha256, x0, top, x1, bottom FROM page_anchors
        WHERE page_id = ? AND char_start >= ? AND char_end <= ?
        ORDER BY char_start, char_end, page_anchor_id""",
        (str(persisted["page_id"]), start, end),
    ).fetchall()
    if not anchor_rows:
        raise SemanticRetrievalIntegrityError("semantic span has no range-scoped evidence anchors")
    try:
        anchors = tuple(anchor_from_row(persisted, anchor_row) for anchor_row in anchor_rows)
        return SemanticRetrievalHit(
            project_id=project_id,
            paper_id=str(persisted["paper_id"]),
            file_version_id=str(persisted["file_version_id"]),
            document_generation_id=str(persisted["document_generation_id"]),
            page_id=str(persisted["page_id"]),
            physical_page_index=int(persisted["physical_page_index"]),
            display_page_number=int(persisted["display_page_number"]),
            printed_page_label=persisted["printed_page_label"],
            chunk_id=str(persisted["chunk_id"]),
            embedding_span_id=vector_hit.row_id,
            embedding_profile_id=profile.embedding_profile_id,
            vector_generation_id=generation.vector_generation_id,
            start_offset=start,
            end_offset=end,
            embedding_span_text=span_text,
            rank=rank,
            raw_semantic_score=vector_hit.score,
            anchors=anchors,
        )
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise SemanticRetrievalIntegrityError("semantic evidence is malformed") from error
