"""Read-only orchestration for frozen hybrid retrieval evidence."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol

from academic_chatbot.db.connection import DatabasePathError, open_read_only_connection
from academic_chatbot.domain.library import Project
from academic_chatbot.retrieval.hybrid_fusion import (
    FusedHybridCandidate,
    candidate_limit,
    fuse_candidates,
)
from academic_chatbot.retrieval.hybrid_models import (
    HybridCandidateKey,
    HybridChannelState,
    HybridParentChunkContext,
    HybridRetrievalHit,
    HybridRetrievalResults,
)
from academic_chatbot.retrieval.semantic import (
    SemanticRetrievalHit,
    SemanticRetrievalResults,
    SemanticRetrievalService,
    _current_snapshot,
    _generation_sources,
)
from academic_chatbot.retrieval.service import (
    RetrievalHit,
    RetrievalResults,
    RetrievalService,
    _required_text,
    anchor_from_row,
)
from academic_chatbot.storage.paths import ProjectPaths


class HybridRetrievalIntegrityError(RuntimeError):
    """Raised when fused candidates cannot be resolved as current exact evidence."""


class _LexicalSearch(Protocol):
    def search(self, project: Project, query: str, limit: int) -> RetrievalResults: ...


class _SemanticSearch(Protocol):
    def search(self, project: Project, query: str, limit: int) -> SemanticRetrievalResults: ...


class _ParentResolver(Protocol):
    def resolve(
        self, project: Project, candidate: FusedHybridCandidate
    ) -> HybridParentChunkContext: ...


class HybridRetrievalService:
    """Compose accepted channel searches without changing their failure semantics."""

    def __init__(
        self,
        *,
        data_root: Path | str,
        semantic_service: _SemanticSearch,
        lexical_service: _LexicalSearch | None = None,
        parent_resolver: _ParentResolver | None = None,
    ) -> None:
        self._data_root = Path(data_root).resolve(strict=False)
        self._lexical = lexical_service or RetrievalService(data_root=self._data_root)
        self._semantic = semantic_service
        self._resolver = parent_resolver or _ParentChunkEvidenceResolver(data_root=self._data_root)

    @classmethod
    def open_from_model_root(
        cls, *, data_root: Path, project_id: str, profile_id: str, model_root: Path
    ) -> HybridRetrievalService:
        """Open the accepted semantic lifecycle with explicit project/profile/model inputs."""

        return cls(
            data_root=data_root,
            semantic_service=SemanticRetrievalService.open_from_model_root(
                data_root=data_root,
                project_id=project_id,
                profile_id=profile_id,
                model_root=model_root,
            ),
        )

    def search(self, project: Project, query: str, limit: int = 10) -> HybridRetrievalResults:
        """Search both accepted channels and resolve exact current parent evidence."""

        depth = candidate_limit(limit)
        lexical = self._lexical.search(project, query, limit=depth)
        semantic = self._semantic.search(project, query, limit=depth)
        _validate_channel_results(project, query, lexical, semantic)
        fused = fuse_candidates(lexical.hits, semantic.hits, final_limit=limit)
        hits = tuple(
            HybridRetrievalHit(
                identity=candidate.identity,
                parent_chunk=self._resolver.resolve(project, candidate),
                lexical_contribution=candidate.lexical_contribution,
                semantic_contribution=candidate.semantic_contribution,
                trace=candidate.trace,
            )
            for candidate in fused
        )
        return HybridRetrievalResults(
            project_id=project.project_id,
            query=query,
            fusion_profile_id="rrf-v1",
            lexical_state=_state_for(lexical.hits),
            semantic_state=_state_for(semantic.hits),
            hits=hits,
        )


class _ParentChunkEvidenceResolver:
    """Resolve one fused parent identity through a fresh read-only exact-lineage query."""

    def __init__(self, *, data_root: Path) -> None:
        self._data_root = data_root.resolve(strict=False)

    def resolve(
        self, project: Project, candidate: FusedHybridCandidate
    ) -> HybridParentChunkContext:
        _validate_candidate_packets(candidate)
        paths = ProjectPaths.create(self._data_root, project_id=project.project_id)
        try:
            connection = open_read_only_connection(paths.database_path, data_root=self._data_root)
        except DatabasePathError as error:
            raise HybridRetrievalIntegrityError(
                "parent evidence database is unavailable"
            ) from error
        try:
            connection.execute("BEGIN")
            if candidate.semantic_contribution is not None:
                _require_current_semantic_generation(
                    connection, project, candidate.semantic_contribution.semantic_hit
                )
            row = connection.execute(
                _PARENT_CHUNK_SQL,
                (
                    project.project_id,
                    candidate.identity.document_generation_id,
                    candidate.identity.page_id,
                    candidate.identity.chunk_id,
                ),
            ).fetchall()
            if len(row) != 1:
                raise HybridRetrievalIntegrityError(
                    "parent chunk is not exactly current for the fused identity"
                )
            parent = _parent_context_from_row(connection, row=row[0], identity=candidate.identity)
            if parent.identity != candidate.identity:
                raise HybridRetrievalIntegrityError(
                    "resolved parent chunk does not match the fused identity"
                )
            return parent
        except HybridRetrievalIntegrityError:
            raise
        except sqlite3.DatabaseError as error:
            raise HybridRetrievalIntegrityError(
                "parent evidence database state is invalid"
            ) from error
        except (KeyError, TypeError, ValueError) as error:
            raise HybridRetrievalIntegrityError("parent evidence is malformed") from error
        finally:
            connection.rollback()
            connection.close()


def _validate_channel_results(
    project: Project, query: str, lexical: RetrievalResults, semantic: SemanticRetrievalResults
) -> None:
    if lexical.project_id != project.project_id or semantic.project_id != project.project_id:
        raise HybridRetrievalIntegrityError("channel result project does not match the request")
    if lexical.query != query or semantic.query != query:
        raise HybridRetrievalIntegrityError(
            "channel result query does not match the original request"
        )


def _validate_candidate_packets(candidate: FusedHybridCandidate) -> None:
    lexical = candidate.lexical_contribution
    semantic = candidate.semantic_contribution
    if lexical is not None and _identity_for(lexical.lexical_hit) != candidate.identity:
        raise HybridRetrievalIntegrityError(
            "lexical contribution does not match the fused identity"
        )
    if semantic is not None and _identity_for(semantic.semantic_hit) != candidate.identity:
        raise HybridRetrievalIntegrityError(
            "semantic contribution does not match the fused identity"
        )
    if lexical is None and semantic is None:
        raise HybridRetrievalIntegrityError("fused candidate has no channel contribution")


def _require_current_semantic_generation(
    connection: sqlite3.Connection, project: Project, hit: SemanticRetrievalHit
) -> None:
    row = connection.execute(
        """SELECT generation.source_snapshot_sha256
        FROM vector_generation_publications AS publication
        JOIN vector_generations AS generation
          ON generation.vector_generation_id = publication.vector_generation_id
        WHERE publication.project_id = ? AND publication.embedding_profile_id = ?
          AND publication.vector_generation_id = ? AND generation.state = 'FILES_FINALIZED'""",
        (project.project_id, hit.embedding_profile_id, hit.vector_generation_id),
    ).fetchone()
    if row is None:
        raise HybridRetrievalIntegrityError(
            "semantic generation is no longer the published generation"
        )
    sources, snapshot = _current_snapshot(
        connection, project_id=project.project_id, embedding_profile_id=hit.embedding_profile_id
    )
    if (
        str(row[0]) != snapshot
        or _generation_sources(connection, hit.vector_generation_id) != sources
    ):
        raise HybridRetrievalIntegrityError("semantic generation is no longer current")


def _identity_for(hit: RetrievalHit | SemanticRetrievalHit) -> HybridCandidateKey:
    return HybridCandidateKey(
        project_id=hit.project_id,
        document_generation_id=hit.document_generation_id,
        page_id=hit.page_id,
        chunk_id=hit.chunk_id,
    )


def _state_for(hits: tuple[object, ...]) -> HybridChannelState:
    return HybridChannelState.HEALTHY_RESULTS if hits else HybridChannelState.HEALTHY_EMPTY


def _parent_context_from_row(
    connection: sqlite3.Connection, *, row: sqlite3.Row, identity: HybridCandidateKey
) -> HybridParentChunkContext:
    page_text = _required_text(row, "canonical_text")
    chunk_text = _required_text(row, "chunk_text")
    start_offset = int(row["start_offset"])
    end_offset = int(row["end_offset"])
    if not 0 <= start_offset < end_offset <= len(page_text) or (
        page_text[start_offset:end_offset] != chunk_text
    ):
        raise HybridRetrievalIntegrityError("parent chunk does not match its canonical page range")
    anchors = connection.execute(
        """SELECT page_anchor_id, evidence_id, char_start, char_end, anchor_text,
        anchor_text_sha256, boxes_sha256, x0, top, x1, bottom FROM page_anchors
        WHERE page_id = ? AND char_start < ? AND char_end > ?
        ORDER BY char_start, char_end, page_anchor_id""",
        (identity.page_id, end_offset, start_offset),
    ).fetchall()
    if not anchors or any(
        int(anchor["char_start"]) < start_offset or int(anchor["char_end"]) > end_offset
        for anchor in anchors
    ):
        raise HybridRetrievalIntegrityError("parent chunk has invalid range-scoped anchors")
    for anchor in anchors:
        anchor_from_row(row, anchor)
    return HybridParentChunkContext(
        identity=identity,
        paper_id=str(row["paper_id"]),
        file_version_id=str(row["file_version_id"]),
        physical_page_index=int(row["physical_page_index"]),
        display_page_number=int(row["display_page_number"]),
        printed_page_label=row["printed_page_label"],
        chunk_ordinal=int(row["chunk_ordinal"]),
        start_offset=start_offset,
        end_offset=end_offset,
        chunk_text=chunk_text,
    )


_PARENT_CHUNK_SQL = """
    SELECT projects.project_id, papers.paper_id, file_versions.file_version_id,
        file_versions.sha256 AS source_pdf_sha256, document_generations.document_generation_id,
        pages.page_id, pages.physical_page_index, pages.page_number AS display_page_number,
        pages.printed_page_label, pages.canonical_text, pages.canonical_text_sha256,
        pages.parser_profile_sha256, pages.page_width_points, pages.page_height_points,
        pages.source_page_rotation_degrees, chunks.chunk_id, chunks.ordinal AS chunk_ordinal,
        chunks.start_offset, chunks.end_offset, chunks.chunk_text,
        chunks.chunk_text AS indexed_chunk_text, CAST(0.0 AS REAL) AS raw_bm25_score
    FROM chunks JOIN generation_publications
      ON generation_publications.document_generation_id = chunks.document_generation_id
    JOIN document_generations
      ON document_generations.document_generation_id = chunks.document_generation_id
      AND document_generations.file_version_id = generation_publications.file_version_id
    JOIN file_versions ON file_versions.file_version_id = document_generations.file_version_id
    JOIN papers ON papers.paper_id = file_versions.paper_id
    JOIN projects ON projects.project_id = papers.project_id
    JOIN pages ON pages.page_id = chunks.page_id
      AND pages.document_generation_id = chunks.document_generation_id
    WHERE projects.project_id = ? AND chunks.document_generation_id = ?
      AND chunks.page_id = ? AND chunks.chunk_id = ?
"""
