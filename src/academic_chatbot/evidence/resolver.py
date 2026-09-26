"""Authoritative, bounded evidence reconstruction in one read-only snapshot."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import stat
from pathlib import Path
from typing import Literal, NoReturn, cast

from pydantic import ValidationError

from academic_chatbot.db.connection import DatabasePathError, open_read_only_connection
from academic_chatbot.documents.chunking import LexicalChunk, _chunk_id
from academic_chatbot.documents.import_service import (
    document_generation_id_for,
    processing_profile_id_for,
)
from academic_chatbot.domain.library import file_version_id_for
from academic_chatbot.embeddings.models import (
    EmbeddingProfile,
    EmbeddingSpanIdentity,
    canonical_json_bytes,
)
from academic_chatbot.embeddings.vector_build import _empty_payload
from academic_chatbot.evidence import models as m
from academic_chatbot.library.repository import _page_anchor_id_for, _page_id_for
from academic_chatbot.retrieval.exact_memmap import (
    ExactVectorStore,
    VectorOpenLimitError,
    VectorOpenLimits,
)
from academic_chatbot.retrieval.semantic import _current_snapshot, _generation_sources, _mapping
from academic_chatbot.retrieval.service import RetrievalIntegrityError, anchor_from_row
from academic_chatbot.storage.paths import ProjectPaths


class EvidenceResolutionError(ValueError):
    """Exception envelope containing only the frozen, serializable safe error."""

    def __init__(self, code: m.EvidenceErrorCode, position: int | None = None) -> None:
        self.error = m.EvidenceBundleError(code=code, input_position=position)
        super().__init__(self.error.message)


def _fail(code: m.EvidenceErrorCode, position: int | None = None) -> NoReturn:
    raise EvidenceResolutionError(code, position) from None


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _guard(path: Path, *, file: bool, unavailable: m.EvidenceErrorCode) -> None:
    """Inspect the un-resolved path so containment cannot erase a reparse entry."""
    try:
        metadata = path.lstat()
    except OSError:
        _fail(unavailable)
    if stat.S_ISLNK(metadata.st_mode) or (
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        _fail(m.EvidenceErrorCode.UNSAFE_PATH)
    if not (stat.S_ISREG(metadata.st_mode) if file else stat.S_ISDIR(metadata.st_mode)):
        _fail(m.EvidenceErrorCode.UNSAFE_PATH)


def _validate_database_path(root: Path, project_id: str) -> ProjectPaths:
    if ".." in root.parts:
        _fail(m.EvidenceErrorCode.UNSAFE_PATH)
    absolute = root.absolute()
    for component in (*reversed(absolute.parents), absolute):
        _guard(component, file=False, unavailable=m.EvidenceErrorCode.STORAGE_UNAVAILABLE)
    try:
        paths = ProjectPaths.create(absolute, project_id=project_id)
    except ValueError:
        _fail(m.EvidenceErrorCode.UNSAFE_PATH)
    for path in (absolute / "projects", absolute / "projects" / project_id):
        _guard(path, file=False, unavailable=m.EvidenceErrorCode.STORAGE_UNAVAILABLE)
    _guard(paths.database_path, file=True, unavailable=m.EvidenceErrorCode.STORAGE_UNAVAILABLE)
    return paths


def _count(connection: sqlite3.Connection, sql: str, args: tuple[object, ...], cap: int) -> None:
    if connection.execute(sql, args).fetchone()[0] > cap:
        _fail(m.EvidenceErrorCode.RESOURCE_LIMIT)


def _resolve_scope(
    connection: sqlite3.Connection, scope: m.BundleSourceScope
) -> tuple[m.ResolvedBundleSource, str]:
    project = connection.execute(
        "SELECT project_id FROM projects WHERE project_id = ?", (scope.project_id,)
    ).fetchone()
    paper = connection.execute(
        "SELECT project_id FROM papers WHERE paper_id = ?", (scope.paper_id,)
    ).fetchone()
    version = connection.execute(
        "SELECT paper_id, sha256 FROM file_versions WHERE file_version_id = ?",
        (scope.file_version_id,),
    ).fetchone()
    generation = connection.execute(
        "SELECT file_version_id, pipeline_version FROM document_generations "
        "WHERE document_generation_id = ?",
        (scope.document_generation_id,),
    ).fetchone()
    if any(row is None for row in (project, paper, version, generation)):
        _fail(m.EvidenceErrorCode.SOURCE_NOT_FOUND)
    if (
        paper[0] != scope.project_id
        or version[0] != scope.paper_id
        or generation[0] != scope.file_version_id
    ):
        _fail(m.EvidenceErrorCode.SOURCE_SCOPE_MISMATCH)
    if (
        file_version_id_for(paper_id=scope.paper_id, sha256=version[1]) != scope.file_version_id
        or document_generation_id_for(
            file_version_id=scope.file_version_id, processing_profile_id=generation[1]
        )
        != scope.document_generation_id
    ):
        _fail(m.EvidenceErrorCode.STORAGE_INTEGRITY)
    publication = connection.execute(
        "SELECT document_generation_id FROM generation_publications WHERE file_version_id = ?",
        (scope.file_version_id,),
    ).fetchone()
    if publication is None or publication[0] != scope.document_generation_id:
        _fail(m.EvidenceErrorCode.GENERATION_NOT_CURRENT)
    return m.ResolvedBundleSource(**scope.model_dump(), source_pdf_sha256=version[1]), generation[1]


def _read_coverage(
    connection: sqlite3.Connection, request: m.EvidenceBundleRequest
) -> m.EvidenceCoverage:
    args = (request.scope.document_generation_id,)
    _count(
        connection,
        "SELECT count(*) FROM pages WHERE document_generation_id = ?",
        args,
        m.MAX_SOURCE_PAGES,
    )
    rows = connection.execute(
        """SELECT p.page_id, p.physical_page_index, p.page_number, p.printed_page_label,
        p.extraction_quality, p.needs_ocr, EXISTS(SELECT 1 FROM chunks c
        WHERE c.page_id = p.page_id AND c.document_generation_id = p.document_generation_id)
        FROM pages p WHERE p.document_generation_id = ?
        ORDER BY COALESCE(p.physical_page_index, p.page_number - 1), p.page_id""",
        args,
    ).fetchall()
    categories: dict[str, Literal["adequate", "low", "empty", "unknown"]] = {
        "adequate_native_text": "adequate",
        "low_native_text": "low",
        "empty_native_text": "empty",
    }
    pages = tuple(
        m.CoveragePage(
            page_id=r[0],
            physical_page_index=r[1],
            display_page_number=r[2],
            printed_page_label=r[3],
            extraction_category=categories.get(r[4], "unknown"),
            needs_ocr=None if r[5] is None else bool(r[5]),
            has_chunks=bool(r[6]),
        )
        for r in rows
    )
    flagged = tuple(
        p.page_id for p in pages if p.extraction_category in ("low", "empty") or p.needs_ocr
    )
    unknown = any(
        p.extraction_category == "unknown" or p.needs_ocr is None or p.physical_page_index is None
        for p in pages
    )
    status = (
        m.CoverageStatus.KNOWN_NATIVE_GAPS
        if flagged
        else m.CoverageStatus.UNKNOWN_METADATA
        if unknown
        else m.CoverageStatus.NO_FLAGGED_NATIVE_GAPS
    )
    return m.EvidenceCoverage(
        status=status,
        pages=pages,
        candidate_page_ids=tuple(dict.fromkeys(c.parent.page_id for c in request.candidates)),
        packed_page_ids=(),
        flagged_page_ids=flagged,
        excluded_page_ids=tuple(p.page_id for p in pages if not p.has_chunks),
    )


def _read_registered_profile(connection: sqlite3.Connection, profile_id: str) -> EmbeddingProfile:
    size = connection.execute(
        "SELECT length(CAST(canonical_profile_json AS BLOB)) FROM embedding_profiles "
        "WHERE embedding_profile_id = ?",
        (profile_id,),
    ).fetchone()
    if size is None:
        _fail(m.EvidenceErrorCode.PROFILE_MISMATCH)
    if size[0] > m.MAX_PROFILE_BYTES:
        _fail(m.EvidenceErrorCode.RESOURCE_LIMIT)
    row = connection.execute(
        "SELECT * FROM embedding_profiles WHERE embedding_profile_id = ?", (profile_id,)
    ).fetchone()
    try:
        raw = row["canonical_profile_json"].encode("utf-8")
        profile = EmbeddingProfile.model_validate(json.loads(raw))
        canonical = canonical_json_bytes(
            profile.model_dump(mode="json", exclude={"embedding_profile_id"})
        )
        if (
            raw != canonical
            or hashlib.sha256(raw).hexdigest() != row["canonical_profile_sha256"]
            or profile.embedding_profile_id != profile_id
            or profile.dimension != 384
            or row["dimension"] != profile.dimension
            or row["span_policy_id"] != profile.span_policy
            or re.fullmatch("[0-9a-f]{64}", row["artifact_manifest_sha256"]) is None
        ):
            _fail(m.EvidenceErrorCode.PROFILE_MISMATCH)
    except (ValueError, TypeError, AttributeError):
        _fail(m.EvidenceErrorCode.PROFILE_MISMATCH)
    return profile


def _validate_vector_generation(
    connection: sqlite3.Connection, paths: ProjectPaths, request: m.EvidenceBundleRequest
) -> frozenset[str]:
    origin = request.origin
    assert origin.embedding_profile_id is not None and origin.vector_generation_id is not None
    profile = _read_registered_profile(connection, origin.embedding_profile_id)
    publication = connection.execute(
        "SELECT vector_generation_id FROM vector_generation_publications "
        "WHERE project_id = ? AND embedding_profile_id = ?",
        (request.scope.project_id, origin.embedding_profile_id),
    ).fetchone()
    if publication is None:
        _fail(m.EvidenceErrorCode.VECTOR_UNAVAILABLE)
    if publication[0] != origin.vector_generation_id:
        _fail(m.EvidenceErrorCode.VECTOR_NOT_CURRENT)
    row = connection.execute(
        "SELECT * FROM vector_generations WHERE vector_generation_id = ?",
        (origin.vector_generation_id,),
    ).fetchone()
    if (
        row is None
        or row["project_id"] != request.scope.project_id
        or row["embedding_profile_id"] != origin.embedding_profile_id
        or row["state"] != "FILES_FINALIZED"
    ):
        _fail(m.EvidenceErrorCode.VECTOR_INTEGRITY)
    _count(
        connection,
        """SELECT count(*) FROM file_versions fv JOIN papers p ON p.paper_id = fv.paper_id
        JOIN generation_publications gp ON gp.file_version_id = fv.file_version_id
        WHERE p.project_id = ?""",
        (request.scope.project_id,),
        m.MAX_SOURCE_SNAPSHOT_ROWS,
    )
    _count(
        connection,
        "SELECT count(*) FROM vector_generation_sources WHERE vector_generation_id = ?",
        (origin.vector_generation_id,),
        m.MAX_SOURCE_SNAPSHOT_ROWS,
    )
    sources, snapshot = _current_snapshot(
        connection,
        project_id=request.scope.project_id,
        embedding_profile_id=origin.embedding_profile_id,
    )
    if (
        snapshot != row["source_snapshot_sha256"]
        or sources != _generation_sources(connection, origin.vector_generation_id)
        or not any(
            s.file_version_id == request.scope.file_version_id
            and s.document_generation_id == request.scope.document_generation_id
            for s in sources
        )
    ):
        _fail(m.EvidenceErrorCode.VECTOR_NOT_CURRENT)
    _count(
        connection,
        "SELECT count(*) FROM vector_generation_spans WHERE vector_generation_id = ?",
        (origin.vector_generation_id,),
        m.MAX_VECTOR_ROWS,
    )
    mapping = _mapping(connection, origin.vector_generation_id)
    if (
        tuple(i for i, _ in mapping) != tuple(range(len(mapping)))
        or len(set(s for _, s in mapping)) != len(mapping)
        or row["embeddable_spans"] != len(mapping)
    ):
        _fail(m.EvidenceErrorCode.VECTOR_INTEGRITY)
    # Every mapped row must be an admissible span belonging to this exact source snapshot.
    valid = connection.execute(
        """SELECT count(*) FROM vector_generation_spans vm
        JOIN embedding_spans s ON s.embedding_span_id = vm.embedding_span_id
        JOIN chunks c ON c.chunk_id = s.chunk_id AND c.page_id = s.page_id
          AND c.document_generation_id = s.document_generation_id
        JOIN pages p ON p.page_id = c.page_id
          AND p.document_generation_id = c.document_generation_id
        JOIN vector_generation_sources gs ON gs.vector_generation_id = vm.vector_generation_id
          AND gs.document_generation_id = s.document_generation_id
        WHERE vm.vector_generation_id = ? AND s.embedding_profile_id = ?
          AND s.coverage_status = 'EMBEDDABLE'
          AND s.start_offset >= c.start_offset AND s.end_offset <= c.end_offset
          AND s.start_offset < s.end_offset""",
        (origin.vector_generation_id, origin.embedding_profile_id),
    ).fetchone()[0]
    if valid != len(mapping):
        _fail(m.EvidenceErrorCode.VECTOR_INTEGRITY)
    span_rows = connection.execute(
        """SELECT s.*, vm.vector_row FROM vector_generation_spans vm
        JOIN embedding_spans s ON s.embedding_span_id = vm.embedding_span_id
        WHERE vm.vector_generation_id = ? ORDER BY vm.vector_row""",
        (origin.vector_generation_id,),
    ).fetchall()
    indexed = set()
    for span in span_rows:
        identity = EmbeddingSpanIdentity(
            **{
                name: span[name]
                for name in (
                    "document_generation_id",
                    "chunk_id",
                    "page_id",
                    "start_offset",
                    "end_offset",
                    "embedding_profile_id",
                )
            }
        )
        if (
            identity.embedding_span_id != span["embedding_span_id"]
            or type(span["vector_row"]) is not int
        ):
            _fail(m.EvidenceErrorCode.VECTOR_INTEGRITY)
        indexed.add(span["document_generation_id"])
    totals = connection.execute(
        """SELECT count(CASE WHEN s.coverage_status = 'EMBEDDABLE' THEN 1 END),
        count(CASE WHEN s.coverage_status = 'EXCLUDED_UNEMBEDDABLE' THEN 1 END)
        FROM embedding_spans s JOIN vector_generation_sources gs
        ON gs.document_generation_id = s.document_generation_id
        WHERE gs.vector_generation_id = ? AND s.embedding_profile_id = ?""",
        (origin.vector_generation_id, origin.embedding_profile_id),
    ).fetchone()
    expected_counts = {
        "eligible_native_chunks": sum(s.eligible_native_chunk_count for s in sources),
        "embeddable_spans": totals[0],
        "excluded_unembeddable_spans": totals[1],
        "needs_ocr_pages": sum(s.needs_ocr_page_count for s in sources),
        "indexed_documents": len(indexed),
        "unindexed_documents": len(sources) - len(indexed),
    }
    if totals[0] != len(mapping) or any(
        row[name] != value for name, value in expected_counts.items()
    ):
        _fail(m.EvidenceErrorCode.VECTOR_INTEGRITY)
    relative = row["artifact_relative_dir"]
    parts = relative.split("/")
    if (
        len(parts) != 6
        or parts[:4] != ["indexes", "semantic", origin.embedding_profile_id, snapshot]
        or parts[4] not in ("generations", "empty-generations")
        or any(not part or part in (".", "..") or "\\" in part or ":" in part for part in parts)
    ):
        _fail(m.EvidenceErrorCode.UNSAFE_PATH)
    artifact = paths.project_root
    for part in parts:
        artifact /= part
        _guard(artifact, file=False, unavailable=m.EvidenceErrorCode.VECTOR_UNAVAILABLE)
    empty = parts[4] == "empty-generations"
    for filename in (
        ("empty-generation.json",)
        if empty
        else ("manifest.json", "vectors.meta.json", "vectors.npy")
    ):
        _guard(artifact / filename, file=True, unavailable=m.EvidenceErrorCode.VECTOR_UNAVAILABLE)
    try:
        if empty:
            if mapping:
                _fail(m.EvidenceErrorCode.VECTOR_INTEGRITY)
            admissible = connection.execute(
                """SELECT count(*) FROM embedding_spans s JOIN vector_generation_sources gs
                ON gs.document_generation_id = s.document_generation_id
                WHERE gs.vector_generation_id = ?
                AND s.embedding_profile_id = ? AND s.coverage_status = 'EMBEDDABLE'""",
                (origin.vector_generation_id, origin.embedding_profile_id),
            ).fetchone()[0]
            if admissible:
                _fail(m.EvidenceErrorCode.VECTOR_INTEGRITY)
            with (artifact / "empty-generation.json").open("rb") as handle:
                raw = handle.read(m.MAX_MANIFEST_BYTES + 1)
            if len(raw) > m.MAX_MANIFEST_BYTES:
                _fail(m.EvidenceErrorCode.RESOURCE_LIMIT)
            payload = _empty_payload(
                profile_sha256=profile.embedding_profile_id.removeprefix("ep-sha256-"),
                source_snapshot_sha256=snapshot,
            )
            if (
                raw != canonical_json_bytes(payload) + b"\n"
                or payload["manifest_sha256"] != row["vector_store_manifest_sha256"]
            ):
                _fail(m.EvidenceErrorCode.VECTOR_INTEGRITY)
        else:
            limits = VectorOpenLimits(
                max_manifest_bytes=m.MAX_MANIFEST_BYTES,
                max_metadata_bytes=m.MAX_VECTOR_METADATA_BYTES,
                max_vectors_file_bytes=m.MAX_VECTOR_FILE_BYTES,
                max_rows=m.MAX_VECTOR_ROWS,
                expected_dimension=384,
            )
            store = ExactVectorStore.open(artifact, limits=limits)
            primary = False
            try:
                if (
                    store.manifest.manifest_sha256 != row["vector_store_manifest_sha256"]
                    or store.manifest.profile_sha256
                    != profile.embedding_profile_id.removeprefix("ep-sha256-")
                    or tuple(enumerate(store.row_ids)) != mapping
                ):
                    _fail(m.EvidenceErrorCode.VECTOR_INTEGRITY)
            except BaseException:
                primary = True
                raise
            finally:
                try:
                    store.close()
                except Exception:
                    if not primary:
                        _fail(m.EvidenceErrorCode.VECTOR_INTEGRITY)
    except EvidenceResolutionError:
        raise
    except VectorOpenLimitError:
        _fail(m.EvidenceErrorCode.RESOURCE_LIMIT)
    except OSError:
        _fail(m.EvidenceErrorCode.VECTOR_UNAVAILABLE)
    except (ValueError, TypeError):
        _fail(m.EvidenceErrorCode.VECTOR_INTEGRITY)
    return frozenset(span for _, span in mapping)


def _admit_pages(connection: sqlite3.Connection, request: m.EvidenceBundleRequest) -> None:
    total = 0
    for page_id in dict.fromkeys(c.parent.page_id for c in request.candidates):
        row = connection.execute(
            "SELECT document_generation_id, length(CAST(canonical_text AS BLOB)) "
            "FROM pages WHERE page_id = ?",
            (page_id,),
        ).fetchone()
        if row is None:
            _fail(m.EvidenceErrorCode.REFERENCE_NOT_FOUND)
        if row[0] != request.scope.document_generation_id:
            _fail(m.EvidenceErrorCode.SOURCE_SCOPE_MISMATCH)
        if row[1] is None:
            _fail(m.EvidenceErrorCode.STORAGE_INTEGRITY)
        total += row[1]
        if row[1] > m.MAX_PAGE_TEXT_BYTES or total > m.MAX_RESOLVED_TEXT_BYTES:
            _fail(m.EvidenceErrorCode.RESOURCE_LIMIT)


def _read_page(
    connection: sqlite3.Connection, page_id: str, scope: m.ResolvedBundleSource
) -> sqlite3.Row:
    row = connection.execute(
        """SELECT p.*, p.page_number AS display_page_number,
        fv.file_version_id, fv.sha256 AS source_pdf_sha256 FROM pages p
        JOIN document_generations dg ON dg.document_generation_id = p.document_generation_id
        JOIN file_versions fv ON fv.file_version_id = dg.file_version_id WHERE p.page_id = ?""",
        (page_id,),
    ).fetchone()
    if (
        row is None
        or row["physical_page_index"] is None
        or row["page_id"]
        != _page_id_for(
            document_generation_id=scope.document_generation_id,
            physical_page_index=row["physical_page_index"],
        )
        or _digest(row["canonical_text"]) != row["canonical_text_sha256"]
    ):
        _fail(m.EvidenceErrorCode.STORAGE_INTEGRITY)
    return cast(sqlite3.Row, row)


def _resolve_anchors(
    connection: sqlite3.Connection, page: sqlite3.Row, start: int, end: int
) -> tuple[m.BundleAnchor, ...]:
    lengths = connection.execute(
        """SELECT length(CAST(anchor_text AS BLOB)) FROM page_anchors WHERE page_id = ?
        AND char_start >= ? AND char_end <= ?
        ORDER BY char_start, char_end, page_anchor_id LIMIT ?""",
        (page["page_id"], start, end, m.MAX_CHUNK_ANCHORS + 1),
    ).fetchall()
    if len(lengths) > m.MAX_CHUNK_ANCHORS:
        _fail(m.EvidenceErrorCode.RESOURCE_LIMIT)
    # A valid single-space anchor sequence cannot contain more source bytes
    # than its admitted page. Corrupt denormalized text must not bypass that cap.
    if sum(length[0] for length in lengths) > m.MAX_PAGE_TEXT_BYTES:
        _fail(m.EvidenceErrorCode.RESOURCE_LIMIT)
    rows = connection.execute(
        """SELECT * FROM page_anchors WHERE page_id = ?
        AND char_start >= ? AND char_end <= ?
        ORDER BY char_start, char_end, page_anchor_id LIMIT ?""",
        (page["page_id"], start, end, m.MAX_CHUNK_ANCHORS + 1),
    ).fetchall()
    if len(rows) > m.MAX_CHUNK_ANCHORS:
        _fail(m.EvidenceErrorCode.RESOURCE_LIMIT)
    cursor = start
    anchors = []
    for row in rows:
        anchor = anchor_from_row(page, row)
        if (
            row["page_anchor_id"]
            != _page_anchor_id_for(page_id=page["page_id"], evidence_id=anchor.evidence_id)
            or anchor.char_start != cursor
        ):
            _fail(m.EvidenceErrorCode.STORAGE_INTEGRITY)
        cursor = anchor.char_end + 1
        anchors.append(
            m.BundleAnchor(
                page_anchor_id=row["page_anchor_id"],
                evidence_id=anchor.evidence_id,
                char_start=anchor.char_start,
                char_end=anchor.char_end,
                anchor_text=anchor.anchor_text,
                anchor_text_sha256=anchor.anchor_text_sha256,
                boxes_sha256=anchor.boxes_sha256,
                boxes=anchor.boxes,
            )
        )
    if (
        not anchors
        or cursor - 1 != end
        or " ".join(a.anchor_text for a in anchors) != page["canonical_text"][start:end]
    ):
        _fail(m.EvidenceErrorCode.STORAGE_INTEGRITY)
    return tuple(anchors)


def _resolve_parent(
    connection: sqlite3.Connection,
    reference: m.EvidenceCandidateRef,
    page: sqlite3.Row,
    pipeline: str,
) -> sqlite3.Row:
    admission = connection.execute(
        "SELECT page_id, document_generation_id, length(CAST(chunk_text AS BLOB)) "
        "FROM chunks WHERE chunk_id = ?",
        (reference.parent.chunk_id,),
    ).fetchone()
    if admission is None:
        _fail(m.EvidenceErrorCode.REFERENCE_NOT_FOUND)
    if (
        admission[0] != reference.parent.page_id
        or admission[1] != reference.parent.document_generation_id
    ):
        _fail(m.EvidenceErrorCode.SOURCE_SCOPE_MISMATCH)
    if admission[2] > m.MAX_PAGE_TEXT_BYTES:
        _fail(m.EvidenceErrorCode.RESOURCE_LIMIT)
    row = connection.execute(
        "SELECT * FROM chunks WHERE chunk_id = ?", (reference.parent.chunk_id,)
    ).fetchone()
    if row is None:
        _fail(m.EvidenceErrorCode.REFERENCE_NOT_FOUND)
    if (
        row["page_id"] != reference.parent.page_id
        or row["document_generation_id"] != reference.parent.document_generation_id
    ):
        _fail(m.EvidenceErrorCode.SOURCE_SCOPE_MISMATCH)
    if (
        processing_profile_id_for(lexical_chunk_profile_id=row["processing_profile_id"]) != pipeline
        or row["chunk_id"]
        != _chunk_id(
            document_generation_id=row["document_generation_id"],
            page_id=row["page_id"],
            start_offset=row["start_offset"],
            end_offset=row["end_offset"],
            processing_profile_id=row["processing_profile_id"],
        )
        or not 0 <= row["start_offset"] < row["end_offset"] <= len(page["canonical_text"])
        or page["canonical_text"][row["start_offset"] : row["end_offset"]] != row["chunk_text"]
    ):
        _fail(m.EvidenceErrorCode.STORAGE_INTEGRITY)
    LexicalChunk(
        chunk_id=row["chunk_id"],
        document_generation_id=row["document_generation_id"],
        page_id=row["page_id"],
        ordinal=row["ordinal"],
        start_offset=row["start_offset"],
        end_offset=row["end_offset"],
        text=row["chunk_text"],
        lexical_word_count=row["lexical_word_count"],
        processing_profile_id=row["processing_profile_id"],
    )
    fts = connection.execute(
        "SELECT chunk_text FROM chunk_fts WHERE chunk_id = ? LIMIT 2", (row["chunk_id"],)
    ).fetchall()
    if len(fts) != 1 or fts[0][0] != row["chunk_text"]:
        _fail(m.EvidenceErrorCode.STORAGE_INTEGRITY)
    return cast(sqlite3.Row, row)


def _range(
    page: sqlite3.Row,
    chunk: sqlite3.Row,
    reference: m.EvidenceCandidateRef,
    scope: m.ResolvedBundleSource,
    assertion: m.LexicalRangeRef,
    anchors: tuple[m.BundleAnchor, ...],
    provenance: m.ContributionProvenance,
) -> m.ResolvedEvidenceRange:
    text = page["canonical_text"][assertion.start_offset : assertion.end_offset]
    digest = _digest(text)
    if digest != assertion.expected_text_sha256:
        _fail(m.EvidenceErrorCode.TEXT_DIGEST_MISMATCH)
    ids = tuple(a.page_anchor_id for a in anchors)
    if ids != assertion.expected_anchor_ids:
        _fail(m.EvidenceErrorCode.ANCHOR_MISMATCH)
    return m.ResolvedEvidenceRange(
        text=text,
        anchors=anchors,
        contributions=(provenance,),
        source=m.EvidenceSourceRef(
            scope=scope,
            parent=reference.parent,
            physical_page_index=page["physical_page_index"],
            display_page_number=page["display_page_number"],
            printed_page_label=page["printed_page_label"],
            canonical_page_text_sha256=page["canonical_text_sha256"],
            parser_profile_sha256=page["parser_profile_sha256"],
            processing_profile_id=chunk["processing_profile_id"],
            page_width_points=page["page_width_points"],
            page_height_points=page["page_height_points"],
            source_page_rotation_degrees=page["source_page_rotation_degrees"],
            start_offset=assertion.start_offset,
            end_offset=assertion.end_offset,
            text_sha256=digest,
            anchor_ids=ids,
            provenance=(provenance,),
        ),
    )


def _resolve_group(
    connection: sqlite3.Connection,
    request: m.EvidenceBundleRequest,
    scope: m.ResolvedBundleSource,
    pipeline: str,
    position: int,
    page: sqlite3.Row,
    mapping: frozenset[str],
) -> m.ResolvedCandidateGroup:
    ref = request.candidates[position]
    chunk = _resolve_parent(connection, ref, page, pipeline)
    start, end = chunk["start_offset"], chunk["end_offset"]
    parent_anchors = _resolve_anchors(connection, page, start, end)
    ranges = []
    context = None
    if ref.lexical is not None:
        if (ref.lexical.start_offset, ref.lexical.end_offset) != (start, end):
            _fail(m.EvidenceErrorCode.RANGE_MISMATCH)
        ranges.append(
            _range(
                page,
                chunk,
                ref,
                scope,
                ref.lexical,
                parent_anchors,
                m.ContributionProvenance(channel="lexical"),
            )
        )
    if ref.semantic is not None:
        assertion = ref.semantic
        span = connection.execute(
            "SELECT * FROM embedding_spans WHERE embedding_span_id = ?",
            (assertion.embedding_span_id,),
        ).fetchone()
        if span is None:
            _fail(m.EvidenceErrorCode.REFERENCE_NOT_FOUND)
        if span["embedding_profile_id"] != request.origin.embedding_profile_id:
            _fail(m.EvidenceErrorCode.PROFILE_MISMATCH)
        if any(
            span[name] != getattr(ref.parent, name)
            for name in ("chunk_id", "page_id", "document_generation_id")
        ):
            _fail(m.EvidenceErrorCode.SOURCE_SCOPE_MISMATCH)
        if (span["start_offset"], span["end_offset"]) != (
            assertion.start_offset,
            assertion.end_offset,
        ):
            _fail(m.EvidenceErrorCode.RANGE_MISMATCH)
        identity = EmbeddingSpanIdentity(
            **{
                name: span[name]
                for name in (
                    "document_generation_id",
                    "chunk_id",
                    "page_id",
                    "start_offset",
                    "end_offset",
                    "embedding_profile_id",
                )
            }
        )
        if (
            identity.embedding_span_id != assertion.embedding_span_id
            or span["coverage_status"] != "EMBEDDABLE"
            or not start <= span["start_offset"] < span["end_offset"] <= end
        ):
            _fail(m.EvidenceErrorCode.STORAGE_INTEGRITY)
        if assertion.embedding_span_id not in mapping:
            _fail(m.EvidenceErrorCode.REFERENCE_NOT_FOUND)
        anchors = _resolve_anchors(connection, page, assertion.start_offset, assertion.end_offset)
        ranges.append(
            _range(
                page,
                chunk,
                ref,
                scope,
                assertion,
                anchors,
                m.ContributionProvenance(
                    channel="semantic",
                    embedding_profile_id=request.origin.embedding_profile_id,
                    vector_generation_id=request.origin.vector_generation_id,
                    embedding_span_id=assertion.embedding_span_id,
                ),
            )
        )
        if ref.lexical is None and (assertion.start_offset, assertion.end_offset) != (start, end):
            context = m.ResolvedContext(
                parent=ref.parent,
                physical_page_index=page["physical_page_index"],
                start_offset=start,
                end_offset=end,
                text=chunk["chunk_text"],
                text_sha256=_digest(chunk["chunk_text"]),
                anchors=parent_anchors,
            )
    return m.ResolvedCandidateGroup(
        input_position=position, reference=ref, ranges=tuple(ranges), context=context
    )


class EvidenceReadResolver:
    """Resolve assertions without searching, loading models, or writing storage."""

    def __init__(self, *, data_root: Path) -> None:
        self._data_root = Path(data_root)

    def resolve(self, request: m.EvidenceBundleRequest) -> m.ResolvedEvidenceInput:
        try:
            request = m.EvidenceBundleRequest.model_validate(request)
        except (ValidationError, ValueError, TypeError):
            _fail(m.EvidenceErrorCode.INVALID_REQUEST)
        paths = _validate_database_path(self._data_root, request.scope.project_id)
        try:
            connection = open_read_only_connection(paths.database_path, data_root=paths.data_root)
        except (DatabasePathError, OSError, sqlite3.Error):
            _fail(m.EvidenceErrorCode.STORAGE_UNAVAILABLE)
        primary = False
        position = None
        try:
            connection.execute("BEGIN")
            scope, pipeline = _resolve_scope(connection, request.scope)
            coverage = _read_coverage(connection, request)
            mapping = (
                _validate_vector_generation(connection, paths, request)
                if request.origin.mode != "lexical"
                else frozenset()
            )
            for ref in request.candidates:
                if (
                    ref.parent.project_id != scope.project_id
                    or ref.parent.document_generation_id != scope.document_generation_id
                ):
                    _fail(m.EvidenceErrorCode.SOURCE_SCOPE_MISMATCH)
            _admit_pages(connection, request)
            cache: dict[str, sqlite3.Row] = {}
            groups = []
            for position, reference in enumerate(request.candidates):
                page_id = reference.parent.page_id
                if page_id not in cache:
                    cache[page_id] = _read_page(connection, page_id, scope)
                groups.append(
                    _resolve_group(
                        connection, request, scope, pipeline, position, cache[page_id], mapping
                    )
                )
            result = m.ResolvedEvidenceInput(
                request=request, resolved_scope=scope, coverage=coverage, groups=tuple(groups)
            )
        except EvidenceResolutionError as error:
            primary = True
            raise EvidenceResolutionError(error.error.code, position) from None
        except (
            sqlite3.Error,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
            RetrievalIntegrityError,
        ):
            primary = True
            _fail(m.EvidenceErrorCode.STORAGE_INTEGRITY, position)
        except BaseException:
            primary = True
            raise
        finally:
            cleanup_failed = False
            try:
                connection.rollback()
            except Exception:
                cleanup_failed = True
            try:
                connection.close()
            except Exception:
                cleanup_failed = True
            if cleanup_failed and not primary:
                _fail(m.EvidenceErrorCode.STORAGE_UNAVAILABLE)
        return result
