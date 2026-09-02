"""Project-local construction and publication of immutable semantic vector generations."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from academic_chatbot.embeddings.models import EmbeddingProfile, EmbeddingRole, canonical_json_bytes
from academic_chatbot.embeddings.repository import (
    ActiveChunkSource,
    EmbeddingPersistenceError,
    EmbeddingRepository,
    EmbeddingSpan,
    SourceSnapshot,
    VectorGeneration,
    VectorGenerationCoverage,
    VectorGenerationState,
)
from academic_chatbot.embeddings.spans import (
    CanonicalWordRange,
    ChunkSourceEvidence,
    SpanConstructionError,
    construct_document_spans,
    source_text_for_span,
)
from academic_chatbot.retrieval.exact_memmap import ExactVectorStore
from academic_chatbot.storage.paths import PathEscapeError, ProjectPaths

VECTOR_BUILD_BATCH_SIZE = 16
_NORMALIZATION_ATOL = 1e-5
_EMPTY_MANIFEST_FILENAME = "empty-generation.json"


class VectorBuildError(ValueError):
    """Raised when a vector candidate cannot safely become current."""


class _DocumentTokenizer(Protocol):
    @property
    def profile(self) -> EmbeddingProfile: ...

    def prepare(self, role: EmbeddingRole, texts: Sequence[str]) -> object: ...


class _DocumentEmbedder(Protocol):
    @property
    def profile(self) -> EmbeddingProfile: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class VectorBuildResult:
    generation: VectorGeneration
    reused: bool
    empty: bool


@dataclass(frozen=True, slots=True)
class _OrderedSpan:
    source: ChunkSourceEvidence
    span: EmbeddingSpan
    text: str
    file_version_id: str
    physical_page_index: int
    chunk_ordinal: int


@dataclass(frozen=True, slots=True)
class _ConstructedSpans:
    ordered: tuple[_OrderedSpan, ...]
    excluded_count: int


class ProjectVectorBuilder:
    """Build one all-or-nothing, project-contained vector generation."""

    def __init__(
        self,
        *,
        paths: ProjectPaths,
        repository: EmbeddingRepository,
        profile: EmbeddingProfile,
        tokenizer: _DocumentTokenizer,
        embedder: _DocumentEmbedder,
        batch_size: int = VECTOR_BUILD_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("vector build batch size must be positive")
        if tokenizer.profile != profile or embedder.profile != profile:
            raise VectorBuildError(
                "tokenizer and embedder must use the requested embedding profile"
            )
        self._paths = paths
        self._repository = repository
        self._profile = profile
        self._tokenizer = tokenizer
        self._embedder = embedder
        self._batch_size = batch_size

    def build(self, *, project_id: str) -> VectorBuildResult:
        """Build, verify, finalize, and publish one current semantic generation."""

        if project_id != self._paths.project_id:
            raise VectorBuildError("vector builder cannot access another project")
        if self._repository.get_profile(self._profile.embedding_profile_id) != self._profile:
            raise VectorBuildError("requested embedding profile is not registered for this project")
        snapshot = self._repository.current_source_snapshot(
            project_id=project_id, embedding_profile_id=self._profile.embedding_profile_id
        )
        existing = self._repository.generation_for_snapshot(
            project_id=project_id,
            embedding_profile_id=self._profile.embedding_profile_id,
            source_snapshot_sha256=snapshot.source_snapshot_sha256,
        )
        if existing is not None:
            if existing.state is VectorGenerationState.DB_CANDIDATE:
                self._repository.discard_candidate(existing.vector_generation_id)
            elif existing.state is VectorGenerationState.FILES_FINALIZED:
                try:
                    empty = _verify_finalized_artifact(
                        self._paths, self._repository, existing, self._profile
                    )
                except (OSError, PathEscapeError, ValueError) as error:
                    raise VectorBuildError(
                        "authoritative active vector artifact is corrupt"
                    ) from error
                self._repository.publish(existing.vector_generation_id)
                return VectorBuildResult(existing, reused=True, empty=empty)
            else:
                raise VectorBuildError("stale vector generation is not reusable")

        constructed = self._construct_and_persist_spans(project_id=project_id)
        ordered = constructed.ordered
        coverage = _coverage(snapshot, ordered, excluded_count=constructed.excluded_count)
        workspace = _workspace_for(
            self._paths,
            profile_id=self._profile.embedding_profile_id,
            source_snapshot_sha256=snapshot.source_snapshot_sha256,
        )
        try:
            if ordered:
                rows = self._embed_ordered(ordered)
                row_ids = tuple(item.span.embedding_span_id for item in ordered)
                with ExactVectorStore.build(
                    workspace,
                    rows=rows,
                    row_ids=row_ids,
                    profile_sha256=_profile_sha256(self._profile),
                    normalization_atol=_NORMALIZATION_ATOL,
                ) as store:
                    artifact_dir = store.generation_dir
                    manifest_sha256 = store.manifest.manifest_sha256
                empty = False
            else:
                artifact_dir, manifest_sha256 = _finalize_empty_artifact(
                    workspace=workspace,
                    profile_sha256=_profile_sha256(self._profile),
                    source_snapshot_sha256=snapshot.source_snapshot_sha256,
                )
                row_ids = ()
                empty = True
        except VectorBuildError:
            raise
        except (OSError, TypeError, ValueError) as error:
            raise VectorBuildError("vector artifact build failed") from error

        artifact_relative_dir = self._paths.to_relative_posix(artifact_dir)
        candidate = self._repository.create_candidate(
            project_id=project_id,
            embedding_profile_id=self._profile.embedding_profile_id,
            artifact_relative_dir=artifact_relative_dir,
            coverage=coverage,
        )
        if candidate.source_snapshot_sha256 != snapshot.source_snapshot_sha256:
            if candidate.state is VectorGenerationState.DB_CANDIDATE:
                self._repository.discard_candidate(candidate.vector_generation_id)
            raise VectorBuildError("active document source changed during vector build")
        if candidate.state is not VectorGenerationState.DB_CANDIDATE:
            raise VectorBuildError("vector generation candidate is not writable")
        try:
            for vector_row, span_id in enumerate(row_ids):
                self._repository.attach_vector_row(
                    vector_generation_id=candidate.vector_generation_id,
                    vector_row=vector_row,
                    embedding_span_id=span_id,
                )
            if self._repository.vector_row_mapping(candidate.vector_generation_id) != tuple(
                enumerate(row_ids)
            ):
                raise VectorBuildError(
                    "persisted vector row mapping differs from the vector store order"
                )
            verified_empty = _verify_candidate_artifact(
                self._paths,
                candidate,
                self._profile,
                expected_row_ids=row_ids,
            )
            if verified_empty != empty:
                raise VectorBuildError(
                    "vector artifact empty-state does not match candidate coverage"
                )
            finalized = self._repository.finalize_candidate(
                candidate.vector_generation_id,
                vector_store_manifest_sha256=manifest_sha256,
            )
            self._repository.publish(finalized.vector_generation_id)
        except (EmbeddingPersistenceError, OSError, ValueError) as error:
            raise VectorBuildError(
                "vector candidate could not be finalized or published"
            ) from error
        return VectorBuildResult(finalized, reused=False, empty=empty)

    def _construct_and_persist_spans(self, *, project_id: str) -> _ConstructedSpans:
        sources = self._repository.active_chunk_sources(
            project_id=project_id, embedding_profile_id=self._profile.embedding_profile_id
        )
        ordered: list[_OrderedSpan] = []
        excluded_count = 0
        try:
            for active in sources:
                source = _chunk_source(active)
                for span in construct_document_spans(
                    source=source, profile=self._profile, tokenizer=self._tokenizer
                ):
                    self._repository.persist_span(span)
                    if span.status.value == "EMBEDDABLE":
                        ordered.append(
                            _OrderedSpan(
                                source=source,
                                span=span,
                                text=source_text_for_span(source=source, span=span),
                                file_version_id=active.file_version_id,
                                physical_page_index=active.physical_page_index,
                                chunk_ordinal=active.chunk_ordinal,
                            )
                        )
                    else:
                        excluded_count += 1
        except (EmbeddingPersistenceError, SpanConstructionError, ValueError) as error:
            raise VectorBuildError("exact semantic span construction failed") from error
        ordered.sort(
            key=lambda item: (
                item.file_version_id,
                item.span.identity.document_generation_id,
                item.physical_page_index,
                item.chunk_ordinal,
                item.span.identity.start_offset,
                item.span.embedding_span_id,
            )
        )
        return _ConstructedSpans(tuple(ordered), excluded_count)

    def _embed_ordered(self, ordered: Sequence[_OrderedSpan]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(ordered), self._batch_size):
            texts = tuple(item.text for item in ordered[start : start + self._batch_size])
            try:
                vectors = self._embedder.embed_documents(texts)
            except Exception as error:
                raise VectorBuildError("document embedding failed") from error
            _validate_embedding_batch(vectors, count=len(texts), dimension=self._profile.dimension)
            batches.append(vectors)
        return np.concatenate(batches, axis=0, dtype=np.float32)


def _chunk_source(active: ActiveChunkSource) -> ChunkSourceEvidence:
    return ChunkSourceEvidence(
        document_generation_id=active.document_generation_id,
        chunk_id=active.chunk_id,
        page_id=active.page_id,
        page_text=active.page_text,
        chunk_ordinal=active.chunk_ordinal,
        chunk_start_offset=active.chunk_start_offset,
        chunk_end_offset=active.chunk_end_offset,
        words=tuple(
            CanonicalWordRange(
                text=word.text,
                start_offset=word.start_offset,
                end_offset=word.end_offset,
            )
            for word in active.words
        ),
    )


def _coverage(
    snapshot: SourceSnapshot, ordered: Sequence[_OrderedSpan], *, excluded_count: int
) -> VectorGenerationCoverage:
    embedded_generations = {item.span.identity.document_generation_id for item in ordered}
    eligible_chunks = sum(item.eligible_native_chunk_count for item in snapshot.sources)
    needs_ocr_pages = sum(item.needs_ocr_page_count for item in snapshot.sources)
    embeddable = len(ordered)
    return VectorGenerationCoverage(
        eligible_native_chunks=eligible_chunks,
        embeddable_spans=embeddable,
        excluded_unembeddable_spans=excluded_count,
        needs_ocr_pages=needs_ocr_pages,
        indexed_documents=len(embedded_generations),
        unindexed_documents=len(snapshot.sources) - len(embedded_generations),
    )


def _validate_embedding_batch(vectors: object, *, count: int, dimension: int) -> None:
    if not isinstance(vectors, np.ndarray) or vectors.dtype != np.dtype(np.float32):
        raise VectorBuildError("embedder output must be a float32 ndarray")
    if vectors.shape != (count, dimension) or not np.isfinite(vectors).all():
        raise VectorBuildError("embedder output has an invalid shape or non-finite value")
    norms = np.sqrt(np.sum(vectors * vectors, axis=1, dtype=np.float32), dtype=np.float32)
    if not np.allclose(norms, np.float32(1.0), rtol=_NORMALIZATION_ATOL, atol=_NORMALIZATION_ATOL):
        raise VectorBuildError("embedder output must be L2-normalized")


def _profile_sha256(profile: EmbeddingProfile) -> str:
    return profile.embedding_profile_id.removeprefix("ep-sha256-")


def _workspace_for(paths: ProjectPaths, *, profile_id: str, source_snapshot_sha256: str) -> Path:
    try:
        relative = f"indexes/semantic/{profile_id}/{source_snapshot_sha256}"
        raw_workspace = paths.project_root.joinpath(*relative.split("/"))
        current = paths.project_root
        for part in relative.split("/"):
            current = current / part
            if current.exists():
                metadata = current.lstat()
                reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                if (
                    current.is_symlink()
                    or (reparse and metadata.st_file_attributes & reparse)
                    or not stat.S_ISDIR(metadata.st_mode)
                ):
                    raise VectorBuildError("semantic vector workspace contains a reparse entry")
            else:
                current.mkdir()
        workspace = paths.resolve_relative(relative)
        if workspace != raw_workspace.resolve(strict=False):
            raise VectorBuildError("semantic vector workspace resolved unexpectedly")
    except VectorBuildError:
        raise
    except (OSError, PathEscapeError, ValueError) as error:
        raise VectorBuildError("semantic vector workspace is not safely contained") from error
    return workspace


def _empty_payload(*, profile_sha256: str, source_snapshot_sha256: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "semantic-empty-generation-v1",
        "profile_sha256": profile_sha256,
        "source_snapshot_sha256": source_snapshot_sha256,
        "vector_count": 0,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def _finalize_empty_artifact(
    *, workspace: Path, profile_sha256: str, source_snapshot_sha256: str
) -> tuple[Path, str]:
    payload = _empty_payload(
        profile_sha256=profile_sha256, source_snapshot_sha256=source_snapshot_sha256
    )
    manifest_sha256 = str(payload["manifest_sha256"])
    staging_root = workspace / ".empty-staging"
    destination = workspace / "empty-generations" / f"empty-sha256-{manifest_sha256}"
    staging_root.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _verify_empty_artifact(
            destination,
            expected_manifest_sha256=manifest_sha256,
            profile_sha256=profile_sha256,
            source_snapshot_sha256=source_snapshot_sha256,
        )
        return destination, manifest_sha256
    staging = staging_root / str(uuid.uuid4())
    staging.mkdir()
    manifest_path = staging / _EMPTY_MANIFEST_FILENAME
    try:
        with manifest_path.open("wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(staging, destination)
        except OSError:
            if destination.exists():
                _verify_empty_artifact(
                    destination,
                    expected_manifest_sha256=manifest_sha256,
                    profile_sha256=profile_sha256,
                    source_snapshot_sha256=source_snapshot_sha256,
                )
            else:
                raise
        return destination, manifest_sha256
    finally:
        if staging.exists():
            # It is a UUID path created by this invocation beneath the checked workspace.
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()


def _verify_candidate_artifact(
    paths: ProjectPaths,
    generation: VectorGeneration,
    profile: EmbeddingProfile,
    *,
    expected_row_ids: tuple[str, ...],
) -> bool:
    artifact = paths.resolve_relative(generation.artifact_relative_dir)
    expected_profile_sha256 = _profile_sha256(profile)
    empty_path = artifact / _EMPTY_MANIFEST_FILENAME
    if empty_path.exists():
        if expected_row_ids:
            raise VectorBuildError("empty vector artifact cannot contain row identifiers")
        _verify_empty_artifact(
            artifact,
            expected_manifest_sha256=None,
            profile_sha256=expected_profile_sha256,
            source_snapshot_sha256=generation.source_snapshot_sha256,
        )
        return True
    with ExactVectorStore.open(artifact) as store:
        if (
            store.manifest.profile_sha256 != expected_profile_sha256
            or store.manifest.dimension != profile.dimension
            or tuple(store.row_ids) != expected_row_ids
        ):
            raise VectorBuildError("vector store does not match the candidate profile or row order")
    return False


def _verify_finalized_artifact(
    paths: ProjectPaths,
    repository: EmbeddingRepository,
    generation: VectorGeneration,
    profile: EmbeddingProfile,
) -> bool:
    if generation.vector_store_manifest_sha256 is None:
        raise VectorBuildError("finalized vector generation lacks a manifest hash")
    artifact = paths.resolve_relative(generation.artifact_relative_dir)
    empty_path = artifact / _EMPTY_MANIFEST_FILENAME
    if empty_path.exists():
        _verify_empty_artifact(
            artifact,
            expected_manifest_sha256=generation.vector_store_manifest_sha256,
            profile_sha256=_profile_sha256(profile),
            source_snapshot_sha256=generation.source_snapshot_sha256,
        )
        return True
    with ExactVectorStore.open(artifact) as store:
        mapping = repository.vector_row_mapping(generation.vector_generation_id)
        expected_row_ids = tuple(span_id for _, span_id in mapping)
        if (
            store.manifest.manifest_sha256 != generation.vector_store_manifest_sha256
            or store.manifest.profile_sha256 != _profile_sha256(profile)
            or store.manifest.dimension != profile.dimension
            or tuple(store.row_ids) != expected_row_ids
            or tuple(row for row, _ in mapping) != tuple(range(len(mapping)))
        ):
            raise VectorBuildError("finalized vector artifact does not match immutable metadata")
    return False


def _verify_empty_artifact(
    artifact: Path,
    *,
    expected_manifest_sha256: str | None,
    profile_sha256: str,
    source_snapshot_sha256: str,
) -> None:
    try:
        raw = (artifact / _EMPTY_MANIFEST_FILENAME).read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VectorBuildError("empty vector artifact manifest is unreadable") from error
    if not isinstance(payload, dict):
        raise VectorBuildError("empty vector artifact manifest is invalid")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    actual = payload.get("manifest_sha256")
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if (
        raw != canonical_json_bytes(payload) + b"\n"
        or actual != expected
        or (expected_manifest_sha256 is not None and actual != expected_manifest_sha256)
        or unsigned
        != {
            "schema_version": "semantic-empty-generation-v1",
            "profile_sha256": profile_sha256,
            "source_snapshot_sha256": source_snapshot_sha256,
            "vector_count": 0,
        }
    ):
        raise VectorBuildError("empty vector artifact manifest does not match immutable metadata")
