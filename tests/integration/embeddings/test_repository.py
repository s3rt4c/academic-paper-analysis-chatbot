from __future__ import annotations

# ruff: noqa: E501
import sqlite3
from pathlib import Path

import pytest

from academic_chatbot.db.connection import connect_project_database
from academic_chatbot.db.migrations import MigrationRunner
from academic_chatbot.embeddings.models import EmbeddingProfile, EmbeddingSpanIdentity
from academic_chatbot.embeddings.repository import (
    EmbeddingRepository,
    EmbeddingSpan,
    EmbeddingSpanStatus,
    VectorGenerationCoverage,
    VectorGenerationState,
)
from academic_chatbot.storage.paths import ProjectPaths
from tests.unit.embeddings.conftest import frozen_profile_payload as _frozen_profile_payload


@pytest.fixture
def frozen_profile_payload() -> dict[str, object]:
    return _frozen_profile_payload.__wrapped__()  # type: ignore[attr-defined]


def _runner() -> MigrationRunner:
    return MigrationRunner(
        Path(__file__).parents[3] / "src" / "academic_chatbot" / "db" / "migrations"
    )


def _repository(tmp_path: Path) -> tuple[EmbeddingRepository, ProjectPaths]:
    paths = ProjectPaths.create(tmp_path / "data", project_id="project-1")
    _runner().migrate_copy(paths.database_path, data_root=paths.data_root)
    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        connection.execute("INSERT INTO projects VALUES ('project-1', '2026-09-01T00:00:00Z')")
        connection.execute(
            "INSERT INTO papers VALUES ('paper-1', 'project-1', '2026-09-01T00:00:00Z')"
        )
        connection.execute(
            "INSERT INTO papers VALUES ('paper-2', 'project-1', '2026-09-01T00:00:00Z')"
        )
        for suffix, paper, generation, text, needs_ocr in (
            ("one", "paper-1", "generation-1", "alpha alpha beta", 0),
            ("two", "paper-2", "generation-2", "", 1),
        ):
            connection.execute(
                "INSERT INTO file_versions VALUES (?, ?, ?, ?, ?)",
                (
                    f"file-{suffix}",
                    paper,
                    (suffix * 64)[:64],
                    f"originals/{suffix}.pdf",
                    "2026-09-01T00:00:00Z",
                ),
            )
            connection.execute(
                "INSERT INTO document_generations VALUES (?, ?, ?, ?)",
                (generation, f"file-{suffix}", "native-v1", "2026-09-01T00:00:00Z"),
            )
            connection.execute(
                """INSERT INTO pages (page_id, document_generation_id, page_number, canonical_text,
                   canonical_text_sha256, needs_ocr) VALUES (?, ?, 1, ?, ?, ?)""",
                (f"page-{suffix}", generation, text, "a" * 64, needs_ocr),
            )
            connection.execute(
                "INSERT INTO generation_publications VALUES (?, ?)",
                (f"file-{suffix}", generation),
            )
        connection.execute(
            """INSERT INTO chunks VALUES ('chunk-one', 'generation-1', 'page-one', 0, 0, 16,
               'alpha alpha beta', 3, 'lexical-chunk-v1')"""
        )
    finally:
        connection.close()
    return EmbeddingRepository(paths), paths


def _span(
    profile: EmbeddingProfile,
    *,
    start: int = 0,
    end: int = 16,
    status: EmbeddingSpanStatus = EmbeddingSpanStatus.EMBEDDABLE,
) -> EmbeddingSpan:
    identity = EmbeddingSpanIdentity(
        document_generation_id="generation-1",
        chunk_id="chunk-one",
        page_id="page-one",
        start_offset=start,
        end_offset=end,
        embedding_profile_id=profile.embedding_profile_id,
    )
    return EmbeddingSpan(identity=identity, status=status)


def _coverage() -> VectorGenerationCoverage:
    return VectorGenerationCoverage(
        eligible_native_chunks=1,
        embeddable_spans=1,
        excluded_unembeddable_spans=0,
        needs_ocr_pages=1,
        indexed_documents=1,
        unindexed_documents=1,
    )


def test_profile_span_snapshot_and_finalized_publication(
    tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> None:
    repository, paths = _repository(tmp_path)
    profile = EmbeddingProfile.model_validate(frozen_profile_payload)
    repository.register_profile(profile, artifact_manifest_sha256="b" * 64)
    with pytest.raises(ValueError, match="another project"):
        repository.current_source_snapshot(
            project_id="project-2", embedding_profile_id=profile.embedding_profile_id
        )
    assert repository.get_profile(profile.embedding_profile_id) == profile

    span = _span(profile)
    repository.persist_span(span)
    snapshot = repository.current_source_snapshot(
        project_id="project-1", embedding_profile_id=profile.embedding_profile_id
    )
    assert snapshot.sources[1].eligible_native_chunk_count == 0
    assert snapshot.sources[1].needs_ocr_page_count == 1

    candidate = repository.create_candidate(
        project_id="project-1",
        embedding_profile_id=profile.embedding_profile_id,
        artifact_relative_dir="indexes/semantic/profile/generation",
        coverage=_coverage(),
    )
    assert candidate.state is VectorGenerationState.DB_CANDIDATE
    repository.attach_vector_row(
        vector_generation_id=candidate.vector_generation_id,
        vector_row=0,
        embedding_span_id=span.embedding_span_id,
    )
    assert repository.vector_row_mapping(candidate.vector_generation_id) == (
        (0, span.embedding_span_id),
    )
    finalized = repository.finalize_candidate(
        candidate.vector_generation_id, vector_store_manifest_sha256="c" * 64
    )
    repository.publish(finalized.vector_generation_id)
    assert (
        repository.active_generation(
            project_id="project-1", embedding_profile_id=profile.embedding_profile_id
        )
        == finalized
    )
    assert repository.is_generation_current(finalized.vector_generation_id)

    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        connection.execute(
            "INSERT INTO document_generations VALUES ('generation-1b', 'file-one', 'native-v2', '2026-09-01T00:00:00Z')"
        )
        connection.execute(
            """INSERT INTO pages (page_id, document_generation_id, page_number, canonical_text,
            canonical_text_sha256, needs_ocr) VALUES ('page-one-b', 'generation-1b', 1, 'new text', ?, 0)""",
            ("d" * 64,),
        )
        connection.execute(
            "UPDATE generation_publications SET document_generation_id = 'generation-1b' WHERE file_version_id = 'file-one'"
        )
    finally:
        connection.close()
    assert repository.active_generation(
        project_id="project-1", embedding_profile_id=profile.embedding_profile_id
    ) is None


def test_snapshot_staleness_exclusions_and_path_rejection(
    tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> None:
    repository, paths = _repository(tmp_path)
    profile = EmbeddingProfile.model_validate(frozen_profile_payload)
    repository.register_profile(profile, artifact_manifest_sha256="b" * 64)
    excluded = _span(profile, status=EmbeddingSpanStatus.EXCLUDED_UNEMBEDDABLE)
    repository.persist_span(excluded)
    candidate = repository.create_candidate(
        project_id="project-1",
        embedding_profile_id=profile.embedding_profile_id,
        artifact_relative_dir="indexes/semantic/profile/generation",
        coverage=VectorGenerationCoverage(1, 0, 1, 1, 0, 2),
    )
    with pytest.raises(ValueError, match="excluded"):
        repository.attach_vector_row(
            vector_generation_id=candidate.vector_generation_id,
            vector_row=0,
            embedding_span_id=excluded.embedding_span_id,
        )
    with pytest.raises(ValueError, match="relative"):
        repository.create_candidate(
            project_id="project-1",
            embedding_profile_id=profile.embedding_profile_id,
            artifact_relative_dir="../escape",
            coverage=_coverage(),
        )

    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        connection.execute(
            "INSERT INTO document_generations VALUES ('generation-1b', 'file-one', 'native-v2', '2026-09-01T00:00:00Z')"
        )
        connection.execute(
            "INSERT INTO pages (page_id, document_generation_id, page_number, canonical_text, canonical_text_sha256, needs_ocr) VALUES ('page-one-b', 'generation-1b', 1, 'new text', ?, 0)",
            ("d" * 64,),
        )
        connection.execute(
            "UPDATE generation_publications SET document_generation_id = 'generation-1b' WHERE file_version_id = 'file-one'"
        )
    finally:
        connection.close()
    assert not repository.is_generation_current(candidate.vector_generation_id)
    with pytest.raises(ValueError, match="stale"):
        repository.finalize_candidate(
            candidate.vector_generation_id, vector_store_manifest_sha256="c" * 64
        )


def test_database_rejects_bad_lineage_and_publication_state(
    tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> None:
    repository, paths = _repository(tmp_path)
    profile = EmbeddingProfile.model_validate(frozen_profile_payload)
    repository.register_profile(profile, artifact_manifest_sha256="b" * 64)
    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="embedding span"):
            connection.execute(
                "INSERT INTO embedding_spans VALUES ('bad', ?, 'generation-1', 'chunk-one', 'page-two', 0, 1, 'EMBEDDABLE')",
                (profile.embedding_profile_id,),
            )
    finally:
        connection.close()


def test_finalization_rejects_an_unmapped_embeddable_source_span(
    tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> None:
    repository, _ = _repository(tmp_path)
    profile = EmbeddingProfile.model_validate(frozen_profile_payload)
    repository.register_profile(profile, artifact_manifest_sha256="b" * 64)
    first = _span(profile, start=0, end=5)
    second = _span(profile, start=6, end=11)
    repository.persist_span(first)
    repository.persist_span(second)
    candidate = repository.create_candidate(
        project_id="project-1",
        embedding_profile_id=profile.embedding_profile_id,
        artifact_relative_dir="indexes/semantic/profile/generation",
        coverage=VectorGenerationCoverage(1, 2, 0, 1, 1, 1),
    )
    repository.attach_vector_row(
        vector_generation_id=candidate.vector_generation_id,
        vector_row=0,
        embedding_span_id=first.embedding_span_id,
    )
    with pytest.raises(ValueError, match="coverage"):
        repository.finalize_candidate(
            candidate.vector_generation_id, vector_store_manifest_sha256="c" * 64
        )
