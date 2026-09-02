from __future__ import annotations

# ruff: noqa: E501
import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from academic_chatbot.db.connection import connect_project_database
from academic_chatbot.db.migrations import MigrationRunner
from academic_chatbot.embeddings.models import EmbeddingProfile, EmbeddingRole
from academic_chatbot.embeddings.reconciliation import (
    VectorReconciliationError,
    reconcile_vector_generations,
)
from academic_chatbot.embeddings.repository import EmbeddingRepository, VectorGenerationState
from academic_chatbot.embeddings.vector_build import ProjectVectorBuilder, VectorBuildError
from academic_chatbot.retrieval.exact_memmap import ExactVectorStore
from academic_chatbot.storage.paths import ProjectPaths
from tests.unit.embeddings.conftest import frozen_profile_payload as _frozen_profile_payload


@dataclass
class _Tokenizer:
    profile: EmbeddingProfile
    maximum_words: int = 510

    def prepare(self, role: EmbeddingRole, texts: tuple[str, ...]) -> object:
        assert role is EmbeddingRole.DOCUMENT
        if any(len(text.split(" ")) > self.maximum_words for text in texts):
            from academic_chatbot.embeddings.tokenizer import EmbeddingInputTooLongError

            raise EmbeddingInputTooLongError("document exceeds embedding profile token budget")
        return object()


@dataclass
class _Embedder:
    profile: EmbeddingProfile
    bad_dtype: bool = False
    after_call: Callable[[], None] | None = None
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        self.calls.append(texts)
        rows = np.empty((len(texts), self.profile.dimension), dtype=np.float32)
        for index, text in enumerate(texts):
            seed = hashlib.sha256(text.encode("utf-8")).digest()
            values = np.frombuffer((seed * 12)[: self.profile.dimension], dtype=np.uint8)
            rows[index] = values.astype(np.float32) + np.float32(1.0)
        rows /= np.sqrt(np.sum(rows * rows, axis=1, keepdims=True, dtype=np.float32))
        if self.bad_dtype:
            return rows.astype(np.float64)
        if self.after_call is not None:
            self.after_call()
        return rows


def _runner() -> MigrationRunner:
    return MigrationRunner(
        Path(__file__).parents[3] / "src" / "academic_chatbot" / "db" / "migrations"
    )


def _project(tmp_path: Path, *, native_chunk: bool) -> tuple[EmbeddingRepository, ProjectPaths]:
    paths = ProjectPaths.create(tmp_path / "data", project_id="project-one")
    _runner().migrate_copy(paths.database_path, data_root=paths.data_root)
    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        connection.execute("INSERT INTO projects VALUES ('project-one', '2026-09-01T00:00:00Z')")
        connection.execute(
            "INSERT INTO papers VALUES ('paper-one', 'project-one', '2026-09-01T00:00:00Z')"
        )
        connection.execute(
            "INSERT INTO file_versions VALUES ('file-one', 'paper-one', ?, 'originals/a.pdf', '2026-09-01T00:00:00Z')",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO document_generations VALUES ('generation-one', 'file-one', 'native-v1', '2026-09-01T00:00:00Z')"
        )
        text = "alpha beta gamma" if native_chunk else ""
        connection.execute(
            "INSERT INTO pages (page_id, document_generation_id, page_number, physical_page_index, canonical_text, canonical_text_sha256, needs_ocr) VALUES ('page-one', 'generation-one', 1, 0, ?, ?, ?)",
            (text, "b" * 64, int(not native_chunk)),
        )
        connection.execute(
            "INSERT INTO generation_publications VALUES ('file-one', 'generation-one')"
        )
        if native_chunk:
            connection.execute(
                "INSERT INTO chunks VALUES ('chunk-one', 'generation-one', 'page-one', 0, 0, 16, 'alpha beta gamma', 3, 'lexical-chunk-v1')"
            )
            cursor = 0
            for ordinal, word in enumerate(("alpha", "beta", "gamma")):
                connection.execute(
                    "INSERT INTO page_anchors VALUES (?, ?, 'page-one', ?, ?, ?, ?, ?, 0.0, 0.0, 1.0, 1.0)",
                    (
                        f"anchor-{ordinal}",
                        f"evidence-{ordinal}",
                        cursor,
                        cursor + len(word),
                        word,
                        "c" * 64,
                        "d" * 64,
                    ),
                )
                cursor += len(word) + 1
    finally:
        connection.close()
    return EmbeddingRepository(paths), paths


def _profile() -> EmbeddingProfile:
    return EmbeddingProfile.model_validate(_frozen_profile_payload.__wrapped__())  # type: ignore[attr-defined]


def _builder(
    repository: EmbeddingRepository,
    paths: ProjectPaths,
    embedder: _Embedder,
    *,
    tokenizer_maximum_words: int = 510,
) -> ProjectVectorBuilder:
    return ProjectVectorBuilder(
        paths=paths,
        repository=repository,
        profile=embedder.profile,
        tokenizer=_Tokenizer(embedder.profile, maximum_words=tokenizer_maximum_words),
        embedder=embedder,
    )


def test_builds_an_immutable_project_local_store_with_exact_row_mapping(tmp_path: Path) -> None:
    repository, paths = _project(tmp_path, native_chunk=True)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
    embedder = _Embedder(profile)

    result = _builder(repository, paths, embedder).build(project_id="project-one")

    assert result.generation.state is VectorGenerationState.FILES_FINALIZED
    assert not result.reused and not result.empty
    assert result.generation.coverage.embeddable_spans == 1
    assert (
        repository.active_generation(
            project_id="project-one", embedding_profile_id=profile.embedding_profile_id
        )
        == result.generation
    )
    with ExactVectorStore.open(
        paths.resolve_relative(result.generation.artifact_relative_dir)
    ) as store:
        assert store.manifest.manifest_sha256 == result.generation.vector_store_manifest_sha256
        assert tuple(enumerate(store.row_ids)) == repository.vector_row_mapping(
            result.generation.vector_generation_id
        )
    calls_before_reuse = tuple(embedder.calls)
    reused = _builder(repository, paths, embedder).build(project_id="project-one")
    assert reused.reused and reused.generation == result.generation
    assert tuple(embedder.calls) == calls_before_reuse


def test_represents_an_empty_native_coverage_generation_without_a_dummy_vector(
    tmp_path: Path,
) -> None:
    repository, paths = _project(tmp_path, native_chunk=False)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
    embedder = _Embedder(profile)

    result = _builder(repository, paths, embedder).build(project_id="project-one")

    assert result.empty
    assert result.generation.coverage.embeddable_spans == 0
    assert result.generation.coverage.needs_ocr_pages == 1
    artifact_dir = paths.resolve_relative(result.generation.artifact_relative_dir)
    assert (artifact_dir / "empty-generation.json").is_file()
    assert not (artifact_dir / "vectors.npy").exists()
    assert embedder.calls == []


def test_keeps_an_excluded_oversized_word_as_durable_coverage_without_a_vector(
    tmp_path: Path,
) -> None:
    repository, paths = _project(tmp_path, native_chunk=True)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
    embedder = _Embedder(profile)

    result = _builder(repository, paths, embedder, tokenizer_maximum_words=0).build(
        project_id="project-one"
    )

    assert result.empty
    assert result.generation.coverage.excluded_unembeddable_spans == 3
    assert result.generation.coverage.embeddable_spans == 0
    assert embedder.calls == []


def test_rejects_bad_embedder_output_without_creating_an_active_generation(tmp_path: Path) -> None:
    repository, paths = _project(tmp_path, native_chunk=True)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)

    with pytest.raises(VectorBuildError, match="float32"):
        _builder(repository, paths, _Embedder(profile, bad_dtype=True)).build(
            project_id="project-one"
        )

    assert (
        repository.active_generation(
            project_id="project-one", embedding_profile_id=profile.embedding_profile_id
        )
        is None
    )


def test_rejects_a_project_identifier_outside_the_builder_paths(tmp_path: Path) -> None:
    repository, paths = _project(tmp_path, native_chunk=True)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)

    with pytest.raises(VectorBuildError, match="another project"):
        _builder(repository, paths, _Embedder(profile)).build(project_id="project-two")


def test_rejects_a_source_snapshot_that_changes_after_embedding(tmp_path: Path) -> None:
    repository, paths = _project(tmp_path, native_chunk=True)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)

    def switch_active_generation() -> None:
        connection = connect_project_database(paths.database_path, data_root=paths.data_root)
        try:
            connection.execute(
                "INSERT INTO document_generations VALUES ('generation-two', 'file-one', 'native-v2', '2026-09-01T00:00:01Z')"
            )
            connection.execute(
                "INSERT INTO pages (page_id, document_generation_id, page_number, physical_page_index, canonical_text, canonical_text_sha256, needs_ocr) VALUES ('page-two', 'generation-two', 1, 0, 'delta', ?, 0)",
                ("f" * 64,),
            )
            connection.execute(
                "INSERT INTO chunks VALUES ('chunk-two', 'generation-two', 'page-two', 0, 0, 5, 'delta', 1, 'lexical-chunk-v1')"
            )
            connection.execute(
                "INSERT INTO page_anchors VALUES ('anchor-two', 'evidence-two', 'page-two', 0, 5, 'delta', ?, ?, 0.0, 0.0, 1.0, 1.0)",
                ("c" * 64, "d" * 64),
            )
            connection.execute(
                "UPDATE generation_publications SET document_generation_id = 'generation-two' WHERE file_version_id = 'file-one'"
            )
        finally:
            connection.close()

    with pytest.raises(VectorBuildError, match="source changed"):
        _builder(repository, paths, _Embedder(profile, after_call=switch_active_generation)).build(
            project_id="project-one"
        )

    assert (
        repository.active_generation(
            project_id="project-one", embedding_profile_id=profile.embedding_profile_id
        )
        is None
    )


def test_reconciliation_verifies_active_artifacts_and_never_activates_an_orphan(
    tmp_path: Path,
) -> None:
    repository, paths = _project(tmp_path, native_chunk=True)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
    result = _builder(repository, paths, _Embedder(profile)).build(project_id="project-one")

    healthy = reconcile_vector_generations(
        paths=paths, repository=repository, profile=profile, project_id="project-one"
    )
    assert healthy.active_generation_id == result.generation.vector_generation_id

    vectors_path = paths.resolve_relative(result.generation.artifact_relative_dir) / "vectors.npy"
    vectors_path.write_bytes(b"corrupt")
    with pytest.raises(VectorReconciliationError, match="active"):
        reconcile_vector_generations(
            paths=paths, repository=repository, profile=profile, project_id="project-one"
        )
    assert (
        repository.active_generation(
            project_id="project-one", embedding_profile_id=profile.embedding_profile_id
        )
        == result.generation
    )


def test_repeat_build_fails_closed_when_the_authoritative_artifact_is_corrupt(
    tmp_path: Path,
) -> None:
    repository, paths = _project(tmp_path, native_chunk=True)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
    result = _builder(repository, paths, _Embedder(profile)).build(project_id="project-one")
    (paths.resolve_relative(result.generation.artifact_relative_dir) / "vectors.npy").write_bytes(
        b"corrupt"
    )

    with pytest.raises(VectorBuildError, match="active vector artifact"):
        _builder(repository, paths, _Embedder(profile)).build(project_id="project-one")


def test_reconciliation_recovers_a_verified_finalized_candidate_after_pointer_crash(
    tmp_path: Path,
) -> None:
    repository, paths = _project(tmp_path, native_chunk=True)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
    result = _builder(repository, paths, _Embedder(profile)).build(project_id="project-one")
    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        connection.execute("DELETE FROM vector_generation_publications")
    finally:
        connection.close()

    recovered = reconcile_vector_generations(
        paths=paths, repository=repository, profile=profile, project_id="project-one"
    )

    assert recovered.recovered_generation_ids == (result.generation.vector_generation_id,)
    assert repository.active_generation(
        project_id="project-one", embedding_profile_id=profile.embedding_profile_id
    ) == result.generation


def test_stale_retirement_rechecks_the_snapshot_inside_its_transaction(tmp_path: Path) -> None:
    repository, paths = _project(tmp_path, native_chunk=True)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
    result = _builder(repository, paths, _Embedder(profile)).build(project_id="project-one")
    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        connection.execute("DELETE FROM vector_generation_publications")
    finally:
        connection.close()
    with pytest.raises(ValueError, match="current"):
        repository.mark_stale(result.generation.vector_generation_id)

    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        connection.execute(
            "INSERT INTO document_generations VALUES ('generation-two', 'file-one', 'native-v2', '2026-09-01T00:00:01Z')"
        )
        connection.execute(
            "INSERT INTO pages (page_id, document_generation_id, page_number, canonical_text, canonical_text_sha256, needs_ocr) VALUES ('page-two', 'generation-two', 1, '', ?, 1)",
            ("f" * 64,),
        )
        connection.execute(
            "UPDATE generation_publications SET document_generation_id = 'generation-two' WHERE file_version_id = 'file-one'"
        )
    finally:
        connection.close()
    repository.mark_stale(result.generation.vector_generation_id)
    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        state = connection.execute(
            "SELECT state FROM vector_generations WHERE vector_generation_id = ?",
            (result.generation.vector_generation_id,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert state == "STALE"
