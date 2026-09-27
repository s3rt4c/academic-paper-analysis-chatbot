from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from academic_chatbot.domain.library import Project
from academic_chatbot.retrieval.semantic import (
    SemanticArtifactIntegrityError,
    SemanticQueryError,
    SemanticRetrievalService,
    _artifact_path,
    _load_registered_profile_with_manifest_hash,
    _require_ordinary_artifact_files,
)
from academic_chatbot.retrieval.semantic_position import PositionalAcquisitionSelection
from academic_chatbot.retrieval.semantic_query import CANONICAL_CONCERN_TEXT
from academic_chatbot.storage.paths import ProjectPaths
from tests.integration.embeddings.test_vector_publication import _profile, _project


@dataclass
class _RecordingQueryEmbedder:
    profile: object
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def embed_queries(self, texts: tuple[str, ...]) -> np.ndarray:
        self.calls.append(texts)
        vector = np.zeros((1, self.profile.dimension), dtype=np.float32)  # type: ignore[attr-defined]
        vector[0, 0] = 1.0
        return vector


def test_query_embedding_receives_one_exact_selected_representation() -> None:
    profile = _profile()
    embedder = _RecordingQueryEmbedder(profile)
    service = SemanticRetrievalService(
        data_root=Path("unused"), profile=profile, embedder=embedder
    )

    vector = service._embed_query(CANONICAL_CONCERN_TEXT)

    assert vector.shape == (profile.dimension,)
    assert embedder.calls == [(CANONICAL_CONCERN_TEXT,)]


def test_positional_acquisition_rejects_unknown_concern_before_embedding() -> None:
    profile = _profile()
    embedder = _RecordingQueryEmbedder(profile)
    service = SemanticRetrievalService(
        data_root=Path("unused"), profile=profile, embedder=embedder
    )

    with pytest.raises(SemanticQueryError):
        service.acquire_positional(
            Project(project_id="project-one", display_name="Research"),
            "user query",
            positional_selection=PositionalAcquisitionSelection(
                concern_id="unsupported-concern-v1"
            ),
        )

    assert embedder.calls == []


def test_semantic_artifact_path_rejects_nonsemantic_project_locations(tmp_path: Path) -> None:
    paths = ProjectPaths.create(tmp_path / "data", project_id="project-one")

    with pytest.raises(SemanticArtifactIntegrityError, match="semantic index root"):
        _artifact_path(
            paths,
            "derivatives/untrusted-vector",
            profile_id="ep-sha256-" + "a" * 64,
            source_snapshot_sha256="b" * 64,
        )


def test_registered_profile_loading_retains_its_manifest_binding(tmp_path: Path) -> None:
    repository, paths = _project(tmp_path, native_chunk=False)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)

    loaded, manifest_hash = _load_registered_profile_with_manifest_hash(
        paths, profile.embedding_profile_id
    )

    assert loaded == profile
    assert manifest_hash == "e" * 64


def test_semantic_artifact_rejects_a_symlinked_vector_file(tmp_path: Path) -> None:
    artifact = tmp_path / "generation"
    artifact.mkdir()
    target = tmp_path / "outside-vectors.npy"
    target.write_bytes(b"not-a-vector")
    try:
        (artifact / "vectors.npy").symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable on this Windows host: {error}")
    (artifact / "manifest.json").write_text("{}", encoding="utf-8")
    (artifact / "vectors.meta.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SemanticArtifactIntegrityError, match="symlink or reparse"):
        _require_ordinary_artifact_files(artifact, empty=False)
