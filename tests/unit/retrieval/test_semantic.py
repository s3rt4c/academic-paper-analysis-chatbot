from __future__ import annotations

from pathlib import Path

import pytest

from academic_chatbot.retrieval.semantic import (
    SemanticArtifactIntegrityError,
    _artifact_path,
    _load_registered_profile_with_manifest_hash,
    _require_ordinary_artifact_files,
)
from academic_chatbot.storage.paths import ProjectPaths
from tests.integration.embeddings.test_vector_publication import _profile, _project


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
