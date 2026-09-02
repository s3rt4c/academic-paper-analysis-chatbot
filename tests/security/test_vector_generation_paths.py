from __future__ import annotations

from pathlib import Path

import pytest

from academic_chatbot.embeddings.reconciliation import reconcile_vector_generations
from academic_chatbot.embeddings.vector_build import VectorBuildError
from academic_chatbot.storage.paths import PathEscapeError
from tests.integration.embeddings.test_vector_publication import (
    _builder,
    _Embedder,
    _profile,
    _project,
)


def test_project_vector_artifacts_remain_contained_and_orphans_stay_inert(tmp_path: Path) -> None:
    repository, paths = _project(tmp_path, native_chunk=True)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
    result = _builder(repository, paths, _Embedder(profile)).build(project_id="project-one")
    artifact = paths.resolve_relative(result.generation.artifact_relative_dir)
    assert artifact.is_relative_to(paths.project_root)

    orphan = paths.resolve_relative(
        f"indexes/semantic/{profile.embedding_profile_id}/untrusted-orphan"
    )
    orphan.mkdir(parents=True)
    marker = orphan / "do-not-activate.txt"
    marker.write_text("inert", encoding="utf-8")
    reconciliation = reconcile_vector_generations(
        paths=paths, repository=repository, profile=profile, project_id="project-one"
    )
    assert reconciliation.active_generation_id == result.generation.vector_generation_id
    assert marker.read_text(encoding="utf-8") == "inert"


def test_build_rejects_a_reparse_workspace_component(tmp_path: Path) -> None:
    repository, paths = _project(tmp_path, native_chunk=True)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
    outside = tmp_path / "outside-project-root"
    outside.mkdir()
    try:
        (paths.project_root / "indexes").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable on this Windows host: {error}")

    with pytest.raises(VectorBuildError, match=r"reparse|symbolic link"):
        _builder(repository, paths, _Embedder(profile)).build(project_id="project-one")


@pytest.mark.parametrize("relative", ("../escape", "C:/escape", "//server/share"))
def test_project_vector_paths_reject_windows_escape_forms(tmp_path: Path, relative: str) -> None:
    _, paths = _project(tmp_path, native_chunk=False)

    with pytest.raises(PathEscapeError):
        paths.resolve_relative(relative)

    with pytest.raises(VectorBuildError, match="another project"):
        _builder(_, paths, _Embedder(_profile())).build(project_id="other-project")
