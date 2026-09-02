from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from academic_chatbot.embeddings.artifacts import (
    EmbeddingArtifactPaths,
    load_verified_artifacts,
    verify_artifact_inventory,
)
from academic_chatbot.embeddings.models import (
    ArtifactFile,
    EmbeddingArtifactManifest,
    EmbeddingProfile,
    artifact_set_sha256_for,
    canonical_json_bytes,
)


def _manifest(
    profile: EmbeddingProfile, *, artifacts: tuple[ArtifactFile, ...]
) -> EmbeddingArtifactManifest:
    return EmbeddingArtifactManifest(
        manifest_schema_version="embedding-artifact-manifest-v1",
        embedding_profile=profile,
        embedding_profile_id=profile.embedding_profile_id,
        declared_license="MIT",
        artifacts=artifacts,
    )


def _profile_with_inventory(
    frozen_profile_payload: dict[str, object], artifacts: tuple[ArtifactFile, ...]
) -> EmbeddingProfile:
    payload = frozen_profile_payload.copy()
    payload["artifacts"] = [artifact.model_dump(mode="json") for artifact in artifacts]
    payload["artifact_set_sha256"] = artifact_set_sha256_for(artifacts)
    return EmbeddingProfile.model_validate(payload)


@pytest.mark.parametrize(
    "relative_path",
    [
        "C:/model.onnx",
        "C:\\model.onnx",
        "//server/share/model.onnx",
        "../model.onnx",
        "onnx/../model.onnx",
        "onnx//model.onnx",
        "onnx\\model.onnx",
    ],
)
def test_artifact_file_rejects_unsafe_path_syntax(relative_path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactFile(filename=relative_path, byte_size=1, sha256="a" * 64)


def test_manifest_rejects_duplicate_or_profile_mismatched_artifacts(
    frozen_profile_payload: dict[str, object],
) -> None:
    profile = EmbeddingProfile.model_validate(frozen_profile_payload)
    artifact = ArtifactFile(filename="config.json", byte_size=1, sha256="a" * 64)

    with pytest.raises(ValidationError, match="unique"):
        _manifest(profile, artifacts=(artifact, artifact))
    with pytest.raises(ValidationError, match="profile"):
        EmbeddingArtifactManifest(
            manifest_schema_version="embedding-artifact-manifest-v1",
            embedding_profile=profile,
            embedding_profile_id="ep-sha256-" + "a" * 64,
            declared_license="MIT",
            artifacts=(artifact,),
        )


def test_equivalent_logical_manifests_have_identical_canonical_payloads(
    frozen_profile_payload: dict[str, object],
) -> None:
    profile = EmbeddingProfile.model_validate(frozen_profile_payload)
    first = _manifest(profile, artifacts=profile.artifacts)
    second = EmbeddingArtifactManifest.model_validate(
        {
            "manifest_schema_version": "embedding-artifact-manifest-v1",
            "embedding_profile": frozen_profile_payload,
            "embedding_profile_id": profile.embedding_profile_id,
            "declared_license": "MIT",
            "artifacts": [artifact.model_dump(mode="json") for artifact in profile.artifacts],
        }
    )

    assert first.canonical_payload() == second.canonical_payload()


def test_model_root_keeps_artifact_paths_beneath_profile_directory(
    tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> None:
    profile = EmbeddingProfile.model_validate(frozen_profile_payload)
    paths = EmbeddingArtifactPaths.create(tmp_path, profile=profile)

    assert paths.profile_directory == tmp_path / "embeddings" / profile.embedding_profile_id
    assert paths.artifact_path("onnx/model.onnx").is_relative_to(paths.profile_directory)


def test_verify_artifact_inventory_accepts_exact_synthetic_files(
    tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> None:
    content = b"tiny tokenizer"
    artifact = ArtifactFile(
        filename="tokenizer.json",
        byte_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    profile = _profile_with_inventory(frozen_profile_payload, (artifact,))
    paths = EmbeddingArtifactPaths.create(tmp_path, profile=profile)
    artifact_path = paths.profile_directory / "tokenizer.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(content)
    manifest = _manifest(profile, artifacts=(artifact,))

    verified = verify_artifact_inventory(paths, manifest)

    assert verified == {"tokenizer.json": artifact_path}


@pytest.mark.parametrize("tamper", [b"different", b""])
def test_verify_artifact_inventory_rejects_size_or_hash_mismatch(
    tmp_path: Path, frozen_profile_payload: dict[str, object], tamper: bytes
) -> None:
    content = b"original"
    artifact = ArtifactFile(
        filename="tokenizer.json",
        byte_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    profile = _profile_with_inventory(frozen_profile_payload, (artifact,))
    paths = EmbeddingArtifactPaths.create(tmp_path, profile=profile)
    artifact_path = paths.profile_directory / "tokenizer.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(content)
    manifest = _manifest(profile, artifacts=(artifact,))
    artifact_path.write_bytes(tamper)

    with pytest.raises(ValueError, match=r"size|SHA-256"):
        verify_artifact_inventory(paths, manifest)


def test_verify_artifact_inventory_rejects_extra_or_missing_files(
    tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> None:
    artifact = ArtifactFile(filename="tokenizer.json", byte_size=1, sha256="a" * 64)
    profile = _profile_with_inventory(frozen_profile_payload, (artifact,))
    paths = EmbeddingArtifactPaths.create(tmp_path, profile=profile)
    paths.profile_directory.mkdir(parents=True)
    (paths.profile_directory / "extra.txt").write_text("unexpected", encoding="utf-8")
    manifest = _manifest(profile, artifacts=(artifact,))

    with pytest.raises(ValueError, match=r"missing|unexpected"):
        verify_artifact_inventory(paths, manifest)


def test_verify_artifact_inventory_rejects_symlink_escape_when_supported(
    tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> None:
    content = b"outside artifact"
    artifact = ArtifactFile(
        filename="tokenizer.json",
        byte_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    profile = _profile_with_inventory(frozen_profile_payload, (artifact,))
    paths = EmbeddingArtifactPaths.create(tmp_path / "model-root", profile=profile)
    paths.profile_directory.mkdir(parents=True)
    outside = tmp_path / "outside-tokenizer.json"
    outside.write_bytes(content)
    try:
        os.symlink(outside, paths.profile_directory / artifact.filename)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable on this Windows configuration: {error}")

    manifest = _manifest(profile, artifacts=(artifact,))

    with pytest.raises(ValueError, match=r"symlink|reparse"):
        verify_artifact_inventory(paths, manifest)


def test_load_verified_artifacts_requires_a_canonical_on_disk_manifest(
    tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> None:
    content = b"tiny tokenizer"
    artifact = ArtifactFile(
        filename="tokenizer.json",
        byte_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    profile = _profile_with_inventory(frozen_profile_payload, (artifact,))
    paths = EmbeddingArtifactPaths.create(tmp_path, profile=profile)
    paths.profile_directory.mkdir(parents=True)
    (paths.profile_directory / artifact.filename).write_bytes(content)
    manifest = _manifest(profile, artifacts=(artifact,))
    (paths.profile_directory / "manifest.json").write_bytes(
        canonical_json_bytes(manifest.canonical_payload())
    )

    verified = load_verified_artifacts(tmp_path, profile=profile)

    assert verified.manifest == manifest
    assert verified.artifact_path("tokenizer.json") == paths.profile_directory / "tokenizer.json"


def test_load_verified_artifacts_rejects_noncanonical_manifest_bytes(
    tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> None:
    content = b"tiny tokenizer"
    artifact = ArtifactFile(
        filename="tokenizer.json",
        byte_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    profile = _profile_with_inventory(frozen_profile_payload, (artifact,))
    paths = EmbeddingArtifactPaths.create(tmp_path, profile=profile)
    paths.profile_directory.mkdir(parents=True)
    (paths.profile_directory / artifact.filename).write_bytes(content)
    manifest = _manifest(profile, artifacts=(artifact,))
    (paths.profile_directory / "manifest.json").write_bytes(
        canonical_json_bytes(manifest.canonical_payload()) + b"\n"
    )

    with pytest.raises(ValueError, match="canonical"):
        load_verified_artifacts(tmp_path, profile=profile)


def test_load_verified_artifacts_rejects_hard_linked_artifact_when_supported(
    tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> None:
    content = b"hard-linked tokenizer"
    artifact = ArtifactFile(
        filename="tokenizer.json",
        byte_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    profile = _profile_with_inventory(frozen_profile_payload, (artifact,))
    paths = EmbeddingArtifactPaths.create(tmp_path / "model-root", profile=profile)
    paths.profile_directory.mkdir(parents=True)
    source = tmp_path / "source-tokenizer.json"
    source.write_bytes(content)
    try:
        os.link(source, paths.profile_directory / artifact.filename)
    except OSError as error:
        pytest.skip(f"hard-link creation is unavailable on this Windows configuration: {error}")
    manifest = _manifest(profile, artifacts=(artifact,))
    (paths.profile_directory / "manifest.json").write_bytes(
        canonical_json_bytes(manifest.canonical_payload())
    )

    with pytest.raises(ValueError, match="hard-linked"):
        load_verified_artifacts(tmp_path / "model-root", profile=profile)


def test_load_verified_artifacts_rejects_a_symlinked_model_root_when_supported(
    tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> None:
    content = b"tiny tokenizer"
    artifact = ArtifactFile(
        filename="tokenizer.json",
        byte_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    profile = _profile_with_inventory(frozen_profile_payload, (artifact,))
    backing_root = tmp_path / "backing-root"
    paths = EmbeddingArtifactPaths.create(backing_root, profile=profile)
    paths.profile_directory.mkdir(parents=True)
    (paths.profile_directory / artifact.filename).write_bytes(content)
    manifest = _manifest(profile, artifacts=(artifact,))
    (paths.profile_directory / "manifest.json").write_bytes(
        canonical_json_bytes(manifest.canonical_payload())
    )
    symlink_root = tmp_path / "symlink-root"
    try:
        os.symlink(backing_root, symlink_root, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable on this Windows configuration: {error}")

    with pytest.raises(ValueError, match=r"model root.*non-reparse"):
        load_verified_artifacts(symlink_root, profile=profile)
