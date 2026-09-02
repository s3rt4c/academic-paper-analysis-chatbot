"""Safe local model-root paths and exact external-artifact verification."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from academic_chatbot.embeddings.models import (
    EmbeddingArtifactManifest,
    EmbeddingContractError,
    EmbeddingProfile,
    _validate_relative_posix_path,
    canonical_json_bytes,
)
from academic_chatbot.storage.paths import ensure_path_beneath

_MANIFEST_FILENAME = "manifest.json"
_HASH_BLOCK_BYTES = 1024 * 1024


class EmbeddingArtifactError(EmbeddingContractError):
    """Stable expected failure at the local embedding-artifact boundary."""


@dataclass(frozen=True, slots=True)
class EmbeddingArtifactPaths:
    """Application-managed, profile-scoped external model locations."""

    model_root: Path
    profile: EmbeddingProfile

    @classmethod
    def create(cls, model_root: Path, *, profile: EmbeddingProfile) -> EmbeddingArtifactPaths:
        return cls(model_root=Path(model_root).absolute(), profile=profile)

    @property
    def embeddings_directory(self) -> Path:
        return self.model_root / "embeddings"

    @property
    def profile_directory(self) -> Path:
        return self.embeddings_directory / self.profile.embedding_profile_id

    def artifact_path(self, relative_path: str) -> Path:
        safe_path = _validate_relative_posix_path(relative_path)
        return ensure_path_beneath(
            root=self.profile_directory,
            candidate=self.profile_directory / Path(*safe_path.split("/")),
        )


@dataclass(frozen=True, slots=True)
class VerifiedEmbeddingArtifacts:
    """A fully verified immutable manifest and its ordinary artifact files."""

    manifest: EmbeddingArtifactManifest
    _artifact_paths: Mapping[str, Path]

    def artifact_path(self, filename: str) -> Path:
        try:
            return self._artifact_paths[filename]
        except KeyError as error:
            raise EmbeddingArtifactError(
                "artifact is not present in the verified manifest"
            ) from error


def load_verified_artifacts(
    model_root: Path,
    *,
    profile: EmbeddingProfile,
) -> VerifiedEmbeddingArtifacts:
    """Load a canonical on-disk manifest and verify its exact artifact inventory."""

    paths = EmbeddingArtifactPaths.create(model_root, profile=profile)
    _require_ordinary_directory(paths.model_root, "model root")
    _require_ordinary_directory(paths.embeddings_directory, "embeddings directory")
    _require_ordinary_directory(paths.profile_directory, "profile directory")
    manifest_path = paths.profile_directory / _MANIFEST_FILENAME
    _require_ordinary_file(manifest_path, _MANIFEST_FILENAME)
    try:
        raw_manifest = manifest_path.read_bytes()
        decoded_manifest = json.loads(raw_manifest)
        manifest = EmbeddingArtifactManifest.model_validate(decoded_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise EmbeddingArtifactError("embedding artifact manifest is invalid") from error
    if raw_manifest != canonical_json_bytes(manifest.canonical_payload()):
        raise EmbeddingArtifactError("embedding artifact manifest is not canonical UTF-8 JSON")
    artifact_paths = verify_artifact_inventory(paths, manifest)
    return VerifiedEmbeddingArtifacts(manifest, MappingProxyType(artifact_paths))


def verify_artifact_inventory(
    paths: EmbeddingArtifactPaths,
    manifest: EmbeddingArtifactManifest,
) -> dict[str, Path]:
    """Verify the exact artifact inventory without opening an ONNX graph.

    `manifest.json` is the only permitted non-artifact file because later
    import/open code stores the verified manifest alongside the artifacts.
    """

    if manifest.embedding_profile != paths.profile:
        raise EmbeddingArtifactError("artifact manifest does not belong to the requested profile")
    _require_ordinary_directory(paths.model_root, "model root")
    _require_ordinary_directory(paths.embeddings_directory, "embeddings directory")
    _require_ordinary_directory(paths.profile_directory, "profile directory")

    expected = {artifact.filename: artifact for artifact in manifest.artifacts}
    actual = _listed_profile_files(paths.profile_directory)
    unexpected = set(actual).difference(expected, {_MANIFEST_FILENAME})
    missing = set(expected).difference(actual)
    if missing:
        raise EmbeddingArtifactError("artifact inventory is missing expected files")
    if unexpected:
        raise EmbeddingArtifactError("artifact inventory contains unexpected files")

    verified: dict[str, Path] = {}
    for filename, artifact in expected.items():
        candidate = paths.artifact_path(filename)
        _require_ordinary_file(candidate, filename)
        if candidate.stat().st_size != artifact.byte_size:
            raise EmbeddingArtifactError(f"{filename} size does not match the manifest")
        digest = _file_sha256(candidate)
        if not hmac.compare_digest(digest, artifact.sha256):
            raise EmbeddingArtifactError(f"{filename} SHA-256 does not match the manifest")
        verified[filename] = candidate
    return verified


def _listed_profile_files(profile_directory: Path) -> set[str]:
    result: set[str] = set()
    for candidate in profile_directory.rglob("*"):
        metadata = candidate.lstat()
        if candidate.is_symlink() or _is_reparse_point(metadata):
            raise EmbeddingArtifactError(
                "artifact inventory must not contain symlink or reparse entries"
            )
        if candidate.is_dir():
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise EmbeddingArtifactError("artifact inventory must contain only regular files")
        try:
            relative = candidate.relative_to(profile_directory).as_posix()
        except ValueError as error:
            raise EmbeddingArtifactError(
                "artifact inventory escaped the profile directory"
            ) from error
        result.add(_validate_relative_posix_path(relative))
    return result


def _require_ordinary_directory(path: Path, description: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise EmbeddingArtifactError(f"{description} does not exist") from error
    if path.is_symlink() or _is_reparse_point(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise EmbeddingArtifactError(f"{description} must be an ordinary non-reparse directory")


def _require_ordinary_file(path: Path, description: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise EmbeddingArtifactError(f"{description} is missing") from error
    if path.is_symlink() or _is_reparse_point(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise EmbeddingArtifactError(f"{description} must be an ordinary non-reparse file")
    if metadata.st_nlink != 1:
        raise EmbeddingArtifactError(f"{description} must not be hard-linked")


def _is_reparse_point(metadata: os.stat_result) -> bool:
    attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attribute and metadata.st_file_attributes & attribute)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()
