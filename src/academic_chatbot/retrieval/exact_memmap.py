from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from mmap import mmap
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SHA256_RE = re.compile(_SHA256_PATTERN)
_VECTOR_FILENAME = "vectors.npy"
_METADATA_FILENAME = "vectors.meta.json"
_MANIFEST_FILENAME = "manifest.json"
_NPY_VERSION = (2, 0)
_VERIFY_BLOCK_ROWS = 65_536


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_file_bytes(payload: Mapping[str, object]) -> bytes:
    return _canonical_json_bytes(payload) + b"\n"


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalization_identity(*, normalization_atol: float) -> dict[str, object]:
    return {
        "normalized_source": True,
        "normalization_atol": normalization_atol,
    }


def _generation_identity_payload(
    *,
    vectors_sha256: str,
    metadata_sha256: str,
    profile_sha256: str,
    row_count: int,
    dimension: int,
    normalization_atol: float,
) -> dict[str, object]:
    return {
        "dimension": dimension,
        "dtype": "<f2",
        "metadata_sha256": metadata_sha256,
        "normalization_policy": _normalization_identity(
            normalization_atol=normalization_atol
        ),
        "npy_version": "2.0",
        "order": "C",
        "profile_sha256": profile_sha256,
        "row_count": row_count,
        "vectors_sha256": vectors_sha256,
    }


class VectorGenerationManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    generation_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    format: Literal["npy"] = "npy"
    npy_version: Literal["2.0"] = "2.0"
    dtype: Literal["<f2"] = "<f2"
    order: Literal["C"] = "C"
    row_count: int = Field(gt=0)
    dimension: int = Field(gt=0)
    normalized_source: Literal[True] = True
    normalization_atol: float = Field(gt=0.0)
    row_id_kind: Literal["embedding_span_id"] = "embedding_span_id"
    vectors_filename: Literal["vectors.npy"] = "vectors.npy"
    metadata_filename: Literal["vectors.meta.json"] = "vectors.meta.json"
    vectors_file_bytes: int = Field(gt=0)
    vector_payload_bytes: int = Field(gt=0)
    metadata_file_bytes: int = Field(gt=0)
    vectors_sha256: str = Field(pattern=_SHA256_PATTERN)
    metadata_sha256: str = Field(pattern=_SHA256_PATTERN)
    profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_content_hashes(self) -> Self:
        if not math.isfinite(self.normalization_atol):
            raise ValueError("normalization_atol must be finite")
        expected_payload_bytes = self.row_count * self.dimension * 2
        if self.vector_payload_bytes != expected_payload_bytes:
            raise ValueError(
                "vector_payload_bytes does not match row_count, dimension, and dtype"
            )
        identity_payload = _generation_identity_payload(
            vectors_sha256=self.vectors_sha256,
            metadata_sha256=self.metadata_sha256,
            profile_sha256=self.profile_sha256,
            row_count=self.row_count,
            dimension=self.dimension,
            normalization_atol=self.normalization_atol,
        )
        expected_generation_id = f"sha256-{_canonical_sha256(identity_payload)}"
        if not hmac.compare_digest(self.generation_id, expected_generation_id):
            raise ValueError(
                "generation_id does not match the canonical generation identity"
            )
        manifest_payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        expected_manifest_hash = _canonical_sha256(manifest_payload)
        if not hmac.compare_digest(self.manifest_sha256, expected_manifest_hash):
            raise ValueError(
                "manifest_sha256 does not match the canonical manifest payload"
            )
        return self


class _VectorMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    row_count: int = Field(gt=0)
    row_id_kind: Literal["embedding_span_id"] = "embedding_span_id"
    row_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_row_ids(self) -> Self:
        if len(self.row_ids) != self.row_count:
            raise ValueError("metadata row_ids count does not match row_count")
        if any(not row_id for row_id in self.row_ids):
            raise ValueError("metadata row_ids must not contain empty identifiers")
        if len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError("metadata row_ids must be unique")
        return self


@dataclass(frozen=True, slots=True)
class VectorHit:
    row_id: str
    score: float
    vector_row: int


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        loaded: Any = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name} is not valid UTF-8 JSON") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return cast(dict[str, object], loaded)


def _write_canonical_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("wb") as handle:
        handle.write(_canonical_json_file_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _close_mapping(mapping: np.memmap[Any, Any]) -> None:
    backing_map = cast(mmap | None, getattr(mapping, "_mmap", None))
    if backing_map is not None:
        backing_map.close()


def _read_npy_header(path: Path) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version != _NPY_VERSION:
            raise ValueError("vectors.npy must use NPY version 2.0")
        return np.lib.format.read_array_header_2_0(handle)


def _mapping_is_finite(mapping: np.memmap[Any, Any]) -> bool:
    for start in range(0, mapping.shape[0], _VERIFY_BLOCK_ROWS):
        if not bool(np.isfinite(mapping[start : start + _VERIFY_BLOCK_ROWS]).all()):
            return False
    return True


def _validate_build_inputs(
    *,
    rows: NDArray[np.float32],
    row_ids: Sequence[str],
    profile_sha256: str,
    normalization_atol: float,
) -> tuple[str, ...]:
    if not isinstance(rows, np.ndarray) or rows.dtype != np.dtype(np.float32):
        raise TypeError("rows must have dtype float32")
    if rows.ndim != 2:
        raise ValueError("rows must be two-dimensional")
    if rows.shape[0] == 0:
        raise ValueError("rows must contain at least one row")
    if rows.shape[1] == 0:
        raise ValueError("rows must contain at least one dimension")
    if not bool(np.isfinite(rows).all()):
        raise ValueError("rows must contain only finite values")
    squared_norms = np.einsum("ij,ij->i", rows, rows, dtype=np.float32)
    if bool(np.any(squared_norms == np.float32(0.0))):
        raise ValueError("rows must be non-zero")
    np.sqrt(squared_norms, out=squared_norms)
    if not math.isfinite(normalization_atol) or normalization_atol <= 0.0:
        raise ValueError("normalization_atol must be finite and positive")
    deviations = np.abs(squared_norms - np.float32(1.0))
    if not bool(np.all(deviations <= normalization_atol)):
        raise ValueError(
            "rows must be row-wise L2-normalized within normalization_atol"
        )
    if _SHA256_RE.fullmatch(profile_sha256) is None:
        raise ValueError("profile_sha256 must be a lowercase SHA-256 digest")

    stable_row_ids = tuple(row_ids)
    if len(stable_row_ids) != rows.shape[0]:
        raise ValueError("row_ids count must match row count")
    if any(not isinstance(row_id, str) for row_id in stable_row_ids):
        raise TypeError("row_ids must contain strings")
    if any(not row_id for row_id in stable_row_ids):
        raise ValueError("row_ids must not contain empty identifiers")
    if len(set(stable_row_ids)) != len(stable_row_ids):
        raise ValueError("row_ids must be unique")
    return stable_row_ids


def _build_manifest(
    *,
    vectors_path: Path,
    metadata_path: Path,
    row_count: int,
    dimension: int,
    profile_sha256: str,
    normalization_atol: float,
) -> VectorGenerationManifest:
    vectors_sha256 = _file_sha256(vectors_path)
    metadata_sha256 = _file_sha256(metadata_path)
    identity_payload = _generation_identity_payload(
        vectors_sha256=vectors_sha256,
        metadata_sha256=metadata_sha256,
        profile_sha256=profile_sha256,
        row_count=row_count,
        dimension=dimension,
        normalization_atol=normalization_atol,
    )
    manifest_payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "generation_id": f"sha256-{_canonical_sha256(identity_payload)}",
        "format": "npy",
        "npy_version": "2.0",
        "dtype": "<f2",
        "order": "C",
        "row_count": row_count,
        "dimension": dimension,
        "normalized_source": True,
        "normalization_atol": normalization_atol,
        "row_id_kind": "embedding_span_id",
        "vectors_filename": _VECTOR_FILENAME,
        "metadata_filename": _METADATA_FILENAME,
        "vectors_file_bytes": vectors_path.stat().st_size,
        "vector_payload_bytes": row_count * dimension * 2,
        "metadata_file_bytes": metadata_path.stat().st_size,
        "vectors_sha256": vectors_sha256,
        "metadata_sha256": metadata_sha256,
        "profile_sha256": profile_sha256,
    }
    manifest_payload["manifest_sha256"] = _canonical_sha256(manifest_payload)
    return VectorGenerationManifest.model_validate(manifest_payload)


def _remove_staging_directory(staging_dir: Path, staging_root: Path) -> None:
    if staging_dir.parent != staging_root:
        raise RuntimeError("Refusing to clean a directory outside the staging root")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)


class ExactVectorStore:
    def __init__(
        self,
        *,
        generation_dir: Path,
        manifest: VectorGenerationManifest,
        row_ids: tuple[str, ...],
        vectors: np.memmap[Any, Any],
    ) -> None:
        self.generation_dir = generation_dir
        self.manifest = manifest
        self.row_ids = row_ids
        self._vectors = vectors
        self._closed = False

    @classmethod
    def build(
        cls,
        root: Path,
        *,
        rows: NDArray[np.float32],
        row_ids: Sequence[str],
        profile_sha256: str,
        normalization_atol: float,
    ) -> Self:
        stable_row_ids = _validate_build_inputs(
            rows=rows,
            row_ids=row_ids,
            profile_sha256=profile_sha256,
            normalization_atol=normalization_atol,
        )
        workspace = Path(root)
        staging_root = workspace / ".staging"
        generations_root = workspace / "generations"
        staging_root.mkdir(parents=True, exist_ok=True)
        generations_root.mkdir(parents=True, exist_ok=True)
        staging_dir = staging_root / str(uuid.uuid4())
        staging_dir.mkdir()

        try:
            vectors_path = staging_dir / _VECTOR_FILENAME
            metadata_path = staging_dir / _METADATA_FILENAME
            manifest_path = staging_dir / _MANIFEST_FILENAME

            writable = np.lib.format.open_memmap(
                vectors_path,
                mode="w+",
                dtype=np.dtype("<f2"),
                shape=rows.shape,
                fortran_order=False,
                version=_NPY_VERSION,
            )
            try:
                writable[...] = rows
                writable.flush()
            finally:
                _close_mapping(writable)
            _fsync_file(vectors_path)

            metadata = _VectorMetadata(
                row_count=rows.shape[0],
                row_ids=stable_row_ids,
            )
            _write_canonical_json(metadata_path, metadata.model_dump(mode="json"))
            _fsync_file(metadata_path)

            manifest = _build_manifest(
                vectors_path=vectors_path,
                metadata_path=metadata_path,
                row_count=rows.shape[0],
                dimension=rows.shape[1],
                profile_sha256=profile_sha256,
                normalization_atol=normalization_atol,
            )
            staged_store = cls._open_verified(
                staging_dir,
                expected_manifest=manifest,
                require_generation_name=False,
                require_manifest=False,
            )
            staged_store.close()

            _write_canonical_json(manifest_path, manifest.model_dump(mode="json"))
            _fsync_file(manifest_path)
            staged_store = cls._open_verified(
                staging_dir,
                expected_manifest=manifest,
                require_generation_name=False,
                require_manifest=True,
            )
            staged_store.close()

            destination = generations_root / manifest.generation_id
            if destination.exists():
                return cls._reuse_existing(destination, manifest)

            try:
                os.replace(staging_dir, destination)
            except OSError as publication_error:
                if destination.exists():
                    try:
                        return cls._reuse_existing(destination, manifest)
                    except (OSError, ValueError) as collision_error:
                        raise ValueError(
                            "Existing generation does not match the expected content."
                        ) from collision_error
                raise publication_error
            return cls.open(destination)
        finally:
            _remove_staging_directory(staging_dir, staging_root)

    @classmethod
    def open(cls, generation_dir: Path) -> Self:
        return cls._open_verified(
            Path(generation_dir),
            expected_manifest=None,
            require_generation_name=True,
            require_manifest=True,
        )

    @classmethod
    def _reuse_existing(
        cls,
        generation_dir: Path,
        expected_manifest: VectorGenerationManifest,
    ) -> Self:
        try:
            existing = cls.open(generation_dir)
        except (OSError, ValueError) as error:
            raise ValueError(
                "Existing generation does not match the expected content."
            ) from error
        if existing.manifest != expected_manifest:
            existing.close()
            raise ValueError(
                "Existing generation does not match the expected content."
            )
        return existing

    @classmethod
    def _open_verified(
        cls,
        generation_dir: Path,
        *,
        expected_manifest: VectorGenerationManifest | None,
        require_generation_name: bool,
        require_manifest: bool,
    ) -> Self:
        if not generation_dir.is_dir():
            raise ValueError("Generation directory does not exist")
        manifest_path = generation_dir / _MANIFEST_FILENAME
        if require_manifest:
            manifest_payload = _load_json_object(manifest_path)
            manifest = VectorGenerationManifest.model_validate(manifest_payload)
            if manifest_path.read_bytes() != _canonical_json_file_bytes(
                manifest.model_dump(mode="json")
            ):
                raise ValueError("manifest.json is not canonical UTF-8 JSON")
        elif expected_manifest is not None:
            manifest = expected_manifest
        else:
            raise ValueError("Expected manifest is required for staged verification")

        if expected_manifest is not None and manifest != expected_manifest:
            raise ValueError("Generation manifest does not match expected content")
        if require_generation_name and generation_dir.name != manifest.generation_id:
            raise ValueError("Generation directory name does not match generation_id")

        vectors_path = generation_dir / manifest.vectors_filename
        metadata_path = generation_dir / manifest.metadata_filename
        if vectors_path.stat().st_size != manifest.vectors_file_bytes:
            raise ValueError("vectors.npy file size does not match the manifest")
        if metadata_path.stat().st_size != manifest.metadata_file_bytes:
            raise ValueError("vectors.meta.json file size does not match the manifest")
        if not hmac.compare_digest(_file_sha256(vectors_path), manifest.vectors_sha256):
            raise ValueError("vectors.npy SHA-256 does not match the manifest")
        if not hmac.compare_digest(
            _file_sha256(metadata_path), manifest.metadata_sha256
        ):
            raise ValueError("vectors.meta.json SHA-256 does not match the manifest")

        metadata_payload = _load_json_object(metadata_path)
        metadata = _VectorMetadata.model_validate(metadata_payload)
        if metadata_path.read_bytes() != _canonical_json_file_bytes(
            metadata.model_dump(mode="json")
        ):
            raise ValueError("vectors.meta.json is not canonical UTF-8 JSON")
        if metadata.row_count != manifest.row_count:
            raise ValueError("Metadata row count does not match the manifest")
        if metadata.row_id_kind != manifest.row_id_kind:
            raise ValueError("Metadata row ID kind does not match the manifest")

        shape, fortran_order, header_dtype = _read_npy_header(vectors_path)
        if shape != (manifest.row_count, manifest.dimension):
            raise ValueError("vectors.npy shape does not match the manifest")
        if fortran_order:
            raise ValueError("vectors.npy must use C order")
        if header_dtype.str != manifest.dtype:
            raise ValueError("vectors.npy dtype does not match the manifest")

        loaded = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
        if not isinstance(loaded, np.memmap):
            raise ValueError("vectors.npy did not reopen as a memory map")
        mapping = loaded
        try:
            if mapping.shape != shape:
                raise ValueError("Reopened vectors.npy shape does not match its header")
            if mapping.dtype.str != manifest.dtype:
                raise ValueError("Reopened vectors.npy dtype does not match its header")
            if not mapping.flags.c_contiguous:
                raise ValueError("Reopened vectors.npy must be C-contiguous")
            if mapping.flags.writeable:
                raise ValueError("Reopened vectors.npy must be read-only")
            if not _mapping_is_finite(mapping):
                raise ValueError("vectors.npy must contain only finite values")
        except BaseException:
            _close_mapping(mapping)
            raise

        return cls(
            generation_dir=generation_dir,
            manifest=manifest,
            row_ids=metadata.row_ids,
            vectors=mapping,
        )

    def search(
        self,
        query: NDArray[np.float32],
        *,
        limit: int,
        block_rows: int,
    ) -> tuple[VectorHit, ...]:
        self._ensure_open()
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be positive")
        if (
            isinstance(block_rows, bool)
            or not isinstance(block_rows, int)
            or block_rows <= 0
        ):
            raise ValueError("block_rows must be positive")
        if not isinstance(query, np.ndarray) or query.dtype != np.dtype(np.float32):
            raise TypeError("query must have dtype float32")
        if query.ndim != 1:
            raise ValueError("query must be one-dimensional")
        if query.shape[0] != self.manifest.dimension:
            raise ValueError("query dimension must match the store")
        if not bool(np.isfinite(query).all()):
            raise ValueError("query must contain only finite values")

        normalized_query = query.copy(order="C")
        query_scale = np.max(np.abs(normalized_query))
        if query_scale == np.float32(0.0):
            raise ValueError("query must be non-zero")
        normalized_query /= query_scale
        norm_squared = np.dot(normalized_query, normalized_query)
        query_norm = np.sqrt(norm_squared)
        normalized_query /= query_norm

        result_limit = min(limit, self.manifest.row_count)
        best: list[VectorHit] = []
        for start in range(0, self.manifest.row_count, block_rows):
            stop = min(start + block_rows, self.manifest.row_count)
            float32_block = self._vectors[start:stop].astype(np.float32, copy=True)
            scores = np.einsum(
                "ij,j->i",
                float32_block,
                normalized_query,
                dtype=np.float32,
                optimize=False,
            )
            best.extend(
                VectorHit(
                    row_id=self.row_ids[vector_row],
                    score=float(scores[vector_row - start]),
                    vector_row=vector_row,
                )
                for vector_row in range(start, stop)
            )
            best.sort(key=lambda hit: (-hit.score, hit.vector_row))
            del best[result_limit:]
            del scores
            del float32_block
        return tuple(best)

    def close(self) -> None:
        if self._closed:
            return
        _close_mapping(self._vectors)
        self._closed = True

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ExactVectorStore is closed")
