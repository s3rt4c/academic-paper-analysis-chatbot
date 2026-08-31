"""Small, same-directory atomic file publication helper."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol


class ReadableBinary(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class StreamCopyLimitError(ValueError):
    """Raised when a streaming copy exceeds the caller's explicit byte limit."""


@dataclass(frozen=True, slots=True)
class StreamCopyResult:
    sha256: str
    byte_length: int


def _write_and_sync(handle: BinaryIO, payload: bytes) -> None:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())


def _replace_file(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def atomic_write_bytes(target: Path, payload: bytes) -> None:
    """Publish *payload* atomically at an existing target parent directory."""

    target_parent = target.parent
    if not target_parent.is_dir():
        message = f"target parent directory does not exist: {target_parent}"
        raise FileNotFoundError(message)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target_parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            _write_and_sync(handle, payload)
        _replace_file(temporary_path, target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_stream_copy(
    source: ReadableBinary,
    target: Path,
    *,
    max_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> StreamCopyResult:
    """Stream a bounded source to a same-directory temporary file, then publish it."""

    if max_bytes <= 0 or chunk_size <= 0:
        raise ValueError("max_bytes and chunk_size must be positive")
    target_parent = target.parent
    if not target_parent.is_dir():
        message = f"target parent directory does not exist: {target_parent}"
        raise FileNotFoundError(message)

    import hashlib

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target_parent
    )
    temporary_path = Path(temporary_name)
    digest = hashlib.sha256()
    byte_length = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            while chunk := source.read(chunk_size):
                byte_length += len(chunk)
                if byte_length > max_bytes:
                    message = f"stream exceeds the configured limit of {max_bytes} bytes"
                    raise StreamCopyLimitError(message)
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_file(temporary_path, target)
        return StreamCopyResult(sha256=digest.hexdigest(), byte_length=byte_length)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
