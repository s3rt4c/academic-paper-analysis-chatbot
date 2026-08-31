"""Small, same-directory atomic file publication helper."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import BinaryIO


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
