from __future__ import annotations

from pathlib import Path

import pytest

from academic_chatbot.storage import atomic


def test_atomic_write_publishes_a_complete_new_file(tmp_path: Path) -> None:
    target = tmp_path / "metadata.json"

    atomic.atomic_write_bytes(target, b'{"status":"new"}\n')

    assert target.read_bytes() == b'{"status":"new"}\n'
    assert tuple(tmp_path.glob(f".{target.name}.*.tmp")) == ()


def test_atomic_write_replaces_an_existing_complete_file(tmp_path: Path) -> None:
    target = tmp_path / "metadata.json"
    target.write_bytes(b"old")

    atomic.atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"new"


def test_atomic_write_preserves_previous_target_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "metadata.json"
    target.write_bytes(b"old")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(atomic, "_replace_file", fail_replace)

    with pytest.raises(OSError, match="synthetic replace failure"):
        atomic.atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"old"
    assert tuple(tmp_path.glob(f".{target.name}.*.tmp")) == ()


def test_atomic_write_cleans_its_temp_file_when_writing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "metadata.json"
    target.write_bytes(b"old")

    def fail_write(handle: object, payload: bytes) -> None:
        raise OSError("synthetic write failure")

    monkeypatch.setattr(atomic, "_write_and_sync", fail_write)

    with pytest.raises(OSError, match="synthetic write failure"):
        atomic.atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"old"
    assert tuple(tmp_path.glob(f".{target.name}.*.tmp")) == ()
