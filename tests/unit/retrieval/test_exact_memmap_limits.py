"""Tiny existing-format artifacts exercise generic admission, without models."""

import io
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from academic_chatbot.retrieval import exact_memmap as vector


@pytest.fixture
def artifact(tmp_path):
    with vector.ExactVectorStore.build(
        tmp_path, rows=np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32),
        row_ids=("span-a", "span-b", "span-c"), profile_sha256="a" * 64,
        normalization_atol=0.001,
    ) as store:
        return store.generation_dir


def limits(path, **changes):
    values = dict(max_manifest_bytes=(path / "manifest.json").stat().st_size,
                  max_metadata_bytes=(path / "vectors.meta.json").stat().st_size,
                  max_vectors_file_bytes=(path / "vectors.npy").stat().st_size,
                  max_rows=3, expected_dimension=2)
    values.update(changes)
    return vector.VectorOpenLimits(**values)


def forbidden(*args, **kwargs):
    raise AssertionError("Expensive or unrestricted operation occurred before admission")


def test_open_with_limits_rejects_actual_metadata_overflow(artifact, monkeypatch):
    bound = limits(artifact)
    original = Path.open

    class GrowingReader:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, size=-1):
            assert 0 <= size <= bound.max_metadata_bytes + 1
            # Actual stream exceeds the unchanged small stat and declaration.
            return self.handle.read(size) + b" "

    def opening(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        return GrowingReader(handle) if path.name == "vectors.meta.json" else handle

    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(vector.VectorOpenLimitError):
        vector.ExactVectorStore.open(artifact, limits=bound)


@pytest.mark.parametrize("change", [dict(max_rows=2), dict(expected_dimension=3),
                                    dict(max_vectors_file_bytes=1), dict(max_metadata_bytes=1)])
def test_open_with_limits_rejects_shape_before_mapping(artifact, monkeypatch, change):
    bound = limits(artifact, **change)
    monkeypatch.setattr(vector.np, "memmap", forbidden)
    monkeypatch.setattr(vector.np, "load", forbidden)
    with pytest.raises(vector.VectorOpenLimitError):
        vector.ExactVectorStore.open(artifact, limits=bound)


def test_limited_and_default_open_return_identical_search_results(artifact, monkeypatch):
    bound = limits(artifact)
    with vector.ExactVectorStore.open(artifact) as default:
        monkeypatch.setattr(Path, "read_bytes", forbidden)
        monkeypatch.setattr(vector.np, "load", forbidden)
        with vector.ExactVectorStore.open(artifact, limits=bound) as limited:
            assert limited.manifest == default.manifest
            assert limited.row_ids == default.row_ids
            assert limited._vectors.dtype == default._vectors.dtype == np.dtype("<f2")
            assert not limited._vectors.flags.writeable
            for query in ([1, 0], [0, 1], [1, 1], [-1, 1]):
                for count in (1, 3, 5):
                    args = dict(limit=count, block_rows=2)
                    q = np.array(query, dtype=np.float32)
                    assert limited.search(q, **args) == default.search(q, **args)
        assert limited._vectors._mmap.closed
        with pytest.raises(RuntimeError, match="closed"):
            limited.search(np.array([1, 0], dtype=np.float32), limit=1, block_rows=1)


@pytest.mark.parametrize("filename,field", [("manifest.json", "max_manifest_bytes"),
    ("vectors.meta.json", "max_metadata_bytes"), ("vectors.npy", "max_vectors_file_bytes")])
def test_exact_fit_and_one_byte_overflow(artifact, filename, field):
    size = (artifact / filename).stat().st_size
    with vector.ExactVectorStore.open(artifact, limits=limits(artifact, **{field: size})):
        pass
    with pytest.raises(vector.VectorOpenLimitError):
        vector.ExactVectorStore.open(artifact, limits=limits(artifact, **{field: size - 1}))


@pytest.mark.parametrize("field", ["max_manifest_bytes", "max_metadata_bytes",
                                 "max_vectors_file_bytes", "max_rows", "expected_dimension"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "1"])
def test_limits_reject_boolean_and_nonpositive_values(artifact, field, value):
    with pytest.raises(ValueError):
        limits(artifact, **{field: value})


def test_limits_are_frozen(artifact):
    bound = limits(artifact)
    with pytest.raises(FrozenInstanceError):
        bound.max_rows = 20


def test_none_preserves_existing_open_path(monkeypatch, tmp_path):
    calls = []
    sentinel = object()

    def verified(cls, path, **kwargs):
        calls.append((path, kwargs))
        return sentinel

    monkeypatch.setattr(vector.ExactVectorStore, "_open_verified", classmethod(verified))
    assert vector.ExactVectorStore.open(tmp_path) is sentinel
    assert vector.ExactVectorStore.open(tmp_path, limits=None) is sentinel
    assert calls == [(tmp_path, dict(expected_manifest=None, require_generation_name=True,
                                    require_manifest=True))] * 2


@pytest.mark.parametrize("filename", ["manifest.json", "vectors.meta.json", "vectors.npy"])
def test_corruption_is_not_admission_error(artifact, filename):
    bound = limits(artifact)
    path = artifact / filename
    raw = bytearray(path.read_bytes())
    raw[-2] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError) as caught:
        vector.ExactVectorStore.open(artifact, limits=bound)
    assert not isinstance(caught.value, vector.VectorOpenLimitError)


def test_artifact_change_during_verification_rejected_and_mapping_closed(artifact, monkeypatch):
    bound = limits(artifact)
    captured = []

    def changed(mapping):
        captured.append(mapping)
        with (artifact / "manifest.json").open("ab") as handle:
            handle.write(b" ")
        return True

    monkeypatch.setattr(vector, "_mapping_is_finite", changed)
    with pytest.raises(ValueError, match="changed"):
        vector.ExactVectorStore.open(artifact, limits=bound)
    assert captured[0]._mmap.closed


def test_nonfinite_failure_closes_mapping_preserving_primary_error(artifact, monkeypatch):
    bound = limits(artifact)
    captured = []
    original_close = vector._close_mapping

    def nonfinite(mapping):
        captured.append(mapping)
        return False

    def noisy_close(mapping):
        original_close(mapping)
        raise OSError("secondary cleanup error")

    monkeypatch.setattr(vector, "_mapping_is_finite", nonfinite)
    monkeypatch.setattr(vector, "_close_mapping", noisy_close)
    with pytest.raises(ValueError, match="finite"):
        vector.ExactVectorStore.open(artifact, limits=bound)
    assert captured[0]._mmap.closed


def test_header_length_rejected_before_body_parse(artifact, monkeypatch):
    bound = limits(artifact)
    # Stream/hash admission is tested separately. Substitute a hostile header
    # only at the header reader to prove the body is never requested.
    stream = io.BytesIO(b"\x93NUMPY\x02\x00" + (bound.max_manifest_bytes + 1).to_bytes(4, "little"))
    with pytest.raises(vector.VectorOpenLimitError):
        vector._read_bounded_npy_header(stream, max_header_bytes=bound.max_manifest_bytes)
    assert stream.tell() == 12


def test_vector_stream_uses_aggregate_ceiling_and_overflow_byte():
    class RecordedStream(io.BytesIO):
        def __init__(self, raw):
            super().__init__(raw)
            self.sizes = []

        def read(self, size=-1):
            assert size >= 0
            self.sizes.append(size)
            return super().read(size)

    stream = RecordedStream(b"123456")
    with pytest.raises(vector.VectorOpenLimitError):
        vector._hash_bounded_stream(stream, 5)
    assert stream.tell() == 6
    assert sum(stream.sizes) == 6


def test_actual_vector_overflow_ignores_deceptively_small_stat(artifact, monkeypatch):
    bound = limits(artifact)
    original = Path.open
    opened = []

    class GrowingReader:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, size=-1):
            assert 0 <= size <= bound.max_vectors_file_bytes + 1
            return self.handle.read(size) + b"x"

    def opening(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        opened.append(handle)
        return GrowingReader(handle) if path.name == "vectors.npy" else handle

    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(vector.np, "memmap", forbidden)
    with pytest.raises(vector.VectorOpenLimitError):
        vector.ExactVectorStore.open(artifact, limits=bound)
    assert len(opened) == 3 and all(handle.closed for handle in opened)


def rewritten_artifact(artifact, vector_bytes):
    # Re-sign a tiny malformed payload so tests reach structural checks rather
    # than stopping at an unrelated digest mismatch. No production format changes.
    (artifact / "vectors.npy").write_bytes(vector_bytes)
    manifest = vector._build_manifest(vectors_path=artifact / "vectors.npy",
        metadata_path=artifact / "vectors.meta.json", row_count=3, dimension=2,
        profile_sha256="a" * 64, normalization_atol=0.001)
    target = artifact.parent / manifest.generation_id
    target.mkdir(exist_ok=True)
    (target / "vectors.npy").write_bytes(vector_bytes)
    (target / "vectors.meta.json").write_bytes((artifact / "vectors.meta.json").read_bytes())
    (target / "manifest.json").write_bytes(vector._canonical_json_file_bytes(
        manifest.model_dump(mode="json")))
    return target


@pytest.mark.parametrize("kind", ["shape", "dtype", "order", "version", "extent", "nonfinite"])
def test_signed_invalid_npy_remains_integrity_failure(artifact, monkeypatch, kind):
    rows = np.array([[1, 0], [0, 1], [1, 0]], dtype="<f2")
    if kind == "shape":
        rows = np.ones((3, 3), dtype="<f2")
    elif kind == "dtype":
        rows = rows.astype("<f4")
    elif kind == "order":
        rows = np.asfortranarray(rows)
    elif kind == "nonfinite":
        rows[0, 0] = np.nan
    stream = io.BytesIO()
    np.lib.format.write_array(stream, rows, version=(1, 0) if kind == "version" else (2, 0))
    raw = stream.getvalue() + (b"trailing" if kind == "extent" else b"")
    path = rewritten_artifact(artifact, raw)
    bound = limits(path)
    if kind != "nonfinite":
        monkeypatch.setattr(vector.np, "memmap", forbidden)
    with pytest.raises(ValueError) as caught:
        vector.ExactVectorStore.open(path, limits=bound)
    assert not isinstance(caught.value, vector.VectorOpenLimitError)


def test_binary_cleanup_error_does_not_hide_primary_error(artifact, monkeypatch):
    bound = limits(artifact)
    original = Path.open
    opened = []

    class BadManifest:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, size=-1):
            return b"invalid JSON"

        def close(self):
            self.handle.close()
            raise OSError("secondary handle close failure")

    def opening(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        opened.append(handle)
        return BadManifest(handle)

    monkeypatch.setattr(Path, "open", opening)
    with pytest.raises(ValueError, match="JSON"):
        vector.ExactVectorStore.open(artifact, limits=bound)
    assert all(handle.closed for handle in opened)


def test_header_extent_rejected_before_reading_past_verified_file():
    stream = io.BytesIO(b"\x93NUMPY\x02\x00" + (100).to_bytes(4, "little") + b"x" * 100)
    with pytest.raises(ValueError, match="extent"):
        vector._read_bounded_npy_header(stream, max_header_bytes=1000, file_bytes=20)
    assert stream.tell() == 12
