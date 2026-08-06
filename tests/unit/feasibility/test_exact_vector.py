import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError


def _exact_memmap() -> ModuleType:
    try:
        module = importlib.import_module("academic_chatbot.retrieval.exact_memmap")
    except ModuleNotFoundError as error:
        pytest.fail("ExactVectorStore is not implemented.")
        raise AssertionError from error
    if not hasattr(module, "ExactVectorStore"):
        pytest.fail("ExactVectorStore is not implemented.")
    return module


def build_store(
    root: Path, rows: np.ndarray, row_ids: tuple[str, ...]
) -> object:
    return _exact_memmap().ExactVectorStore.build(
        root,
        rows=rows,
        row_ids=row_ids,
        profile_sha256="1" * 64,
        normalization_atol=1e-5,
    )


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_canonical_json(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(_canonical_json_bytes(payload) + b"\n")


def _write_rehashed_manifest(path: Path, payload: dict[str, object]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    payload["manifest_sha256"] = _canonical_sha256(unsigned)
    _write_canonical_json(path, payload)


def _refresh_generation_manifest(generation_dir: Path) -> Path:
    manifest_path = generation_dir / "manifest.json"
    vectors_path = generation_dir / "vectors.npy"
    metadata_path = generation_dir / "vectors.meta.json"
    manifest = _read_json_object(manifest_path)
    manifest["vectors_file_bytes"] = vectors_path.stat().st_size
    manifest["metadata_file_bytes"] = metadata_path.stat().st_size
    manifest["vectors_sha256"] = _file_sha256(vectors_path)
    manifest["metadata_sha256"] = _file_sha256(metadata_path)
    identity = {
        "dimension": manifest["dimension"],
        "dtype": manifest["dtype"],
        "metadata_sha256": manifest["metadata_sha256"],
        "normalization_policy": {
            "normalized_source": manifest["normalized_source"],
            "normalization_atol": manifest["normalization_atol"],
        },
        "npy_version": manifest["npy_version"],
        "order": manifest["order"],
        "profile_sha256": manifest["profile_sha256"],
        "row_count": manifest["row_count"],
        "vectors_sha256": manifest["vectors_sha256"],
    }
    manifest["generation_id"] = f"sha256-{_canonical_sha256(identity)}"
    _write_rehashed_manifest(manifest_path, manifest)
    destination = generation_dir.parent / str(manifest["generation_id"])
    generation_dir.rename(destination)
    return destination


def _replace_npy(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.lib.format.write_array(handle, array, version=(2, 0), allow_pickle=False)


def test_build_rejects_unnormalized_rows(tmp_path: Path) -> None:
    rows = np.array([[2.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="row-wise L2-normalized"):
        build_store(tmp_path, rows, ("span-1",))


def test_build_rejects_non_float32_rows(tmp_path: Path) -> None:
    rows = np.eye(2, dtype=np.float64)

    with pytest.raises(TypeError, match="rows must have dtype float32"):
        build_store(tmp_path, rows, ("span-1", "span-2"))


def test_build_rejects_non_matrix_rows(tmp_path: Path) -> None:
    rows = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="rows must be two-dimensional"):
        build_store(tmp_path, rows, ("span-1",))


def test_build_rejects_zero_rows(tmp_path: Path) -> None:
    rows = np.empty((0, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="rows must contain at least one row"):
        build_store(tmp_path, rows, ())


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_build_rejects_non_finite_rows(
    tmp_path: Path, invalid_value: float
) -> None:
    rows = np.array([[invalid_value, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="rows must contain only finite values"):
        build_store(tmp_path, rows, ("span-1",))


def test_build_rejects_zero_norm_rows(tmp_path: Path) -> None:
    rows = np.array([[0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="rows must be non-zero"):
        build_store(tmp_path, rows, ("span-1",))


def test_build_rejects_row_id_count_mismatch(tmp_path: Path) -> None:
    rows = np.eye(2, dtype=np.float32)

    with pytest.raises(ValueError, match="row_ids count must match row count"):
        build_store(tmp_path, rows, ("span-1",))


def test_build_rejects_empty_row_ids(tmp_path: Path) -> None:
    rows = np.eye(2, dtype=np.float32)

    with pytest.raises(ValueError, match="row_ids must not contain empty identifiers"):
        build_store(tmp_path, rows, ("span-1", ""))


def test_build_rejects_duplicate_row_ids(tmp_path: Path) -> None:
    rows = np.eye(2, dtype=np.float32)

    with pytest.raises(ValueError, match="row_ids must be unique"):
        build_store(tmp_path, rows, ("span-1", "span-1"))


def test_build_rejects_invalid_profile_hash(tmp_path: Path) -> None:
    store_type = _exact_memmap().ExactVectorStore

    with pytest.raises(ValueError, match="profile_sha256"):
        store_type.build(
            tmp_path,
            rows=np.eye(2, dtype=np.float32),
            row_ids=("span-1", "span-2"),
            profile_sha256="A" * 64,
            normalization_atol=1e-5,
        )


@pytest.mark.parametrize("normalization_atol", [0.0, -1.0, np.inf, np.nan])
def test_build_rejects_invalid_normalization_tolerance(
    tmp_path: Path, normalization_atol: float
) -> None:
    store_type = _exact_memmap().ExactVectorStore

    with pytest.raises(ValueError, match="normalization_atol"):
        store_type.build(
            tmp_path,
            rows=np.eye(2, dtype=np.float32),
            row_ids=("span-1", "span-2"),
            profile_sha256="1" * 64,
            normalization_atol=normalization_atol,
        )


def test_published_generation_uses_verified_npy_layout(tmp_path: Path) -> None:
    rows = np.eye(2, dtype=np.float32)
    store = build_store(tmp_path, rows, ("span-1", "span-2"))

    mapped = np.load(
        store.generation_dir / "vectors.npy", mmap_mode="r", allow_pickle=False
    )
    with (store.generation_dir / "vectors.npy").open("rb") as handle:
        npy_version = np.lib.format.read_magic(handle)

    assert isinstance(mapped, np.memmap)
    assert mapped.dtype == np.dtype("<f2")
    assert mapped.shape == (2, 2)
    assert mapped.flags.c_contiguous
    assert mapped.flags.f_contiguous is False
    assert mapped.flags.writeable is False
    assert npy_version == (2, 0)
    assert store.manifest.vector_payload_bytes == 8
    assert store.manifest.vectors_file_bytes == (
        store.generation_dir / "vectors.npy"
    ).stat().st_size


def test_metadata_is_canonical_utf8_and_preserves_row_order(tmp_path: Path) -> None:
    row_ids = ("span-ç", "span-ğ")
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), row_ids)
    metadata_path = store.generation_dir / "vectors.meta.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )

    assert metadata_path.read_bytes() == expected
    assert payload == {
        "row_count": 2,
        "row_id_kind": "embedding_span_id",
        "row_ids": ["span-ç", "span-ğ"],
        "schema_version": "1.0.0",
    }
    assert store.row_ids == row_ids


def test_manifest_is_canonical_self_hashed_frozen_and_extra_forbid(
    tmp_path: Path,
) -> None:
    module = _exact_memmap()
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))
    manifest_path = store.generation_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )

    assert manifest_path.read_bytes() == expected
    assert payload["manifest_sha256"] == store.manifest.manifest_sha256
    assert store.manifest.generation_id == store.generation_dir.name
    with pytest.raises(ValidationError, match="frozen"):
        store.manifest.row_count = 3

    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        module.VectorGenerationManifest.model_validate(payload)


def test_generation_id_is_deterministic_and_content_derived(tmp_path: Path) -> None:
    module = _exact_memmap()
    rows = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    first = build_store(tmp_path / "first", rows, ("a", "b"))
    second = build_store(tmp_path / "second", rows, ("a", "b"))
    different_profile = module.ExactVectorStore.build(
        tmp_path / "third",
        rows=rows,
        row_ids=("a", "b"),
        profile_sha256="2" * 64,
        normalization_atol=1e-5,
    )

    assert first.manifest.generation_id == second.manifest.generation_id
    assert first.manifest.manifest_sha256 == second.manifest.manifest_sha256
    assert first.manifest.generation_id.startswith("sha256-")
    assert len(first.manifest.generation_id) == len("sha256-") + 64
    assert different_profile.manifest.generation_id != first.manifest.generation_id


def test_build_does_not_mutate_caller_inputs(tmp_path: Path) -> None:
    rows = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    before = rows.copy()

    build_store(tmp_path, rows, ("a", "b"))

    np.testing.assert_array_equal(rows, before)


def test_existing_matching_generation_is_verified_and_reused(tmp_path: Path) -> None:
    rows = np.eye(2, dtype=np.float32)
    first = build_store(tmp_path, rows, ("a", "b"))
    second = build_store(tmp_path, rows, ("a", "b"))

    assert second.generation_dir == first.generation_dir
    assert second.manifest == first.manifest
    assert list((tmp_path / ".staging").iterdir()) == []


def test_publish_failure_preserves_existing_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _exact_memmap()
    first = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))
    real_replace = module.os.replace

    def fail_publish(source: Path, destination: Path) -> None:
        if source.is_dir():
            raise OSError("simulated publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_publish)
    rows = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    with pytest.raises(OSError, match="simulated publish failure"):
        build_store(tmp_path, rows, ("c", "d"))

    assert first.generation_dir.is_dir()
    assert module.ExactVectorStore.open(first.generation_dir).manifest == first.manifest
    assert list((tmp_path / ".staging").iterdir()) == []


def test_matching_generation_published_by_racer_is_verified_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _exact_memmap()
    rows = np.eye(2, dtype=np.float32)
    published = build_store(tmp_path / "publisher", rows, ("a", "b"))
    real_replace = module.os.replace

    def race_publish(source: Path, destination: Path) -> None:
        if source.is_dir():
            shutil.copytree(published.generation_dir, destination)
            raise FileExistsError("simulated publication race")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", race_publish)
    reused = build_store(tmp_path / "target", rows, ("a", "b"))

    assert reused.manifest == published.manifest
    assert list((tmp_path / "target" / ".staging").iterdir()) == []


def test_mismatched_generation_published_by_racer_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _exact_memmap()
    expected_rows = np.eye(2, dtype=np.float32)
    other_rows = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    published = build_store(tmp_path / "publisher", other_rows, ("c", "d"))
    real_replace = module.os.replace

    def race_publish(source: Path, destination: Path) -> None:
        if source.is_dir():
            shutil.copytree(published.generation_dir, destination)
            raise PermissionError("simulated publication race")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", race_publish)
    with pytest.raises(
        ValueError, match="Existing generation does not match the expected content"
    ):
        build_store(tmp_path / "target", expected_rows, ("a", "b"))

    assert published.generation_dir.is_dir()
    assert list((tmp_path / "target" / ".staging").iterdir()) == []


def test_staging_and_generation_share_the_workspace_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _exact_memmap()
    real_replace = module.os.replace
    observed: list[tuple[Path, Path]] = []

    def observe_publish(source: Path, destination: Path) -> None:
        if source.is_dir():
            observed.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", observe_publish)
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))

    assert observed == [(observed[0][0], store.generation_dir)]
    assert observed[0][0].parent == tmp_path / ".staging"
    assert observed[0][1].parent == tmp_path / "generations"
    assert observed[0][0].drive == observed[0][1].drive


def test_search_matches_float32_oracle_and_returns_stable_rows(tmp_path: Path) -> None:
    rows = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    store = build_store(tmp_path, rows, ("a", "b", "c"))

    hits = store.search(
        np.array([1.0, 0.0], dtype=np.float32), limit=2, block_rows=2
    )

    assert [(hit.row_id, hit.vector_row) for hit in hits] == [("a", 0), ("b", 1)]
    np.testing.assert_allclose(
        [hit.score for hit in hits],
        [1.0, np.float32(np.float16(0.8))],
        rtol=0.0,
        atol=0.0,
    )


def test_equal_scores_across_blocks_use_vector_row_order(tmp_path: Path) -> None:
    rows = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    store = build_store(tmp_path, rows, ("a", "b", "c", "d"))

    hits = store.search(
        np.array([1.0, 0.0], dtype=np.float32), limit=2, block_rows=2
    )

    assert [hit.vector_row for hit in hits] == [0, 2]


@pytest.mark.parametrize("limit", [0, -1])
def test_search_rejects_non_positive_limit(tmp_path: Path, limit: int) -> None:
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))

    with pytest.raises(ValueError, match="limit must be positive"):
        store.search(np.ones(2, dtype=np.float32), limit=limit, block_rows=1)


@pytest.mark.parametrize("block_rows", [0, -1])
def test_search_rejects_non_positive_block_rows(
    tmp_path: Path, block_rows: int
) -> None:
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))

    with pytest.raises(ValueError, match="block_rows must be positive"):
        store.search(np.ones(2, dtype=np.float32), limit=1, block_rows=block_rows)


def test_search_rejects_non_float32_query(tmp_path: Path) -> None:
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))

    with pytest.raises(TypeError, match="query must have dtype float32"):
        store.search(np.ones(2, dtype=np.float64), limit=1, block_rows=1)


def test_search_rejects_non_vector_query(tmp_path: Path) -> None:
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))

    with pytest.raises(ValueError, match="query must be one-dimensional"):
        store.search(np.ones((1, 2), dtype=np.float32), limit=1, block_rows=1)


def test_search_rejects_wrong_query_dimension(tmp_path: Path) -> None:
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))

    with pytest.raises(ValueError, match="query dimension must match the store"):
        store.search(np.ones(3, dtype=np.float32), limit=1, block_rows=1)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_search_rejects_non_finite_query(
    tmp_path: Path, invalid_value: float
) -> None:
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))
    query = np.array([invalid_value, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="query must contain only finite values"):
        store.search(query, limit=1, block_rows=1)


def test_search_rejects_zero_norm_query(tmp_path: Path) -> None:
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))

    with pytest.raises(ValueError, match="query must be non-zero"):
        store.search(np.zeros(2, dtype=np.float32), limit=1, block_rows=1)


def test_limit_above_row_count_returns_every_row_in_global_order(
    tmp_path: Path,
) -> None:
    rows = np.array([[0.0, 1.0], [1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
    store = build_store(tmp_path, rows, ("a", "b", "c"))

    hits = store.search(
        np.array([1.0, 0.0], dtype=np.float32), limit=99, block_rows=1
    )

    assert [(hit.row_id, hit.vector_row) for hit in hits] == [
        ("b", 1),
        ("c", 2),
        ("a", 0),
    ]


def test_search_is_independent_of_block_size_including_scores(tmp_path: Path) -> None:
    rows = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.6, 0.8],
            [0.0, 1.0],
            [-0.6, 0.8],
        ],
        dtype=np.float32,
    )
    store = build_store(tmp_path, rows, ("a", "b", "c", "d", "e"))
    query = np.array([0.6, 0.8], dtype=np.float32)

    single_row_blocks = store.search(query, limit=4, block_rows=1)
    one_block = store.search(query, limit=4, block_rows=rows.shape[0])

    assert single_row_blocks == one_block


def test_search_normalizes_a_copy_without_mutating_the_caller(
    tmp_path: Path,
) -> None:
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))
    query = np.array([3.0, 4.0], dtype=np.float32)
    before = query.copy()

    hits = store.search(query, limit=2, block_rows=1)

    np.testing.assert_array_equal(query, before)
    np.testing.assert_allclose(
        [hits[0].score, hits[1].score],
        [0.8, 0.6],
        rtol=0.0,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "magnitude",
    [np.finfo(np.float32).tiny, np.finfo(np.float32).max],
)
def test_search_normalizes_extreme_finite_nonzero_queries(
    tmp_path: Path, magnitude: np.float32
) -> None:
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))
    query = np.array([magnitude, 0.0], dtype=np.float32)

    hits = store.search(query, limit=1, block_rows=1)

    assert hits[0].row_id == "a"
    assert hits[0].score == 1.0


def test_reopened_store_searches_the_same_generation(tmp_path: Path) -> None:
    module = _exact_memmap()
    rows = np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
    built = build_store(tmp_path, rows, ("a", "b"))
    built.close()

    reopened = module.ExactVectorStore.open(built.generation_dir)
    hits = reopened.search(
        np.array([1.0, 0.0], dtype=np.float32), limit=2, block_rows=1
    )

    assert [hit.row_id for hit in hits] == ["a", "b"]
    assert reopened.manifest == built.manifest


def test_open_rejects_vector_file_tampering(tmp_path: Path) -> None:
    module = _exact_memmap()
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))
    generation_dir = store.generation_dir
    store.close()
    vectors_path = generation_dir / "vectors.npy"
    with vectors_path.open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 1]))

    with pytest.raises(ValueError, match=r"vectors\.npy SHA-256"):
        module.ExactVectorStore.open(generation_dir)


def test_open_rejects_metadata_file_tampering(tmp_path: Path) -> None:
    module = _exact_memmap()
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))
    generation_dir = store.generation_dir
    store.close()
    metadata_path = generation_dir / "vectors.meta.json"
    metadata_path.write_bytes(metadata_path.read_bytes().replace(b'"a"', b'"z"'))

    with pytest.raises(ValueError, match=r"vectors\.meta\.json SHA-256"):
        module.ExactVectorStore.open(generation_dir)


def test_open_rejects_manifest_self_hash_tampering(tmp_path: Path) -> None:
    module = _exact_memmap()
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))
    generation_dir = store.generation_dir
    store.close()
    manifest_path = generation_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["vectors_file_bytes"] += 1
    manifest_path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )

    with pytest.raises(ValueError, match="manifest_sha256"):
        module.ExactVectorStore.open(generation_dir)


def test_open_rejects_non_canonical_manifest_bytes(tmp_path: Path) -> None:
    module = _exact_memmap()
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))
    generation_dir = store.generation_dir
    store.close()
    manifest_path = generation_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="",
    )

    with pytest.raises(
        ValueError, match=r"manifest\.json is not canonical UTF-8 JSON"
    ):
        module.ExactVectorStore.open(generation_dir)


def test_open_rejects_rehashed_metadata_count_mismatch(tmp_path: Path) -> None:
    module = _exact_memmap()
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))
    generation_dir = store.generation_dir
    store.close()
    metadata_path = generation_dir / "vectors.meta.json"
    metadata = _read_json_object(metadata_path)
    metadata["row_count"] = 1
    metadata["row_ids"] = ["a"]
    _write_canonical_json(metadata_path, metadata)
    generation_dir = _refresh_generation_manifest(generation_dir)

    with pytest.raises(ValueError, match="Metadata row count does not match"):
        module.ExactVectorStore.open(generation_dir)


def test_open_rejects_rehashed_metadata_row_id_kind(tmp_path: Path) -> None:
    module = _exact_memmap()
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))
    generation_dir = store.generation_dir
    store.close()
    metadata_path = generation_dir / "vectors.meta.json"
    metadata = _read_json_object(metadata_path)
    metadata["row_id_kind"] = "document_id"
    _write_canonical_json(metadata_path, metadata)
    generation_dir = _refresh_generation_manifest(generation_dir)

    with pytest.raises(ValueError, match="row_id_kind"):
        module.ExactVectorStore.open(generation_dir)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (
            np.ones((1, 4), dtype=np.dtype("<f2")),
            r"vectors\.npy shape does not match the manifest",
        ),
        (
            np.asfortranarray(
                np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.dtype("<f2"))
            ),
            r"vectors\.npy must use C order",
        ),
        (
            np.eye(2, dtype=np.dtype("<f4")),
            r"vectors\.npy dtype does not match the manifest",
        ),
    ],
    ids=("shape", "fortran-order", "dtype"),
)
def test_open_rejects_rehashed_npy_header_semantic_mismatch(
    tmp_path: Path, replacement: np.ndarray, message: str
) -> None:
    module = _exact_memmap()
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))
    generation_dir = store.generation_dir
    store.close()
    _replace_npy(generation_dir / "vectors.npy", replacement)
    generation_dir = _refresh_generation_manifest(generation_dir)

    with pytest.raises(ValueError, match=message):
        module.ExactVectorStore.open(generation_dir)


def test_open_rejects_rehashed_non_finite_mapped_vector(tmp_path: Path) -> None:
    module = _exact_memmap()
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))
    generation_dir = store.generation_dir
    store.close()
    replacement = np.eye(2, dtype=np.dtype("<f2"))
    replacement[0, 0] = np.nan
    _replace_npy(generation_dir / "vectors.npy", replacement)
    generation_dir = _refresh_generation_manifest(generation_dir)

    with pytest.raises(ValueError, match=r"vectors\.npy must contain only finite"):
        module.ExactVectorStore.open(generation_dir)


def test_racing_fully_hashed_but_semantically_invalid_generation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _exact_memmap()
    rows = np.eye(2, dtype=np.float32)
    published = build_store(tmp_path / "publisher", rows, ("a", "b"))
    published.close()
    manifest_path = published.generation_dir / "manifest.json"
    manifest = _read_json_object(manifest_path)
    manifest["vector_payload_bytes"] = int(manifest["vector_payload_bytes"]) + 2
    _write_rehashed_manifest(manifest_path, manifest)

    assert manifest["vectors_sha256"] == _file_sha256(
        published.generation_dir / "vectors.npy"
    )
    assert manifest["metadata_sha256"] == _file_sha256(
        published.generation_dir / "vectors.meta.json"
    )
    assert manifest["manifest_sha256"] == _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    real_replace = module.os.replace

    def race_publish(source: Path, destination: Path) -> None:
        if source.is_dir():
            shutil.copytree(published.generation_dir, destination)
            raise FileExistsError("simulated semantic publication race")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", race_publish)
    with pytest.raises(
        ValueError, match="Existing generation does not match the expected content"
    ):
        build_store(tmp_path / "target", rows, ("a", "b"))

    destination = (
        tmp_path / "target" / "generations" / published.generation_dir.name
    )
    assert destination.is_dir()
    assert published.generation_dir.is_dir()
    assert list((tmp_path / "target" / ".staging").iterdir()) == []


def test_close_is_idempotent_and_closed_search_fails_stably(tmp_path: Path) -> None:
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))

    store.close()
    store.close()

    with pytest.raises(RuntimeError, match="ExactVectorStore is closed"):
        store.search(np.ones(2, dtype=np.float32), limit=1, block_rows=1)


def test_close_releases_mapping_for_file_replacement(tmp_path: Path) -> None:
    module = _exact_memmap()
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))
    generation_dir = store.generation_dir
    vectors_path = generation_dir / "vectors.npy"
    moved_path = generation_dir / "vectors.closed.npy"

    store.close()
    os.replace(vectors_path, moved_path)
    os.replace(moved_path, vectors_path)

    assert vectors_path.is_file()
    assert not moved_path.exists()
    reopened = module.ExactVectorStore.open(generation_dir)
    reopened.close()


def test_context_manager_closes_the_store(tmp_path: Path) -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    store = build_store(tmp_path, np.eye(2, dtype=np.float32), ("a", "b"))

    with store as entered:
        assert entered is store
        assert entered.search(query, limit=1, block_rows=1)[0].row_id == "a"

    with pytest.raises(RuntimeError, match="ExactVectorStore is closed"):
        store.search(query, limit=1, block_rows=1)


def _exact_vector() -> ModuleType:
    try:
        module = importlib.import_module("academic_chatbot.feasibility.exact_vector")
    except ModuleNotFoundError as error:
        pytest.fail("Exact-vector benchmark is not implemented.")
        raise AssertionError from error
    return module


def _sample_vector_hardware() -> object:
    from academic_chatbot.feasibility.hardware import HardwareFacts, MemoryModuleFact

    return HardwareFacts(
        cpu_model="Test CPU",
        physical_cores=4,
        logical_cores=8,
        instruction_sets=("AVX", "AVX2"),
        ram_bytes=17_179_869_184,
        usable_ram_bytes=16_000_000_000,
        ram_layout=(
            MemoryModuleFact(
                capacity_bytes=17_179_869_184,
                speed_mhz=5600,
                manufacturer="Test Memory",
                part_number="TEST-16G",
                bank_label="BANK 0",
                device_locator="DIMM 0",
            ),
        ),
        windows_build="10.0.test",
        power_profile="Balanced",
        gpu_model="Test GPU",
        vram_bytes=8_000_000_000,
        gpu_offload_available=None,
        storage_kind="NVMe SSD",
        collected_at="2026-07-13T00:00:00Z",
        collection_diagnostics=("fresh inventory",),
    )


def _sample_reference_hardware() -> object:
    from academic_chatbot.feasibility.hardware import build_record

    return build_record(
        facts=_sample_vector_hardware(),
        gguf_name="research-model-Q4_K_M.gguf",
        gguf_sha256="1" * 64,
        gguf_quantization="Q4_K_M",
        model_manifest_sha256="2" * 64,
        llama_release="b-test",
        llama_flags=("--ctx-size", "4096"),
        runtime_manifest_sha256="3" * 64,
        benchmark_corpus_sha256="4" * 64,
        gpu_offload_available=False,
        collected_at="2026-07-13T00:00:00Z",
    )


def _small_vector_profile(module: ModuleType) -> object:
    return module.VectorBenchmarkProfile(
        row_count=8,
        dimension=384,
        query_count=4,
        warmup_query_count=1,
        top_k=4,
        block_rows=3,
        row_seed=101,
        query_seed=202,
        process_tree_sample_interval_ms=1,
    )


def _write_vector_config(path: Path, profile: object, **siblings: object) -> None:
    payload = {
        "schema_version": "1.0.0",
        "vector_exact": profile.model_dump(mode="json"),
        **siblings,
    }
    path.write_bytes(_canonical_json_bytes(payload) + b"\n")


class _FakeProcessTreeSampler:
    entered = 0

    def __init__(self, sample_interval_ms: int) -> None:
        from academic_chatbot.feasibility.process_tree import ProcessTreePeak

        self._result = ProcessTreePeak(
            metric="process_tree_sum_uss_bytes",
            peak_bytes=1_000_000,
            sample_interval_ms=sample_interval_ms,
            sample_count=3,
            process_churn_count=0,
            access_error_count=0,
            measurement_valid=True,
        )

    @property
    def result(self) -> object:
        return self._result

    def __enter__(self) -> "_FakeProcessTreeSampler":
        type(self).entered += 1
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_phase0_vector_profile_has_reproducible_defaults() -> None:
    profile = _exact_vector().VectorBenchmarkProfile()

    assert profile.row_count == 100_000
    assert profile.dimension == 384
    assert profile.query_count == 200
    assert profile.row_seed == 20_260_711
    assert profile.query_seed == 20_260_712
    assert profile.p95_threshold_ms is None
    assert profile.top_k_agreement_threshold is None


def test_vector_profile_rejects_top_k_larger_than_row_count() -> None:
    with pytest.raises(ValidationError, match="top_k cannot exceed row_count"):
        _exact_vector().VectorBenchmarkProfile(row_count=3, top_k=4)


def test_profile_loader_hash_ignores_future_sibling_objects(tmp_path: Path) -> None:
    module = _exact_vector()
    profile = _small_vector_profile(module)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_vector_config(first, profile, later_task={"value": 1})
    _write_vector_config(second, profile, later_task={"value": 2})

    first_profile = module.load_vector_profile(first)
    second_profile = module.load_vector_profile(second)

    assert first_profile == second_profile
    assert module.compute_vector_profile_sha256(first_profile) == (
        module.compute_vector_profile_sha256(second_profile)
    )


def test_profile_loader_requires_canonical_utf8_json(tmp_path: Path) -> None:
    module = _exact_vector()
    path = tmp_path / "phase0.json"
    profile = _small_vector_profile(module)
    path.write_text(
        json.dumps({"schema_version": "1.0.0", "vector_exact": profile.model_dump()}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical UTF-8 JSON"):
        module.load_vector_profile(path)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("row_count", "8"),
        ("normalization_atol", "0.00001"),
        ("process_tree_sample_interval_ms", "1"),
    ],
)
def test_profile_loader_rejects_coercible_canonical_types(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
) -> None:
    module = _exact_vector()
    path = tmp_path / "phase0.json"
    profile = module.VectorBenchmarkProfile(
        row_count=8,
        query_count=4,
        top_k=4,
        process_tree_sample_interval_ms=1,
    ).model_dump(mode="json")
    profile[field_name] = invalid_value
    _write_canonical_json(
        path,
        {"schema_version": "1.0.0", "vector_exact": profile},
    )

    with pytest.raises(
        ValueError,
        match="Vector profile vector_exact object is invalid",
    ):
        module.load_vector_profile(path)


def test_cli_rejects_coercible_canonical_types_before_work_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _exact_vector()
    config = tmp_path / "phase0.json"
    workspace = tmp_path / "workspace"
    output = tmp_path / "report.json"
    profile = module.VectorBenchmarkProfile(
        row_count=8,
        query_count=4,
        top_k=4,
        process_tree_sample_interval_ms=1,
    ).model_dump(mode="json")
    profile["row_count"] = "8"
    _write_canonical_json(
        config,
        {"schema_version": "1.0.0", "vector_exact": profile},
    )
    calls: list[str] = []

    def fail_hardware(_path: Path) -> object:
        calls.append("hardware")
        raise AssertionError("hardware loading must not start")

    def fail_benchmark(**_kwargs: object) -> object:
        calls.append("benchmark")
        raise AssertionError("benchmark allocation must not start")

    monkeypatch.setattr(module, "_load_hardware_facts", fail_hardware)
    monkeypatch.setattr(module, "run_benchmark", fail_benchmark)

    result = module.main(
        [
            "benchmark",
            "--config",
            str(config),
            "--hardware-facts",
            str(tmp_path / "unused-hardware.json"),
            "--workspace",
            str(workspace),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "Vector profile vector_exact object is invalid.\n"
    assert calls == []
    assert not workspace.exists()
    assert not output.exists()


def test_independent_pcg64_seeds_generate_distinct_normalized_matrices() -> None:
    module = _exact_vector()

    rows = module._generate_normalized_matrix(8, 384, seed=101)
    queries = module._generate_normalized_matrix(4, 384, seed=202)

    assert rows.dtype == np.float32
    assert queries.dtype == np.float32
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(queries, axis=1), 1.0, atol=1e-6)
    assert not any(np.array_equal(query, row) for query in queries for row in rows)
    assert np.array_equal(rows, module._generate_normalized_matrix(8, 384, seed=101))


def test_runtime_attestation_ignores_collection_metadata_and_offload() -> None:
    module = _exact_vector()
    expected = _sample_vector_hardware()
    actual = expected.model_copy(
        update={
            "collected_at": "2026-07-13T00:00:01Z",
            "collection_diagnostics": (),
            "gpu_offload_available": True,
        }
    )

    attestation = module.attest_runtime_hardware(expected, actual)

    assert attestation.matches_expected is True
    assert attestation.mismatch_fields == ()
    assert len(attestation.runtime_hardware_facts_sha256) == 64


def test_runtime_attestation_lists_sorted_stable_mismatch_fields() -> None:
    module = _exact_vector()
    expected = _sample_vector_hardware()
    actual = expected.model_copy(
        update={"storage_kind": "SATA SSD", "cpu_model": "Different CPU"}
    )

    attestation = module.attest_runtime_hardware(expected, actual)

    assert attestation.matches_expected is False
    assert attestation.mismatch_fields == ("cpu_model", "storage_kind")


def test_attestation_mismatch_fails_before_workspace_sampler_or_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _exact_vector()
    expected = _sample_vector_hardware()
    actual = expected.model_copy(update={"cpu_model": "Different CPU"})
    calls: list[str] = []
    workspace = tmp_path / "work"

    def collect() -> object:
        calls.append("collect")
        return actual

    def sampler_factory(_: int) -> object:
        calls.append("sampler")
        raise AssertionError("sampler must not start")

    def fail_allocate(*args: object, **kwargs: object) -> object:
        calls.append("allocate")
        raise AssertionError("allocation must not start")

    monkeypatch.setattr(module, "_generate_normalized_matrix", fail_allocate)

    with pytest.raises(
        RuntimeError,
        match=r"Runtime hardware does not match expected fields: cpu_model\.",
    ):
        module.run_benchmark(
            profile=_small_vector_profile(module),
            hardware_source=expected,
            workspace=workspace,
            hardware_collector=collect,
            sampler_factory=sampler_factory,
        )

    assert calls == ["collect"]
    assert not workspace.exists()


@pytest.mark.parametrize(
    (
        "measurement_status",
        "reference_ram_bytes",
        "measurement_valid",
        "peak_bytes",
        "expected_memory_status",
        "expected_eligible",
    ),
    [
        (
            "provisional",
            17_179_869_184,
            True,
            1_000_000,
            "not_evaluated_non_reference_hardware",
            False,
        ),
        ("bound", 8_000_000_000, True, 1_000_000, "not_evaluated_wrong_reference_ram", False),
        ("bound", 17_179_869_184, False, 1_000_000, "not_evaluated_invalid_measurement", False),
        ("bound", 17_179_869_184, True, 12_884_901_888, "failed", False),
        ("bound", 17_179_869_184, True, 1_000_000, "passed", True),
    ],
)
def test_report_status_matrix(
    measurement_status: str,
    reference_ram_bytes: int,
    measurement_valid: bool,
    peak_bytes: int,
    expected_memory_status: str,
    expected_eligible: bool,
) -> None:
    module = _exact_vector()
    status = module.evaluate_report_status(
        profile=module.VectorBenchmarkProfile(),
        measurement_status=measurement_status,
        reference_ram_bytes=reference_ram_bytes,
        generation_integrity_verified=True,
        deterministic_tie_break_verified=True,
        runtime_hardware_match=True,
        measurement_valid=measurement_valid,
        peak_bytes=peak_bytes,
    )

    assert status.memory_gate_status == expected_memory_status
    assert status.p95_gate_status == "not_evaluated_no_approved_threshold"
    assert status.agreement_gate_status == "not_evaluated_no_approved_threshold"
    assert status.gate_eligible is expected_eligible


def test_small_bound_profile_cannot_become_gate_eligible() -> None:
    module = _exact_vector()
    status = module.evaluate_report_status(
        profile=_small_vector_profile(module),
        measurement_status="bound",
        reference_ram_bytes=17_179_869_184,
        generation_integrity_verified=True,
        deterministic_tie_break_verified=True,
        runtime_hardware_match=True,
        measurement_valid=True,
        peak_bytes=1_000_000,
    )

    assert status.reference_profile_verified is False
    assert status.memory_gate_status == "passed"
    assert status.gate_eligible is False


def test_facts_only_cli_writes_provisional_metric_only_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from academic_chatbot.feasibility.hardware import write_hardware_facts

    module = _exact_vector()
    profile = _small_vector_profile(module)
    facts = _sample_vector_hardware()
    config = tmp_path / "phase0.json"
    facts_path = tmp_path / "hardware-facts.json"
    output = tmp_path / "vector-exact-provisional.json"
    _write_vector_config(config, profile)
    write_hardware_facts(facts_path, facts)
    monkeypatch.setattr(module, "collect_windows_hardware", lambda: facts)
    monkeypatch.setattr(module, "ProcessTreePeakSampler", _FakeProcessTreeSampler)

    result = module.main(
        [
            "benchmark",
            "--config",
            str(config),
            "--hardware-facts",
            str(facts_path),
            "--workspace",
            str(tmp_path / "work"),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    report = module.load_vector_report(output)
    assert report.measurement_status == "provisional"
    assert report.memory_gate_status == "not_evaluated_non_reference_hardware"
    assert report.p95_gate_status == "not_evaluated_no_approved_threshold"
    assert report.agreement_gate_status == "not_evaluated_no_approved_threshold"
    assert report.reference_profile_verified is False
    assert report.runtime_hardware_match is True
    assert report.gate_eligible is False
    assert report.query_count == 4
    assert report.search_latency_ms.sample_count == 4
    assert output.read_bytes().endswith(b"\n")


def test_bound_run_records_reference_hash_and_evaluates_memory(
    tmp_path: Path,
) -> None:
    module = _exact_vector()
    profile = _small_vector_profile(module)
    reference = _sample_reference_hardware()

    report = module.run_benchmark(
        profile=profile,
        hardware_source=reference,
        workspace=tmp_path / "work",
        hardware_collector=lambda: _sample_vector_hardware(),
        sampler_factory=_FakeProcessTreeSampler,
    )

    assert report.measurement_status == "bound"
    assert report.reference_hardware_record_sha256 == reference.record_sha256
    assert report.exact_16_gib_reference_verified is True
    assert report.memory_gate_status == "passed"
    assert report.reference_profile_verified is False
    assert report.gate_eligible is False


def test_invalid_bound_process_measurement_is_not_gate_eligible(tmp_path: Path) -> None:
    from academic_chatbot.feasibility.process_tree import ProcessTreePeak

    module = _exact_vector()

    class InvalidSampler(_FakeProcessTreeSampler):
        def __init__(self, sample_interval_ms: int) -> None:
            self._result = ProcessTreePeak(
                metric="process_tree_sum_rss_bytes",
                peak_bytes=1_000_000,
                sample_interval_ms=sample_interval_ms,
                sample_count=2,
                process_churn_count=0,
                access_error_count=1,
                measurement_valid=False,
            )

    report = module.run_benchmark(
        profile=_small_vector_profile(module),
        hardware_source=_sample_reference_hardware(),
        workspace=tmp_path / "work",
        hardware_collector=lambda: _sample_vector_hardware(),
        sampler_factory=InvalidSampler,
    )

    assert report.memory_gate_status == "not_evaluated_invalid_measurement"
    assert report.gate_eligible is False
    assert "Process-tree memory measurement is invalid." in report.failure_reasons


def test_report_loader_rejects_tampering_and_noncanonical_bytes(tmp_path: Path) -> None:
    module = _exact_vector()
    report = module.run_benchmark(
        profile=_small_vector_profile(module),
        hardware_source=_sample_vector_hardware(),
        workspace=tmp_path / "work",
        hardware_collector=lambda: _sample_vector_hardware(),
        sampler_factory=_FakeProcessTreeSampler,
    )
    output = tmp_path / "report.json"
    module.write_vector_report(output, report)
    payload = _read_json_object(output)
    payload["row_count"] = 7
    _write_canonical_json(output, payload)

    with pytest.raises(ValueError, match="report_sha256"):
        module.load_vector_report(output)

    module.write_vector_report(output, report)
    output.write_text(json.dumps(_read_json_object(output)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical UTF-8 JSON"):
        module.load_vector_report(output)


def test_report_loader_rejects_raw_boolean_to_integer_hash_bypass(
    tmp_path: Path,
) -> None:
    module = _exact_vector()
    report = _run_small_vector_report(module, tmp_path / "work")
    payload = report.model_dump(mode="json")
    original_hash = payload["report_sha256"]
    payload["generation_integrity_verified"] = 1
    output = tmp_path / "report.json"
    _write_canonical_json(output, payload)

    assert payload["report_sha256"] == original_hash
    with pytest.raises(ValueError, match="raw canonical report payload"):
        module.load_vector_report(output)


def test_atomic_report_replace_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _exact_vector()
    report = module.run_benchmark(
        profile=_small_vector_profile(module),
        hardware_source=_sample_vector_hardware(),
        workspace=tmp_path / "work",
        hardware_collector=lambda: _sample_vector_hardware(),
        sampler_factory=_FakeProcessTreeSampler,
    )
    output = tmp_path / "report.json"
    output.write_bytes(b"existing\n")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"cannot replace {source.name} with {destination.name}")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace"):
        module.write_vector_report(output, report)

    assert output.read_bytes() == b"existing\n"
    assert list(tmp_path.glob(".report.json.*.tmp")) == []


@pytest.mark.parametrize(
    "arguments",
    [
        ["benchmark", "--config", "config.json", "--workspace", "work", "--output", "out.json"],
        [
            "benchmark",
            "--config",
            "config.json",
            "--hardware-facts",
            "facts.json",
            "--reference-hardware",
            "reference.json",
            "--workspace",
            "work",
            "--output",
            "out.json",
        ],
    ],
    ids=("missing-source", "mutually-exclusive-sources"),
)
def test_cli_hardware_source_errors_are_stable_english(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = _exact_vector().main(arguments)

    assert result == 2
    assert "error:" in capsys.readouterr().err


def test_cli_rejects_noncanonical_hardware_without_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _exact_vector()
    config = tmp_path / "phase0.json"
    facts_path = tmp_path / "facts.json"
    workspace = tmp_path / "work"
    output = tmp_path / "report.json"
    _write_vector_config(config, _small_vector_profile(module))
    facts_path.write_text(
        json.dumps(_sample_vector_hardware().model_dump(mode="json")),
        encoding="utf-8",
    )

    result = module.main(
        [
            "benchmark",
            "--config",
            str(config),
            "--hardware-facts",
            str(facts_path),
            "--workspace",
            str(workspace),
            "--output",
            str(output),
        ]
    )

    assert result == 1
    assert "Hardware facts file must be canonical UTF-8 JSON." in capsys.readouterr().err
    assert not workspace.exists()
    assert not output.exists()


def test_hardware_facts_loader_rejects_integer_coercion(tmp_path: Path) -> None:
    module = _exact_vector()
    payload = _sample_vector_hardware().model_dump(mode="json")
    payload["physical_cores"] = "4"
    path = tmp_path / "hardware-facts.json"
    _write_canonical_json(path, payload)

    with pytest.raises(
        ValueError, match="Hardware facts file does not match its validated canonical model"
    ):
        module._load_hardware_facts(path)


def test_reference_hardware_loader_rejects_record_binding_coercion(
    tmp_path: Path,
) -> None:
    module = _exact_vector()
    payload = _sample_reference_hardware().model_dump(mode="json")
    payload["physical_cores"] = "4"
    path = tmp_path / "reference-hardware.json"
    _write_canonical_json(path, payload)

    with pytest.raises(
        ValueError,
        match="Reference hardware file does not match its validated canonical model",
    ):
        module._load_reference_hardware(path)


def _run_small_vector_report(module: ModuleType, workspace: Path) -> object:
    return module.run_benchmark(
        profile=_small_vector_profile(module),
        hardware_source=_sample_vector_hardware(),
        workspace=workspace,
        hardware_collector=lambda: _sample_vector_hardware(),
        sampler_factory=_FakeProcessTreeSampler,
    )


def _run_small_vector_report_with_peak(
    module: ModuleType,
    workspace: Path,
    *,
    hardware_source: object,
    peak_bytes: int,
    measurement_valid: bool,
) -> object:
    from academic_chatbot.feasibility.process_tree import ProcessTreePeak

    class PeakSampler(_FakeProcessTreeSampler):
        def __init__(self, sample_interval_ms: int) -> None:
            self._result = ProcessTreePeak(
                metric="process_tree_sum_uss_bytes",
                peak_bytes=peak_bytes,
                sample_interval_ms=sample_interval_ms,
                sample_count=3,
                process_churn_count=0,
                access_error_count=0 if measurement_valid else 1,
                measurement_valid=measurement_valid,
            )

    return module.run_benchmark(
        profile=_small_vector_profile(module),
        hardware_source=hardware_source,
        workspace=workspace,
        hardware_collector=lambda: _sample_vector_hardware(),
        sampler_factory=PeakSampler,
    )


def _write_rehashed_report(path: Path, payload: dict[str, object]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "report_sha256"}
    payload["report_sha256"] = _canonical_sha256(unsigned)
    _write_canonical_json(path, payload)


def test_reference_hardware_cli_writes_bound_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from academic_chatbot.feasibility.hardware import write_reference_hardware

    module = _exact_vector()
    profile = _small_vector_profile(module)
    reference = _sample_reference_hardware()
    config = tmp_path / "phase0.json"
    reference_path = tmp_path / "reference-hardware.json"
    output = tmp_path / "vector-exact.json"
    _write_vector_config(config, profile)
    write_reference_hardware(reference_path, reference)
    monkeypatch.setattr(module, "collect_windows_hardware", _sample_vector_hardware)
    monkeypatch.setattr(module, "ProcessTreePeakSampler", _FakeProcessTreeSampler)

    result = module.main(
        [
            "benchmark",
            "--config",
            str(config),
            "--reference-hardware",
            str(reference_path),
            "--workspace",
            str(tmp_path / "work"),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    report = module.load_vector_report(output)
    assert report.measurement_status == "bound"
    assert report.hardware_source_kind == "reference_hardware"
    assert report.reference_hardware_record_sha256 == reference.record_sha256
    assert report.memory_gate_status == "passed"
    assert report.gate_eligible is False


def test_cli_writes_failed_correctness_report_before_returning_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from academic_chatbot.feasibility.hardware import write_hardware_facts

    module = _exact_vector()
    base_report = _run_small_vector_report(module, tmp_path / "measured-work")
    payload = base_report.model_dump(mode="json")
    payload["generation_integrity_verified"] = False
    payload["correctness_status"] = "failed"
    payload["gate_eligible"] = False
    payload["failure_reasons"] = [
        "Benchmark profile does not match the committed reference profile.",
        "Vector generation integrity verification failed.",
    ]
    unsigned = {
        key: value for key, value in payload.items() if key != "report_sha256"
    }
    payload["report_sha256"] = _canonical_sha256(unsigned)
    failed_report = module.VectorExactReport.model_validate(payload)
    config = tmp_path / "phase0.json"
    facts_path = tmp_path / "hardware-facts.json"
    output = tmp_path / "failed-correctness.json"
    _write_vector_config(config, _small_vector_profile(module))
    write_hardware_facts(facts_path, _sample_vector_hardware())
    monkeypatch.setattr(module, "run_benchmark", lambda **_: failed_report)

    result = module.main(
        [
            "benchmark",
            "--config",
            str(config),
            "--hardware-facts",
            str(facts_path),
            "--workspace",
            str(tmp_path / "unused-work"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "Exact-vector benchmark correctness verification failed.\n"
    assert module.load_vector_report(output).correctness_status == "failed"


def test_cli_writes_invalid_memory_measurement_before_returning_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from academic_chatbot.feasibility.hardware import write_hardware_facts

    module = _exact_vector()
    report = _run_small_vector_report_with_peak(
        module,
        tmp_path / "measured-work",
        hardware_source=_sample_vector_hardware(),
        peak_bytes=1_000_000,
        measurement_valid=False,
    )
    config = tmp_path / "phase0.json"
    facts_path = tmp_path / "hardware-facts.json"
    output = tmp_path / "invalid-memory-measurement.json"
    _write_vector_config(config, _small_vector_profile(module))
    write_hardware_facts(facts_path, _sample_vector_hardware())
    monkeypatch.setattr(module, "run_benchmark", lambda **_: report)

    result = module.main(
        [
            "benchmark",
            "--config",
            str(config),
            "--hardware-facts",
            str(facts_path),
            "--workspace",
            str(tmp_path / "unused-work"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "Exact-vector process-tree memory measurement is invalid.\n"
    assert module.load_vector_report(output).process_tree_peak.measurement_valid is False


def test_cli_bound_memory_gate_failure_is_a_completed_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from academic_chatbot.feasibility.hardware import write_reference_hardware

    module = _exact_vector()
    reference = _sample_reference_hardware()
    report = _run_small_vector_report_with_peak(
        module,
        tmp_path / "measured-work",
        hardware_source=reference,
        peak_bytes=12_884_901_888,
        measurement_valid=True,
    )
    config = tmp_path / "phase0.json"
    reference_path = tmp_path / "reference-hardware.json"
    output = tmp_path / "memory-gate-failed.json"
    _write_vector_config(config, _small_vector_profile(module))
    write_reference_hardware(reference_path, reference)
    monkeypatch.setattr(module, "run_benchmark", lambda **_: report)

    result = module.main(
        [
            "benchmark",
            "--config",
            str(config),
            "--reference-hardware",
            str(reference_path),
            "--workspace",
            str(tmp_path / "unused-work"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == ""
    assert captured.err == ""
    loaded = module.load_vector_report(output)
    assert loaded.memory_gate_status == "failed"
    assert loaded.process_tree_peak.measurement_valid is True


def test_reference_record_hash_tamper_is_rejected_without_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _exact_vector()
    config = tmp_path / "phase0.json"
    reference_path = tmp_path / "reference.json"
    workspace = tmp_path / "work"
    output = tmp_path / "report.json"
    _write_vector_config(config, _small_vector_profile(module))
    payload = _sample_reference_hardware().model_dump(mode="json")
    payload["cpu_model"] = "Tampered CPU"
    _write_canonical_json(reference_path, payload)

    result = module.main(
        [
            "benchmark",
            "--config",
            str(config),
            "--reference-hardware",
            str(reference_path),
            "--workspace",
            str(workspace),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "Reference hardware file is invalid.\n"
    assert not workspace.exists()
    assert not output.exists()


def test_cli_runtime_mismatch_is_sorted_and_leaves_no_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from academic_chatbot.feasibility.hardware import write_hardware_facts

    module = _exact_vector()
    expected = _sample_vector_hardware()
    actual = expected.model_copy(
        update={"storage_kind": "SATA SSD", "cpu_model": "Different CPU"}
    )
    config = tmp_path / "phase0.json"
    facts_path = tmp_path / "facts.json"
    workspace = tmp_path / "work"
    output = tmp_path / "report.json"
    _write_vector_config(config, _small_vector_profile(module))
    write_hardware_facts(facts_path, expected)
    monkeypatch.setattr(module, "collect_windows_hardware", lambda: actual)

    result = module.main(
        [
            "benchmark",
            "--config",
            str(config),
            "--hardware-facts",
            str(facts_path),
            "--workspace",
            str(workspace),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == (
        "Runtime hardware does not match expected fields: cpu_model, storage_kind.\n"
    )
    assert not workspace.exists()
    assert not output.exists()


def test_cli_collection_failure_is_stable_and_leaves_no_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from academic_chatbot.feasibility.hardware import write_hardware_facts

    module = _exact_vector()
    config = tmp_path / "phase0.json"
    facts_path = tmp_path / "facts.json"
    workspace = tmp_path / "work"
    output = tmp_path / "report.json"
    _write_vector_config(config, _small_vector_profile(module))
    write_hardware_facts(facts_path, _sample_vector_hardware())

    def fail_collection() -> object:
        raise RuntimeError("Runtime hardware collection failed.")

    monkeypatch.setattr(module, "collect_windows_hardware", fail_collection)

    result = module.main(
        [
            "benchmark",
            "--config",
            str(config),
            "--hardware-facts",
            str(facts_path),
            "--workspace",
            str(workspace),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "Runtime hardware collection failed.\n"
    assert not workspace.exists()
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hardware_source_kind", "reference_hardware"),
        ("reference_hardware_record_sha256", "5" * 64),
        ("exact_16_gib_reference_verified", True),
        ("memory_gate_status", "passed"),
        ("gate_eligible", True),
        ("runtime_hardware_match", False),
        ("runtime_hardware_mismatch_fields", ["cpu_model"]),
        ("generation_integrity_verified", False),
        ("deterministic_tie_break_verified", False),
        ("correctness_status", "failed"),
    ],
)
def test_rehashed_report_rejects_cross_field_status_tampering(
    tmp_path: Path, field: str, value: object
) -> None:
    module = _exact_vector()
    report = _run_small_vector_report(module, tmp_path / "work")
    payload = report.model_dump(mode="json")
    payload[field] = value
    output = tmp_path / "report.json"
    _write_rehashed_report(output, payload)

    with pytest.raises((ValidationError, ValueError)):
        module.load_vector_report(output)


def test_rehashed_report_rejects_profile_shape_and_count_inconsistency(
    tmp_path: Path,
) -> None:
    module = _exact_vector()
    report = _run_small_vector_report(module, tmp_path / "work")
    payload = report.model_dump(mode="json")
    profile = dict(payload["profile"])
    profile["row_count"] = 9
    payload["profile"] = profile
    payload["profile_sha256"] = _canonical_sha256(profile)
    output = tmp_path / "profile-shape.json"
    _write_rehashed_report(output, payload)

    with pytest.raises(ValueError, match="report dimensions"):
        module.load_vector_report(output)

    payload = report.model_dump(mode="json")
    payload["query_count"] = 3
    output = tmp_path / "query-count.json"
    _write_rehashed_report(output, payload)
    with pytest.raises(ValueError, match="query settings"):
        module.load_vector_report(output)


def test_rehashed_report_rejects_profile_hash_inconsistency(tmp_path: Path) -> None:
    module = _exact_vector()
    report = _run_small_vector_report(module, tmp_path / "work")
    payload = report.model_dump(mode="json")
    payload["profile_sha256"] = "0" * 64
    output = tmp_path / "report.json"
    _write_rehashed_report(output, payload)

    with pytest.raises(ValueError, match="profile_sha256"):
        module.load_vector_report(output)


def test_rehashed_bound_report_rejects_memory_pass_for_invalid_measurement(
    tmp_path: Path,
) -> None:
    module = _exact_vector()
    report = _run_small_vector_report_with_peak(
        module,
        tmp_path / "work",
        hardware_source=_sample_reference_hardware(),
        peak_bytes=1_000_000,
        measurement_valid=False,
    )
    payload = report.model_dump(mode="json")
    payload["memory_gate_status"] = "passed"
    output = tmp_path / "invalid-measurement-forged-pass.json"
    _write_rehashed_report(output, payload)

    with pytest.raises(ValueError, match="derived report status"):
        module.load_vector_report(output)


def test_rehashed_report_rejects_reordered_derived_failure_reasons(
    tmp_path: Path,
) -> None:
    module = _exact_vector()
    report = _run_small_vector_report_with_peak(
        module,
        tmp_path / "work",
        hardware_source=_sample_vector_hardware(),
        peak_bytes=1_000_000,
        measurement_valid=False,
    )
    payload = report.model_dump(mode="json")
    reasons = list(payload["failure_reasons"])
    assert len(reasons) >= 2
    payload["failure_reasons"] = list(reversed(reasons))
    output = tmp_path / "reordered-failure-reasons.json"
    _write_rehashed_report(output, payload)

    with pytest.raises(ValueError, match="derived report status"):
        module.load_vector_report(output)


@pytest.mark.parametrize(
    ("source_kind", "exact_reference", "expected_reference_ram"),
    [
        ("hardware_facts", False, None),
        ("reference_hardware", True, 17_179_869_184),
        ("reference_hardware", False, 17_179_869_183),
    ],
)
def test_report_validator_recomputes_status_with_unambiguous_ram_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    exact_reference: bool,
    expected_reference_ram: int | None,
) -> None:
    module = _exact_vector()
    hardware_source = (
        _sample_vector_hardware()
        if source_kind == "hardware_facts"
        else _sample_reference_hardware()
    )
    report = _run_small_vector_report_with_peak(
        module,
        tmp_path / source_kind,
        hardware_source=hardware_source,
        peak_bytes=1_000_000,
        measurement_valid=True,
    )
    payload = report.model_dump(mode="json")
    if source_kind == "reference_hardware" and not exact_reference:
        payload["exact_16_gib_reference_verified"] = False
        payload["memory_gate_status"] = "not_evaluated_wrong_reference_ram"
        payload["gate_eligible"] = False
        payload["failure_reasons"] = [
            "Benchmark profile does not match the committed reference profile.",
            "Reference hardware RAM does not equal the exact 16 GiB target.",
        ]
        unsigned = {
            key: value for key, value in payload.items() if key != "report_sha256"
        }
        payload["report_sha256"] = _canonical_sha256(unsigned)

    observed_reference_ram: list[int | None] = []
    real_evaluate = module.evaluate_report_status

    def observe_status(**kwargs: object) -> object:
        observed_reference_ram.append(kwargs["reference_ram_bytes"])
        return real_evaluate(**kwargs)

    monkeypatch.setattr(module, "evaluate_report_status", observe_status)

    module.VectorExactReport.model_validate(payload)

    assert observed_reference_ram == [expected_reference_ram]


def test_successful_benchmark_lifecycle_keeps_sampler_around_all_vector_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _exact_vector()
    events: list[str] = []
    sampler_active = False
    generated_seeds: list[int] = []
    search_blocks: list[int] = []
    real_attest = module.attest_runtime_hardware
    real_generate = module._generate_normalized_matrix
    real_search = module.ExactVectorStore.search
    real_oracle = module._oracle_top_k_rows

    class LifecycleSampler(_FakeProcessTreeSampler):
        def __enter__(self) -> "LifecycleSampler":
            nonlocal sampler_active
            events.append("sampler_enter")
            sampler_active = True
            return self

        def __exit__(self, *args: object) -> None:
            nonlocal sampler_active
            events.append("sampler_exit")
            sampler_active = False

    def collect() -> object:
        events.append("collect")
        return _sample_vector_hardware()

    def attest(expected: object, actual: object) -> object:
        events.append("attest")
        return real_attest(expected, actual)

    def generate(row_count: int, dimension: int, *, seed: int) -> np.ndarray:
        assert sampler_active
        events.append("allocate")
        generated_seeds.append(seed)
        return real_generate(row_count, dimension, seed=seed)

    def search(store: object, query: np.ndarray, *, limit: int, block_rows: int) -> object:
        assert sampler_active
        assert query.ndim == 1
        search_blocks.append(block_rows)
        return real_search(store, query, limit=limit, block_rows=block_rows)

    def oracle(rows: np.ndarray, query: np.ndarray, *, top_k: int) -> object:
        assert sampler_active
        assert query.ndim == 1
        events.append("oracle")
        return real_oracle(rows, query, top_k=top_k)

    monkeypatch.setattr(module, "attest_runtime_hardware", attest)
    monkeypatch.setattr(module, "_generate_normalized_matrix", generate)
    monkeypatch.setattr(module.ExactVectorStore, "search", search)
    monkeypatch.setattr(module, "_oracle_top_k_rows", oracle)
    profile = _small_vector_profile(module)

    module.run_benchmark(
        profile=profile,
        hardware_source=_sample_vector_hardware(),
        workspace=tmp_path / "work",
        hardware_collector=collect,
        sampler_factory=LifecycleSampler,
    )

    assert events[:3] == ["collect", "attest", "sampler_enter"]
    assert generated_seeds == [profile.row_seed, profile.query_seed]
    assert search_blocks[: profile.warmup_query_count + profile.query_count] == [
        profile.block_rows
    ] * (profile.warmup_query_count + profile.query_count)
    assert search_blocks[-1] == max(1, profile.block_rows // 2)
    assert events.count("oracle") == profile.query_count
    assert events[-1] == "sampler_exit"


def test_store_mapping_is_closed_before_cli_report_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from academic_chatbot.feasibility.hardware import write_hardware_facts

    module = _exact_vector()
    config = tmp_path / "phase0.json"
    facts_path = tmp_path / "facts.json"
    workspace = tmp_path / "work"
    output = tmp_path / "report.json"
    _write_vector_config(config, _small_vector_profile(module))
    write_hardware_facts(facts_path, _sample_vector_hardware())
    monkeypatch.setattr(module, "collect_windows_hardware", _sample_vector_hardware)
    monkeypatch.setattr(module, "ProcessTreePeakSampler", _FakeProcessTreeSampler)
    real_writer = module.write_vector_report

    def assert_closed_then_write(path: Path, report: object) -> None:
        generation = next((workspace / "generations").iterdir())
        vectors = generation / "vectors.npy"
        moved = generation / "vectors.closed.npy"
        os.replace(vectors, moved)
        os.replace(moved, vectors)
        real_writer(path, report)

    monkeypatch.setattr(module, "write_vector_report", assert_closed_then_write)

    result = module.main(
        [
            "benchmark",
            "--config",
            str(config),
            "--hardware-facts",
            str(facts_path),
            "--workspace",
            str(workspace),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert output.is_file()


def test_default_profile_digest_and_committed_config_are_exact() -> None:
    module = _exact_vector()
    profile = module.VectorBenchmarkProfile()

    assert module.compute_vector_profile_sha256(profile) == (
        "b5e5541a57335109b123ca8c262e65b2c21b77ee0cc4af0a32e99ee0857b4d64"
    )
    config_path = Path("benchmarks/config/phase0.json")
    assert module.load_vector_profile(config_path) == profile


@pytest.mark.parametrize("field", ["p95_threshold_ms", "top_k_agreement_threshold"])
def test_unapproved_thresholds_cannot_be_injected(field: str) -> None:
    payload = _exact_vector().VectorBenchmarkProfile().model_dump(mode="json")
    payload[field] = 0.5

    with pytest.raises(ValidationError):
        _exact_vector().VectorBenchmarkProfile.model_validate(payload)


def test_matrix_generator_matches_explicit_pcg64_reference() -> None:
    module = _exact_vector()
    expected = np.random.Generator(np.random.PCG64(101)).standard_normal(
        (3, 5)
    ).astype(np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)

    actual = module._generate_normalized_matrix(3, 5, seed=101)

    assert np.array_equal(actual, expected)


def test_runtime_hash_covers_complete_fresh_canonical_payload() -> None:
    from academic_chatbot.feasibility.hardware import canonical_sha256

    module = _exact_vector()
    expected = _sample_vector_hardware()
    actual = expected.model_copy(
        update={
            "collected_at": "2026-07-13T00:00:01Z",
            "collection_diagnostics": ("new diagnostic",),
        }
    )

    attestation = module.attest_runtime_hardware(expected, actual)

    assert attestation.runtime_hardware_facts_sha256 == canonical_sha256(
        actual.model_dump(mode="json")
    )
    assert attestation.runtime_hardware_facts_sha256 != canonical_sha256(
        expected.model_dump(mode="json")
    )


@pytest.mark.parametrize(
    ("generation_integrity", "tie_break", "runtime_match"),
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_bound_status_fails_closed_on_integrity_tie_or_runtime_failure(
    generation_integrity: bool, tie_break: bool, runtime_match: bool
) -> None:
    module = _exact_vector()
    status = module.evaluate_report_status(
        profile=module.VectorBenchmarkProfile(),
        measurement_status="bound",
        reference_ram_bytes=17_179_869_184,
        generation_integrity_verified=generation_integrity,
        deterministic_tie_break_verified=tie_break,
        runtime_hardware_match=runtime_match,
        measurement_valid=True,
        peak_bytes=1_000_000,
    )

    assert status.gate_eligible is False
    assert status.correctness_status == (
        "passed" if generation_integrity and tie_break else "failed"
    )


def test_second_run_refuses_published_generation_and_writes_no_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from academic_chatbot.feasibility.hardware import write_hardware_facts

    module = _exact_vector()
    config = tmp_path / "phase0.json"
    facts_path = tmp_path / "facts.json"
    workspace = tmp_path / "work"
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    _write_vector_config(config, _small_vector_profile(module))
    write_hardware_facts(facts_path, _sample_vector_hardware())
    monkeypatch.setattr(module, "collect_windows_hardware", _sample_vector_hardware)
    monkeypatch.setattr(module, "ProcessTreePeakSampler", _FakeProcessTreeSampler)
    base_arguments = [
        "benchmark",
        "--config",
        str(config),
        "--hardware-facts",
        str(facts_path),
        "--workspace",
        str(workspace),
        "--output",
    ]

    assert module.main([*base_arguments, str(first_output)]) == 0
    assert module.main([*base_arguments, str(second_output)]) == 1

    assert first_output.is_file()
    assert not second_output.exists()


def test_forged_wrong_ram_reference_is_revalidated_before_collection(
    tmp_path: Path,
) -> None:
    module = _exact_vector()
    reference = _sample_reference_hardware()
    payload = dict(reference.__dict__)
    payload["ram_bytes"] = 8_000_000_000
    forged = type(reference).model_construct(**payload)
    calls: list[str] = []

    def collect() -> object:
        calls.append("collect")
        return _sample_vector_hardware()

    with pytest.raises(ValidationError, match="16 GiB"):
        module.run_benchmark(
            profile=_small_vector_profile(module),
            hardware_source=forged,
            workspace=tmp_path / "work",
            hardware_collector=collect,
            sampler_factory=_FakeProcessTreeSampler,
        )

    assert calls == []
    assert not (tmp_path / "work").exists()


def test_report_writer_is_exact_canonical_json_without_workspace_path(
    tmp_path: Path,
) -> None:
    module = _exact_vector()
    workspace = tmp_path / "private-workspace"
    report = _run_small_vector_report(module, workspace)
    output = tmp_path / "report.json"

    module.write_vector_report(output, report)

    payload = _read_json_object(output)
    assert output.read_bytes() == _canonical_json_bytes(payload) + b"\n"
    assert str(workspace) not in output.read_text(encoding="utf-8")
