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
