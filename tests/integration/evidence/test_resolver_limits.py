from pathlib import Path

import pytest

from academic_chatbot.evidence.resolver import EvidenceReadResolver, EvidenceResolutionError
from tests.fixtures.evidence_bundle.database import database


def test_page_text_limit(tmp_path: Path) -> None:
    db = database(tmp_path)
    db.execute("UPDATE pages SET canonical_text = ?", ("x" * (1024 * 1024 + 1),))
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
    assert error.value.error.code == "RESOURCE_LIMIT"


@pytest.mark.parametrize("extra", [0, 1])
def test_page_count_boundary(tmp_path, extra):
    db = database(tmp_path)
    conn = db.connect()
    try:
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT INTO pages (page_id, document_generation_id, page_number) VALUES (?, ?, ?)",
            (
                (f"legacy-{i}", db.request.scope.document_generation_id, i + 1)
                for i in range(1, 1000 + extra)
            ),
        )
        conn.commit()
    finally:
        conn.close()
    request = db.request.model_copy(update={"candidates": ()})
    if extra:
        with pytest.raises(EvidenceResolutionError) as error:
            EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
        assert error.value.error.code == "RESOURCE_LIMIT"
    else:
        result = EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
        assert len(result.coverage.pages) == 1000
        assert result.coverage.status == "unknown_metadata"
        assert result.coverage.pages[1].physical_page_index is None


@pytest.mark.parametrize("extra", [0, 1])
def test_page_bytes_admitted_before_materialization(tmp_path, monkeypatch, extra):
    from academic_chatbot.evidence import resolver

    db = database(tmp_path)
    # Multibyte content pins byte admission, not Python character length.
    text = "é" * (512 * 1024) + "x" * extra
    db.execute("UPDATE pages SET canonical_text = ?", (text,))
    reads = []

    def admitted(*args):
        reads.append(True)
        raise RuntimeError("admitted page")

    monkeypatch.setattr(resolver, "_read_page", admitted)
    if extra:
        with pytest.raises(EvidenceResolutionError) as error:
            EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
        assert error.value.error.code == "RESOURCE_LIMIT"
        assert reads == []
    else:
        with pytest.raises(RuntimeError, match="admitted page"):
            EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
        assert reads == [True]


@pytest.mark.parametrize("extra", [0, 1])
def test_distinct_page_bytes_are_summed_before_any_materialization(tmp_path, monkeypatch, extra):
    from academic_chatbot.evidence import resolver

    db = database(tmp_path)
    ref = db.request.candidates[0]
    db.execute("UPDATE pages SET canonical_text = ?", ("x" * 1024 * 1024,))
    refs = [ref]
    for i in range(1, 16 + extra):
        page_id = f"page-{i}"
        db.execute(
            "INSERT INTO pages (page_id, document_generation_id, page_number, canonical_text) "
            "VALUES (?, ?, ?, ?)",
            (
                page_id,
                db.request.scope.document_generation_id,
                i + 1,
                "x" * (1 if i == 16 else 1024 * 1024),
            ),
        )
        refs.append(
            ref.model_copy(update={"parent": ref.parent.model_copy(update={"page_id": page_id})})
        )
    reads = []

    def admitted(*args):
        reads.append(True)
        raise RuntimeError("admitted pages")

    monkeypatch.setattr(resolver, "_read_page", admitted)
    request = db.request.model_copy(update={"candidates": tuple(refs)})
    if extra:
        with pytest.raises(EvidenceResolutionError) as error:
            EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
        assert error.value.error.code == "RESOURCE_LIMIT" and reads == []
    else:
        with pytest.raises(RuntimeError, match="admitted pages"):
            EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
        assert reads == [True]


@pytest.mark.parametrize("extra", [0, 1])
def test_anchor_admission_boundary(tmp_path, extra):
    db = database(tmp_path)
    first = db.rows("SELECT * FROM page_anchors ORDER BY char_start")[0]
    existing = db.rows("SELECT count(*) FROM page_anchors")[0][0]
    for i in range(120 + extra - existing):
        values = list(first)
        values[0] = f"extra-anchor-{i}"
        values[1] = f"extra-evidence-{i}"
        values[3] = 0
        values[4] = i + 6
        db.execute(
            "INSERT INTO page_anchors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(values)
        )
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
    # At the cap reconstruction detects the deliberate duplicate anchor;
    # cap+one is rejected before any reconstruction.
    assert error.value.error.code == ("RESOURCE_LIMIT" if extra else "STORAGE_INTEGRITY")


@pytest.mark.parametrize("extra", [0, 1])
def test_profile_json_byte_boundary_before_materialization(tmp_path, monkeypatch, extra):
    from academic_chatbot.evidence import resolver

    db = database(tmp_path)
    request = db.semantic()
    db.corrupt("UPDATE embedding_profiles SET canonical_profile_json = ?", ("x" * (65536 + extra),))
    loads = []

    def parse(*args):
        loads.append(True)
        raise ValueError("synthetic malformed admitted profile")

    monkeypatch.setattr(resolver.json, "loads", parse)
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
    assert error.value.error.code == ("RESOURCE_LIMIT" if extra else "PROFILE_MISMATCH")
    assert len(loads) == (0 if extra else 1)


@pytest.mark.parametrize("extra", [0, 1])
def test_source_snapshot_row_boundary_before_helper_fetch(tmp_path, monkeypatch, extra):
    from academic_chatbot.evidence import resolver

    db = database(tmp_path)
    request = db.semantic()
    conn = db.connect()
    try:
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT INTO file_versions "
            "VALUES (?, 'paper-one', ?, 'originals/synthetic.pdf', '2026')",
            ((f"version-{i}", f"{i:064x}") for i in range(1, 10000 + extra)),
        )
        conn.executemany(
            "INSERT INTO document_generations VALUES (?, ?, 'synthetic', '2026')",
            ((f"generation-{i}", f"version-{i}") for i in range(1, 10000 + extra)),
        )
        conn.executemany(
            "INSERT INTO generation_publications VALUES (?, ?)",
            ((f"version-{i}", f"generation-{i}") for i in range(1, 10000 + extra)),
        )
        conn.commit()
    finally:
        conn.close()
    calls = []

    def admitted(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("source rows admitted")

    monkeypatch.setattr(resolver, "_current_snapshot", admitted)
    if extra:
        with pytest.raises(EvidenceResolutionError) as error:
            EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
        assert error.value.error.code == "RESOURCE_LIMIT" and not calls
    else:
        with pytest.raises(RuntimeError, match="source rows admitted"):
            EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
        assert calls == [True]


@pytest.mark.parametrize("extra", [0, 1])
def test_mapping_rows_admitted_before_helper_fetch(tmp_path, monkeypatch, extra):
    from academic_chatbot.evidence import resolver

    db = database(tmp_path)
    request = db.semantic()
    db.corrupt("DELETE FROM vector_generation_spans")
    conn = db.connect()
    try:
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT INTO vector_generation_spans VALUES (?, ?, ?)",
            ((request.origin.vector_generation_id, i, f"span-{i}") for i in range(100000 + extra)),
        )
        conn.commit()
    finally:
        conn.close()
    calls = []

    def admitted(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("mapping rows admitted")

    monkeypatch.setattr(resolver, "_mapping", admitted)
    if extra:
        with pytest.raises(EvidenceResolutionError) as error:
            EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
        assert error.value.error.code == "RESOURCE_LIMIT" and not calls
    else:
        with pytest.raises(RuntimeError, match="mapping rows admitted"):
            EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
        assert calls == [True]


def test_empty_manifest_read_is_bounded(tmp_path):
    db = database(tmp_path, empty=True)
    request = db.semantic()
    relative = db.rows("SELECT artifact_relative_dir FROM vector_generations")[0][0]
    (db.paths.project_root / relative / "empty-generation.json").write_bytes(b"x" * 65537)
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
    assert error.value.error.code == "RESOURCE_LIMIT"


@pytest.mark.parametrize("table", ["chunks", "page_anchors"])
def test_corrupt_stored_text_cannot_bypass_page_byte_admission(tmp_path, table):
    db = database(tmp_path)
    column = "chunk_text" if table == "chunks" else "anchor_text"
    db.execute(f"UPDATE {table} SET {column} = ?", ("x" * (1024 * 1024 + 1),))
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(db.request)
    assert error.value.error.code == "RESOURCE_LIMIT"
