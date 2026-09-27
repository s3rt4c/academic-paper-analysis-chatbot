from pathlib import Path

import pytest

from academic_chatbot.evidence.resolver import EvidenceReadResolver, EvidenceResolutionError
from tests.fixtures.evidence_bundle.database import database


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM vector_generation_publications",
        "UPDATE vector_generations SET source_snapshot_sha256 = '" + "a" * 64 + "'",
    ],
)
def test_resolver_rejects_stale_or_corrupt_vector_generation(tmp_path: Path, sql) -> None:
    db = database(tmp_path)
    request = db.semantic()
    db.corrupt(sql)
    with pytest.raises(EvidenceResolutionError):
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)


def test_resolver_handles_valid_empty_vector_manifest_without_model(
    tmp_path: Path, monkeypatch
) -> None:
    from academic_chatbot.evidence import resolver

    db = database(tmp_path, empty=True)
    request = db.semantic()

    def forbidden(*args, **kwargs):
        pytest.fail("empty generation must not open vector payload")

    monkeypatch.setattr(resolver.ExactVectorStore, "open", forbidden)
    result = EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
    assert result.groups == ()


def test_resolver_reads_profile_and_sources_in_one_snapshot(tmp_path: Path, monkeypatch) -> None:
    from academic_chatbot.evidence import resolver

    db = database(tmp_path)
    request = db.semantic()
    opened = []
    original = resolver.open_read_only_connection

    def capture(*args, **kwargs):
        connection = original(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(resolver, "open_read_only_connection", capture)
    result = EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
    assert len(opened) == 1
    assert len(result.groups) == len(request.candidates) > 0


@pytest.mark.parametrize(
    ("sql", "code"),
    [
        (
            "UPDATE embedding_profiles SET canonical_profile_sha256 = '" + "a" * 64 + "'",
            "PROFILE_MISMATCH",
        ),
        ("UPDATE embedding_profiles SET dimension = 12", "PROFILE_MISMATCH"),
        ("UPDATE embedding_profiles SET span_policy_id = 'wrong'", "PROFILE_MISMATCH"),
        (
            "UPDATE embedding_profiles SET artifact_manifest_sha256 = '" + "z" * 64 + "'",
            "PROFILE_MISMATCH",
        ),
        ("UPDATE vector_generations SET state = 'STALE'", "VECTOR_INTEGRITY"),
        (
            "UPDATE vector_generations SET vector_store_manifest_sha256 = '" + "a" * 64 + "'",
            "VECTOR_INTEGRITY",
        ),
        ("DELETE FROM vector_generation_spans", "VECTOR_INTEGRITY"),
        (
            "UPDATE embedding_spans SET coverage_status = 'EXCLUDED_UNEMBEDDABLE'",
            "VECTOR_INTEGRITY",
        ),
        (
            "UPDATE vector_generation_sources SET eligible_native_chunk_count = 999",
            "VECTOR_NOT_CURRENT",
        ),
        ("UPDATE vector_generations SET artifact_relative_dir = 'indexes/wrong'", "UNSAFE_PATH"),
    ],
)
def test_semantic_corruption_fails_exact_code(tmp_path, sql, code):
    db = database(tmp_path)
    request = db.semantic()
    db.corrupt(sql)
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
    assert error.value.error.code == code


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("start_offset", 1, "RANGE_MISMATCH"),
        ("expected_text_sha256", "a" * 64, "TEXT_DIGEST_MISMATCH"),
        ("expected_anchor_ids", ("wrong",), "ANCHOR_MISMATCH"),
        ("embedding_span_id", "missing", "REFERENCE_NOT_FOUND"),
    ],
)
def test_semantic_assertions(tmp_path, field, value, code):
    db = database(tmp_path)
    request = db.semantic()
    ref = request.candidates[0]
    ref = ref.model_copy(update={"semantic": ref.semantic.model_copy(update={field: value})})
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(
            request.model_copy(update={"candidates": (ref,)})
        )
    assert error.value.error.code == code


def test_exact_requested_vector_and_profile_are_not_substituted(tmp_path):
    db = database(tmp_path)
    request = db.semantic()
    for field, code in (
        ("vector_generation_id", "VECTOR_NOT_CURRENT"),
        ("embedding_profile_id", "PROFILE_MISMATCH"),
    ):
        changed = request.model_copy(
            update={"origin": request.origin.model_copy(update={field: "missing"})}
        )
        with pytest.raises(EvidenceResolutionError) as error:
            EvidenceReadResolver(data_root=db.paths.data_root).resolve(changed)
        assert error.value.error.code == code


@pytest.mark.parametrize("small_span", [False, True])
def test_semantic_context_and_dual_channel_separation(tmp_path, small_span):
    from academic_chatbot.evidence.models import CandidateOrigin

    db = database(tmp_path)
    request = db.semantic(maximum_words=8 if small_span else 510)
    semantic = request.candidates[0]
    result = EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
    assert (result.groups[0].context is not None) == small_span
    if small_span:
        assert result.groups[0].context.citable is False
        assert result.groups[0].ranges[0].text != result.groups[0].context.text
    both = semantic.model_copy(update={"lexical": db.request.candidates[0].lexical})
    hybrid = request.model_copy(
        update={
            "origin": CandidateOrigin(
                mode="hybrid",
                fusion_profile_id="rrf-v1",
                embedding_profile_id=request.origin.embedding_profile_id,
                vector_generation_id=request.origin.vector_generation_id,
            ),
            "candidates": (both,),
        }
    )
    group = EvidenceReadResolver(data_root=db.paths.data_root).resolve(hybrid).groups[0]
    assert len(group.ranges) == 2 and group.context is None
    assert tuple(r.contributions[0].channel for r in group.ranges) == ("lexical", "semantic")


def test_lexical_only_hybrid_still_validates_vector_origin(tmp_path):
    from academic_chatbot.evidence.models import CandidateOrigin

    db = database(tmp_path)
    request = db.semantic()
    hybrid = db.request.model_copy(
        update={
            "origin": CandidateOrigin(
                mode="hybrid",
                fusion_profile_id="rrf-v1",
                embedding_profile_id=request.origin.embedding_profile_id,
                vector_generation_id=request.origin.vector_generation_id,
            )
        }
    )
    assert EvidenceReadResolver(data_root=db.paths.data_root).resolve(hybrid).groups
    db.execute("DELETE FROM vector_generation_publications")
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(hybrid)
    assert error.value.error.code == "VECTOR_UNAVAILABLE"


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("damage", ["missing", "bytes", "reparse"])
def test_artifact_damage_and_unsafe_leaf(tmp_path, monkeypatch, empty, damage):
    import stat
    from types import SimpleNamespace

    db = database(tmp_path, empty=empty)
    request = db.semantic()
    relative = db.rows("SELECT artifact_relative_dir FROM vector_generations")[0][0]
    path = (
        db.paths.project_root / relative / ("empty-generation.json" if empty else "manifest.json")
    )
    if damage == "missing":
        path.unlink()
    elif damage == "bytes":
        path.write_bytes(b"{}")
    else:
        original = Path.lstat

        def lstat(candidate, *args, **kwargs):
            info = original(candidate, *args, **kwargs)
            if candidate == path:
                return SimpleNamespace(
                    st_mode=info.st_mode, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT
                )
            return info

        monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
    assert (
        error.value.error.code
        == {"missing": "VECTOR_UNAVAILABLE", "bytes": "VECTOR_INTEGRITY", "reparse": "UNSAFE_PATH"}[
            damage
        ]
    )


@pytest.mark.parametrize(
    "stage", ["_read_registered_profile", "_current_snapshot", "_generation_sources", "_mapping"]
)
def test_semantic_failure_closes_snapshot(tmp_path, monkeypatch, stage):
    import sqlite3

    from academic_chatbot.evidence import resolver

    db = database(tmp_path)
    request = db.semantic()
    opened = []
    original = resolver.open_read_only_connection

    def capture(*args, **kwargs):
        conn = original(*args, **kwargs)
        opened.append(conn)
        return conn

    def fail(*args, **kwargs):
        raise EvidenceResolutionError(resolver.m.EvidenceErrorCode.VECTOR_INTEGRITY)

    monkeypatch.setattr(resolver, "open_read_only_connection", capture)
    monkeypatch.setattr(resolver, stage, fail)
    with pytest.raises(EvidenceResolutionError):
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")


@pytest.mark.parametrize("corrupt", [False, True])
def test_memmap_closed_before_return_or_error(tmp_path, monkeypatch, corrupt):
    from academic_chatbot.evidence import resolver

    db = database(tmp_path)
    request = db.semantic()
    if corrupt:
        db.corrupt("UPDATE vector_generations SET vector_store_manifest_sha256 = ?", ("a" * 64,))
    opened = []
    closed = []
    original = resolver.ExactVectorStore.open
    close = resolver.ExactVectorStore.close

    def capture(*args, **kwargs):
        limits = kwargs["limits"]
        assert (
            limits.max_manifest_bytes,
            limits.max_metadata_bytes,
            limits.max_vectors_file_bytes,
            limits.max_rows,
            limits.expected_dimension,
        ) == (65536, 16 * 1024 * 1024, 128 * 1024 * 1024, 100000, 384)
        store = original(*args, **kwargs)
        opened.append(store)
        return store

    def closing(store):
        closed.append(store)
        close(store)

    monkeypatch.setattr(resolver.ExactVectorStore, "open", capture)
    monkeypatch.setattr(resolver.ExactVectorStore, "close", closing)
    if corrupt:
        with pytest.raises(EvidenceResolutionError):
            EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
    else:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
    assert len(opened) == 1 and opened == closed


def test_empty_generation_rejects_claimed_semantic_span(tmp_path):
    from academic_chatbot.evidence.models import EvidenceCandidateRef, SemanticRangeRef
    from academic_chatbot.retrieval.hybrid_models import ChunkCandidateIdentity

    db = database(tmp_path, empty=True)
    request = db.semantic()
    page_id = db.rows("SELECT page_id FROM pages")[0][0]
    claimed = EvidenceCandidateRef(
        parent=ChunkCandidateIdentity(
            project_id=request.scope.project_id,
            document_generation_id=request.scope.document_generation_id,
            page_id=page_id,
            chunk_id="missing",
        ),
        reported_rank=1,
        semantic=SemanticRangeRef(
            embedding_span_id="missing",
            start_offset=0,
            end_offset=1,
            expected_text_sha256="a" * 64,
            expected_anchor_ids=("missing",),
        ),
    )
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(
            request.model_copy(update={"candidates": (claimed,)})
        )
    assert error.value.error.code == "REFERENCE_NOT_FOUND"


def test_other_source_publication_invalidates_project_snapshot(tmp_path):
    db = database(tmp_path, older=True)
    request = db.semantic()
    # Keep only selected-source candidate assertions; vector freshness still covers the project.
    request = request.model_copy(
        update={
            "candidates": tuple(
                ref
                for ref in request.candidates
                if ref.parent.document_generation_id == request.scope.document_generation_id
            )
        }
    )
    assert EvidenceReadResolver(data_root=db.paths.data_root).resolve(request).groups
    db.execute(
        "DELETE FROM generation_publications WHERE file_version_id <> ?",
        (request.scope.file_version_id,),
    )
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
    assert error.value.error.code == "VECTOR_NOT_CURRENT"


@pytest.mark.parametrize(
    "column",
    [
        "eligible_native_chunks",
        "excluded_unembeddable_spans",
        "needs_ocr_pages",
        "indexed_documents",
        "unindexed_documents",
    ],
)
def test_vector_coverage_counts_are_authoritative(tmp_path, column):
    db = database(tmp_path)
    request = db.semantic()
    db.corrupt(f"UPDATE vector_generations SET {column} = {column} + 1")
    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceReadResolver(data_root=db.paths.data_root).resolve(request)
    assert error.value.error.code == "VECTOR_INTEGRITY"
