from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from academic_chatbot.domain.library import Project
from academic_chatbot.retrieval.semantic import (
    SemanticArtifactIntegrityError,
    SemanticQueryError,
    SemanticRetrievalService,
    _artifact_path,
    _load_registered_profile_with_manifest_hash,
    _require_ordinary_artifact_files,
)
from academic_chatbot.retrieval.semantic_position import PositionalAcquisitionSelection
from academic_chatbot.retrieval.semantic_query import CANONICAL_CONCERN_TEXT
from academic_chatbot.storage.paths import ProjectPaths
from tests.integration.embeddings.test_vector_publication import _profile, _project


@dataclass
class _RecordingQueryEmbedder:
    profile: object
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def embed_queries(self, texts: tuple[str, ...]) -> np.ndarray:
        self.calls.append(texts)
        vector = np.zeros((1, self.profile.dimension), dtype=np.float32)  # type: ignore[attr-defined]
        vector[0, 0] = 1.0
        return vector


def test_query_embedding_receives_one_exact_selected_representation() -> None:
    profile = _profile()
    embedder = _RecordingQueryEmbedder(profile)
    service = SemanticRetrievalService(data_root=Path("unused"), profile=profile, embedder=embedder)

    vector = service._embed_query(CANONICAL_CONCERN_TEXT)

    assert vector.shape == (profile.dimension,)
    assert embedder.calls == [(CANONICAL_CONCERN_TEXT,)]


def test_positional_acquisition_rejects_unknown_concern_before_embedding() -> None:
    profile = _profile()
    embedder = _RecordingQueryEmbedder(profile)
    service = SemanticRetrievalService(data_root=Path("unused"), profile=profile, embedder=embedder)

    with pytest.raises(SemanticQueryError):
        service.acquire_positional(
            Project(project_id="project-one", display_name="Research"),
            "user query",
            positional_selection=PositionalAcquisitionSelection(
                concern_id="unsupported-concern-v1"
            ),
        )

    assert embedder.calls == []


def test_semantic_artifact_path_rejects_nonsemantic_project_locations(tmp_path: Path) -> None:
    paths = ProjectPaths.create(tmp_path / "data", project_id="project-one")

    with pytest.raises(SemanticArtifactIntegrityError, match="semantic index root"):
        _artifact_path(
            paths,
            "derivatives/untrusted-vector",
            profile_id="ep-sha256-" + "a" * 64,
            source_snapshot_sha256="b" * 64,
        )


def test_registered_profile_loading_retains_its_manifest_binding(tmp_path: Path) -> None:
    repository, paths = _project(tmp_path, native_chunk=False)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)

    loaded, manifest_hash = _load_registered_profile_with_manifest_hash(
        paths, profile.embedding_profile_id
    )

    assert loaded == profile
    assert manifest_hash == "e" * 64


def test_semantic_artifact_rejects_a_symlinked_vector_file(tmp_path: Path) -> None:
    artifact = tmp_path / "generation"
    artifact.mkdir()
    target = tmp_path / "outside-vectors.npy"
    target.write_bytes(b"not-a-vector")
    try:
        (artifact / "vectors.npy").symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable on this Windows host: {error}")
    (artifact / "manifest.json").write_text("{}", encoding="utf-8")
    (artifact / "vectors.meta.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SemanticArtifactIntegrityError, match="symlink or reparse"):
        _require_ordinary_artifact_files(artifact, empty=False)


def test_semantic_scoped_starvation_exact_global_ties_and_legacy(tmp_path):
    from tests.fixtures.retrieval.scoped import crowded, semantic_oracle

    fixture = crowded(tmp_path, vectors=True)
    semantic = fixture.semantic
    before = semantic.search(fixture.project, "Alpha", limit=50)
    assert len(before.hits) == 50
    assert all(h.paper_id == "distractor" for h in before.hits)
    assert hasattr(semantic, "search_scoped"), "semantic scoped acquisition absent"
    result = semantic.search_scoped(fixture.request("semantic"))
    expected = semantic_oracle(fixture)
    assert len(expected) == 2
    assert [h.embedding_span_id for h in result.hits] == [r[1] for r in expected]
    assert [h.rank for h in result.hits] == [1, 2]
    assert [h.raw_semantic_score for h in result.hits] == [0.0, 0.0]
    assert semantic.search(fixture.project, "Alpha", limit=50) == before


@pytest.mark.parametrize("corruption", ["dangling", "page", "chunk", "profile", "duplicate"])
def test_scoped_semantic_rejects_ineligible_mapping_corruption(tmp_path, corruption):
    from academic_chatbot.retrieval.scoped_models import ScopedRetrievalError
    from tests.fixtures.retrieval.scoped import crowded

    fixture = crowded(tmp_path, vectors=True)
    semantic = fixture.semantic
    assert hasattr(semantic, "search_scoped"), "semantic scoped acquisition absent"
    request = fixture.request("semantic")
    selected = fixture.db.rows(
        "SELECT chunk_id,page_id FROM chunks WHERE document_generation_id=?",
        (fixture.scope.document_generation_id,),
    )[0]
    foreign = fixture.db.rows(
        "SELECT embedding_span_id FROM embedding_spans WHERE document_generation_id!=?",
        (fixture.scope.document_generation_id,),
    )[0][0]
    if corruption == "dangling":
        fixture.db.corrupt(
            "UPDATE embedding_spans SET chunk_id='absent' WHERE embedding_span_id=?", (foreign,)
        )
    elif corruption == "page":
        fixture.db.corrupt(
            "UPDATE embedding_spans SET page_id=? WHERE embedding_span_id=?", (selected[1], foreign)
        )
    elif corruption == "chunk":
        fixture.db.corrupt(
            "UPDATE embedding_spans SET chunk_id=? WHERE embedding_span_id=?",
            (selected[0], foreign),
        )
    elif corruption == "profile":
        fixture.db.corrupt(
            "UPDATE embedding_spans SET embedding_profile_id='foreign' WHERE embedding_span_id=?",
            (foreign,),
        )
    else:
        fixture.db.corrupt("CREATE TABLE damaged_mapping AS SELECT * FROM vector_generation_spans")
        fixture.db.execute("DROP TABLE vector_generation_spans")
        fixture.db.execute("ALTER TABLE damaged_mapping RENAME TO vector_generation_spans")
        fixture.db.corrupt(
            """UPDATE vector_generation_spans SET embedding_span_id=?
               WHERE vector_row=(SELECT max(vector_row) FROM vector_generation_spans)""",
            (foreign,),
        )
    with pytest.raises(ScopedRetrievalError) as caught:
        semantic.search_scoped(request)
    assert caught.value.code == "VECTOR_INTEGRITY"


def test_scoped_semantic_fresh_identity_and_missing_generation(tmp_path):
    from academic_chatbot.retrieval.scoped_models import ScopedRetrievalError
    from tests.fixtures.retrieval.scoped import crowded

    fixture = crowded(tmp_path, vectors=True)
    semantic = fixture.semantic
    assert hasattr(semantic, "current_identity"), "fresh semantic identity absent"
    request = fixture.request("semantic")
    assert semantic.current_identity(project_id="project-one") == request.semantic_identity
    fixture.db.execute("DELETE FROM vector_generation_publications")
    with pytest.raises(ScopedRetrievalError) as caught:
        semantic.search_scoped(request)
    assert caught.value.code == "VECTOR_UNAVAILABLE"
