from __future__ import annotations

# ruff: noqa: E501
import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from academic_chatbot.db.connection import connect_project_database
from academic_chatbot.domain.library import Project
from academic_chatbot.embeddings.models import EmbeddingProfile, canonical_json_bytes
from academic_chatbot.retrieval.semantic import (
    SemanticArtifactIntegrityError,
    SemanticIndexStaleError,
    SemanticIndexUnavailableError,
    SemanticQueryTooLongError,
    SemanticRetrievalIntegrityError,
    SemanticRetrievalService,
)
from tests.integration.embeddings.test_vector_publication import (
    _builder,
    _Embedder,
    _profile,
    _project,
)


@dataclass
class _QueryEmbedder:
    profile: EmbeddingProfile
    vector: np.ndarray
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def embed_queries(self, texts: tuple[str, ...]) -> np.ndarray:
        self.calls.append(texts)
        return self.vector[np.newaxis, :].astype(np.float32, copy=True)


def _active_service(
    tmp_path: Path, *, duplicate_physical_pdf: bool = False
) -> tuple[SemanticRetrievalService, _QueryEmbedder, object, object]:
    repository, paths = _project(tmp_path, native_chunk=True)
    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        if duplicate_physical_pdf:
            _insert_duplicate_pdf_lineage(connection)
        connection.execute(
            "UPDATE pages SET parser_profile_sha256 = ?, canonical_text_sha256 = ?, page_width_points = 612.0, page_height_points = 792.0, source_page_rotation_degrees = 0 WHERE page_id = 'page-one'",
            ("e" * 64, hashlib.sha256(b"alpha beta gamma").hexdigest()),
        )
        for start, end, word in ((0, 5, "alpha"), (6, 10, "beta"), (11, 16, "gamma")):
            anchor_sha256 = hashlib.sha256(word.encode("utf-8")).hexdigest()
            box = {
                "char_start": start,
                "char_end": end,
                "x0": 0.0,
                "top": 0.0,
                "x1": 1.0,
                "bottom": 1.0,
            }
            boxes_sha256 = hashlib.sha256(canonical_json_bytes([box])).hexdigest()
            evidence_id = (
                "ev-sha256-"
                + hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "file_version_id": "file-one",
                            "pdf_sha256": "a" * 64,
                            "parser_profile_sha256": "e" * 64,
                            "physical_page_index": 0,
                            "char_start": start,
                            "char_end": end,
                            "anchor_text_sha256": anchor_sha256,
                            "boxes_sha256": boxes_sha256,
                        }
                    )
                ).hexdigest()
            )
            connection.execute(
                "UPDATE page_anchors SET evidence_id = ?, anchor_text_sha256 = ?, boxes_sha256 = ? WHERE page_id = 'page-one' AND char_start = ?",
                (evidence_id, anchor_sha256, boxes_sha256, start),
            )
    finally:
        connection.close()
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
    document_embedder = _Embedder(profile)
    built = _builder(repository, paths, document_embedder).build(project_id="project-one")
    vector = document_embedder.embed_documents(("alpha beta gamma",))[0]
    query_embedder = _QueryEmbedder(profile, vector)
    return (
        SemanticRetrievalService(
            data_root=paths.data_root, profile=profile, embedder=query_embedder
        ),
        query_embedder,
        repository,
        built,
    )


def _insert_duplicate_pdf_lineage(connection: sqlite3.Connection) -> None:
    connection.execute(
        "INSERT INTO papers VALUES ('paper-two', 'project-one', '2026-09-02T00:00:00Z')"
    )
    connection.execute(
        "INSERT INTO file_versions VALUES ('file-two', 'paper-two', ?, 'originals/a.pdf', '2026-09-02T00:00:00Z')",
        ("a" * 64,),
    )
    connection.execute(
        "INSERT INTO document_generations VALUES ('generation-two', 'file-two', 'native-v1', '2026-09-02T00:00:00Z')"
    )
    connection.execute(
        "INSERT INTO pages (page_id, document_generation_id, page_number, physical_page_index, canonical_text, canonical_text_sha256, parser_profile_sha256, page_width_points, page_height_points, source_page_rotation_degrees, needs_ocr) VALUES ('page-two', 'generation-two', 1, 0, 'alpha beta gamma', ?, ?, 612.0, 792.0, 0, 0)",
        (hashlib.sha256(b"alpha beta gamma").hexdigest(), "e" * 64),
    )
    connection.execute("INSERT INTO generation_publications VALUES ('file-two', 'generation-two')")
    connection.execute(
        "INSERT INTO chunks VALUES ('chunk-two', 'generation-two', 'page-two', 0, 0, 16, 'alpha beta gamma', 3, 'lexical-chunk-v1')"
    )
    for ordinal, (start, end, word) in enumerate(
        ((0, 5, "alpha"), (6, 10, "beta"), (11, 16, "gamma"))
    ):
        anchor_sha256 = hashlib.sha256(word.encode("utf-8")).hexdigest()
        box = {
            "char_start": start,
            "char_end": end,
            "x0": 0.0,
            "top": 0.0,
            "x1": 1.0,
            "bottom": 1.0,
        }
        boxes_sha256 = hashlib.sha256(canonical_json_bytes([box])).hexdigest()
        evidence_id = (
            "ev-sha256-"
            + hashlib.sha256(
                canonical_json_bytes(
                    {
                        "file_version_id": "file-two",
                        "pdf_sha256": "a" * 64,
                        "parser_profile_sha256": "e" * 64,
                        "physical_page_index": 0,
                        "char_start": start,
                        "char_end": end,
                        "anchor_text_sha256": anchor_sha256,
                        "boxes_sha256": boxes_sha256,
                    }
                )
            ).hexdigest()
        )
        connection.execute(
            "INSERT INTO page_anchors VALUES (?, ?, 'page-two', ?, ?, ?, ?, ?, 0.0, 0.0, 1.0, 1.0)",
            (f"anchor-two-{ordinal}", evidence_id, start, end, word, anchor_sha256, boxes_sha256),
        )


def _project_value() -> Project:
    return Project(project_id="project-one", display_name="Semantic test")


def test_search_returns_current_exact_evidence_and_raw_cosine(tmp_path: Path) -> None:
    service, query_embedder, _, built = _active_service(tmp_path)

    results = service.search(_project_value(), "meaningful query", limit=1)

    assert query_embedder.calls == [("meaningful query",)]
    assert results.project_id == "project-one"
    assert len(results.hits) == 1
    hit = results.hits[0]
    assert hit.rank == 1
    assert hit.raw_semantic_score == pytest.approx(1.0, abs=2e-3)
    assert hit.vector_generation_id == built.generation.vector_generation_id
    assert hit.embedding_profile_id == query_embedder.profile.embedding_profile_id
    assert hit.embedding_span_text == "alpha beta gamma"
    assert (hit.start_offset, hit.end_offset) == (0, 16)
    assert hit.chunk_id == "chunk-one"
    assert [anchor.anchor_text for anchor in hit.anchors] == ["alpha", "beta", "gamma"]


def test_search_returns_empty_for_a_valid_empty_generation(tmp_path: Path) -> None:
    repository, paths = _project(tmp_path, native_chunk=False)
    profile = _profile()
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
    built = _builder(repository, paths, _Embedder(profile)).build(project_id="project-one")
    query_embedder = _QueryEmbedder(profile, np.ones(profile.dimension, dtype=np.float32))

    results = SemanticRetrievalService(
        data_root=paths.data_root, profile=profile, embedder=query_embedder
    ).search(_project_value(), "query")

    assert built.empty
    assert results.hits == ()
    assert query_embedder.calls == []


def test_search_rejects_no_publication_and_stale_publication(tmp_path: Path) -> None:
    service, _, repository, built = _active_service(tmp_path)
    paths = repository._paths  # type: ignore[attr-defined]
    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        connection.execute("DELETE FROM vector_generation_publications")
    finally:
        connection.close()
    with pytest.raises(SemanticIndexUnavailableError):
        service.search(_project_value(), "query")

    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        connection.execute(
            "INSERT INTO vector_generation_publications VALUES ('project-one', ?, ?)",
            (service.profile.embedding_profile_id, built.generation.vector_generation_id),
        )
        connection.execute(
            "INSERT INTO document_generations VALUES ('generation-two', 'file-one', 'native-v2', '2026-09-02T00:00:00Z')"
        )
        connection.execute(
            "INSERT INTO pages (page_id, document_generation_id, page_number, canonical_text, canonical_text_sha256, needs_ocr) VALUES ('page-two', 'generation-two', 1, '', ?, 1)",
            ("f" * 64,),
        )
        connection.execute(
            "UPDATE generation_publications SET document_generation_id = 'generation-two' WHERE file_version_id = 'file-one'"
        )
    finally:
        connection.close()
    with pytest.raises(SemanticIndexStaleError):
        service.search(_project_value(), "query")


def test_search_rejects_tampered_vector_row_mapping_without_mutating_database(
    tmp_path: Path,
) -> None:
    service, _, repository, built = _active_service(tmp_path)
    paths = repository._paths  # type: ignore[attr-defined]
    connection = connect_project_database(paths.database_path, data_root=paths.data_root)
    try:
        connection.execute("DROP TRIGGER vector_generation_spans_immutable_update")
        connection.execute(
            "UPDATE vector_generation_spans SET vector_row = 1 WHERE vector_generation_id = ?",
            (built.generation.vector_generation_id,),
        )
    finally:
        connection.close()
    before = paths.database_path.read_bytes()

    with pytest.raises(SemanticRetrievalIntegrityError, match="row mapping"):
        service.search(_project_value(), "query")

    assert paths.database_path.read_bytes() == before


def test_search_refuses_another_projects_paths(tmp_path: Path) -> None:
    service, _, _, _ = _active_service(tmp_path)

    with pytest.raises(SemanticIndexUnavailableError):
        service.search(Project(project_id="project-two", display_name="Other"), "query")


def test_same_physical_pdf_retains_distinct_paper_and_file_version_hits(tmp_path: Path) -> None:
    service, _, _, _ = _active_service(tmp_path, duplicate_physical_pdf=True)

    hits = service.search(_project_value(), "query", limit=2).hits

    assert [(hit.paper_id, hit.file_version_id) for hit in hits] == [
        ("paper-one", "file-one"),
        ("paper-two", "file-two"),
    ]


def test_search_rejects_an_over_budget_query_without_truncation(tmp_path: Path) -> None:
    service, query_embedder, _, _ = _active_service(tmp_path)

    def _too_long(texts: tuple[str, ...]) -> np.ndarray:
        from academic_chatbot.embeddings.tokenizer import EmbeddingInputTooLongError

        raise EmbeddingInputTooLongError("query exceeds embedding profile token budget")

    query_embedder.embed_queries = _too_long  # type: ignore[method-assign]
    with pytest.raises(SemanticQueryTooLongError, match="token budget"):
        service.search(_project_value(), "untruncated query")


def test_search_fails_closed_for_a_corrupt_active_vector_artifact(tmp_path: Path) -> None:
    service, _, repository, built = _active_service(tmp_path)
    paths = repository._paths  # type: ignore[attr-defined]
    (paths.resolve_relative(built.generation.artifact_relative_dir) / "vectors.npy").write_bytes(
        b"corrupt"
    )

    with pytest.raises(SemanticArtifactIntegrityError, match="missing or corrupt"):
        service.search(_project_value(), "query")
