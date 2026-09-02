from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from academic_chatbot.documents.import_service import DocumentImportService
from academic_chatbot.documents.native_pdf import NativePdfParser
from academic_chatbot.domain.library import Project
from academic_chatbot.embeddings.repository import EmbeddingRepository
from academic_chatbot.library.service import LibraryService
from academic_chatbot.retrieval.semantic import SemanticRetrievalService
from academic_chatbot.storage.paths import ProjectPaths
from tests.integration.embeddings.test_vector_publication import _builder, _Embedder, _profile

_PDF = Path(__file__).parents[1] / "fixtures" / "pdfs" / "native_anchor.pdf"


@dataclass
class _QueryEmbedder:
    profile: object
    vector: np.ndarray

    def embed_queries(self, texts: tuple[str, ...]) -> np.ndarray:
        return self.vector[np.newaxis, :].astype(np.float32, copy=True)


def test_native_pdf_to_semantic_hit_preserves_active_evidence(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    library = LibraryService(data_root=data_root, max_pdf_bytes=1_000_000)
    project = library.create_project(display_name="Native", project_id="project-one")
    library.create_paper(project_id=project.project_id, paper_id="paper-one")
    file_version = library.admit_pdf(
        project_id=project.project_id, paper_id="paper-one", source_path=_PDF
    )
    paths = ProjectPaths.create(data_root, project_id=project.project_id)
    parsed = NativePdfParser(paths).parse(file_version)
    DocumentImportService(library.repository_for_project_id(project.project_id)).publish(parsed)
    profile = _profile()
    repository = EmbeddingRepository(paths)
    repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
    document_embedder = _Embedder(profile)
    built = _builder(repository, paths, document_embedder).build(project_id=project.project_id)
    query_vector = document_embedder.embed_documents((document_embedder.calls[0][0],))[0]
    results = SemanticRetrievalService(
        data_root=data_root,
        profile=profile,
        embedder=_QueryEmbedder(profile, query_vector),
    ).search(Project(project_id=project.project_id, display_name="Native"), "semantic native")

    assert built.generation.vector_generation_id == results.hits[0].vector_generation_id
    assert results.hits[0].file_version_id == file_version.file_version_id
    assert results.hits[0].paper_id == "paper-one"
    assert results.hits[0].anchors
