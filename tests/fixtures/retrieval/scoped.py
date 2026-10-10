"""Crowded, rights-safe persisted sources and independent subset oracles."""

from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

import numpy as np
from reportlab.pdfgen import canvas

from academic_chatbot.documents.import_service import DocumentImportService
from academic_chatbot.documents.native_pdf import NativePdfParser
from academic_chatbot.domain.library import Project
from academic_chatbot.embeddings.repository import EmbeddingRepository
from academic_chatbot.evidence.models import BundleSourceScope
from academic_chatbot.library.service import LibraryService
from academic_chatbot.retrieval.scoped_models import ScopedRetrievalRequest, policy_for
from academic_chatbot.retrieval.semantic import SemanticRetrievalService
from academic_chatbot.storage.paths import ProjectPaths
from tests.fixtures.evidence_bundle.database import Database
from tests.integration.embeddings.test_vector_publication import _builder, _profile


class SparseEmbedder:
    def __init__(self, profile):
        self.profile = profile

    def embed_documents(self, texts):
        rows = np.zeros((len(texts), self.profile.dimension), dtype=np.float32)
        for index, text in enumerate(texts):
            rows[index, 1 if "selectedmarker" in text else 0] = 1
        return rows

    def embed_queries(self, texts):
        rows = np.zeros((len(texts), self.profile.dimension), dtype=np.float32)
        rows[:, 0] = 1
        return rows


def pdf_pages(path, texts):
    pdf = canvas.Canvas(str(path), invariant=1)
    for text in texts:
        for index, line in enumerate(wrap(text, width=70)):
            pdf.drawString(40, 760 - index * 20, line)
        pdf.showPage()
    pdf.save()
    return path


@dataclass
class Crowded:
    db: Database
    scope: BundleSourceScope
    semantic: SemanticRetrievalService | None = None

    @property
    def project(self):
        return Project(project_id=self.scope.project_id, display_name="Synthetic")

    def request(self, mode="lexical", limit=10):
        return ScopedRetrievalRequest(
            scope=self.scope,
            query="Alpha",
            mode=mode,
            limit=limit,
            policy=policy_for(mode),
            semantic_identity=self.semantic.current_identity(project_id=self.scope.project_id)
            if mode != "lexical"
            else None,
        )


def crowded(tmp_path: Path, *, vectors=False):
    root = tmp_path / "data"
    library = LibraryService(data_root=root, max_pdf_bytes=1_000_000)
    library.create_project(display_name="Synthetic", project_id="project-one")
    paths = ProjectPaths.create(root, project_id="project-one")
    scope = None
    for paper, texts in (
        ("selected", ("Alpha selectedmarker " + "padding " * 90,) * 2),
        ("distractor", ("Alpha Alpha Alpha foreignmarker",) * 64),
    ):
        library.create_paper(project_id="project-one", paper_id=paper)
        version = library.admit_pdf(
            project_id="project-one",
            paper_id=paper,
            source_path=pdf_pages(tmp_path / (paper + ".pdf"), texts),
        )
        generation = DocumentImportService(
            library.repository_for_project_id("project-one")
        ).publish(NativePdfParser(paths).parse(version))
        if paper == "selected":
            scope = BundleSourceScope(
                project_id="project-one",
                paper_id=paper,
                file_version_id=version.file_version_id,
                document_generation_id=generation.document_generation_id,
            )
    assert scope is not None
    fixture = Crowded(Database(paths, None), scope)
    if vectors:
        profile = _profile()
        repository = EmbeddingRepository(paths)
        repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
        _builder(repository, paths, SparseEmbedder(profile)).build(project_id="project-one")
        fixture.semantic = SemanticRetrievalService(
            data_root=root, profile=profile, embedder=SparseEmbedder(profile)
        )
    return fixture


def lexical_oracle(fixture, *, query_expression='"Alpha"'):
    # Independent same-corpus SQL: no production scoped SQL/helper and no LIMIT.
    rows = fixture.db.rows(
        """SELECT c.chunk_id, bm25(chunk_fts), p.paper_id, fv.file_version_id,
                  page.physical_page_index, c.ordinal
           FROM chunk_fts JOIN chunks c ON c.chunk_id=chunk_fts.chunk_id
           JOIN pages page ON page.page_id=c.page_id
           JOIN document_generations dg ON dg.document_generation_id=c.document_generation_id
           JOIN file_versions fv ON fv.file_version_id=dg.file_version_id
           JOIN papers p ON p.paper_id=fv.paper_id
           JOIN generation_publications gp ON gp.file_version_id=fv.file_version_id
             AND gp.document_generation_id=dg.document_generation_id
           WHERE chunk_fts MATCH ? AND p.project_id=? AND p.paper_id=?
             AND fv.file_version_id=? AND dg.document_generation_id=?""",
        (
            query_expression,
            fixture.scope.project_id,
            fixture.scope.paper_id,
            fixture.scope.file_version_id,
            fixture.scope.document_generation_id,
        ),
    )
    return sorted(rows, key=lambda r: (r[1], r[2], r[3], r[4], r[5], r[0]))


def semantic_oracle(fixture):
    # Fixture vectors are exactly representable unit basis vectors. Persisted mapping
    # and a separate structural subset query determine GLOBAL rows, never search output.
    return fixture.db.rows(
        """SELECT m.vector_row, m.embedding_span_id, s.chunk_id, s.page_id
           FROM vector_generation_spans m JOIN embedding_spans s
             ON s.embedding_span_id=m.embedding_span_id
           JOIN document_generations dg ON dg.document_generation_id=s.document_generation_id
           JOIN file_versions fv ON fv.file_version_id=dg.file_version_id
           JOIN papers p ON p.paper_id=fv.paper_id
           WHERE p.project_id=? AND p.paper_id=? AND fv.file_version_id=?
             AND dg.document_generation_id=? ORDER BY m.vector_row""",
        tuple(
            getattr(fixture.scope, key)
            for key in ("project_id", "paper_id", "file_version_id", "document_generation_id")
        ),
    )
