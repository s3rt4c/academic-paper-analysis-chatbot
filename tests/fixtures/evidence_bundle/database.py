"""Synthetic published sources; setup uses the existing ingestion pipeline."""

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from academic_chatbot.documents.import_service import DocumentImportService
from academic_chatbot.documents.native_pdf import NativePdfParser
from academic_chatbot.domain.library import Project
from academic_chatbot.embeddings.repository import EmbeddingRepository
from academic_chatbot.evidence.models import (
    BundleSourceScope,
    CandidateOrigin,
    EvidenceBundleRequest,
    EvidenceCandidateRef,
    LexicalRangeRef,
    PreviewBudget,
    SemanticRangeRef,
)
from academic_chatbot.evidence.references import (
    candidate_ref_from_hybrid,
    candidate_ref_from_lexical,
    candidate_ref_from_semantic,
)
from academic_chatbot.library.service import LibraryService
from academic_chatbot.retrieval.hybrid_models import ChunkCandidateIdentity
from academic_chatbot.retrieval.hybrid_service import HybridRetrievalService
from academic_chatbot.retrieval.semantic import SemanticRetrievalService
from academic_chatbot.retrieval.service import RetrievalService
from academic_chatbot.storage.paths import ProjectPaths
from tests.fixtures.evidence_bundle.pdfs import write_pdf
from tests.integration.embeddings.test_vector_publication import _builder, _Embedder, _profile


@dataclass
class Database:
    paths: ProjectPaths
    request: EvidenceBundleRequest

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.paths.database_path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def execute(self, sql: str, args: tuple = ()) -> None:
        connection = self.connect()
        try:
            connection.execute(sql, args)
        finally:
            connection.close()

    def rows(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        connection = self.connect()
        try:
            return connection.execute(sql, args).fetchall()
        finally:
            connection.close()

    def corrupt(self, sql: str, args: tuple = ()) -> None:
        """Remove test DB guards explicitly, then simulate damaged persisted state."""
        for row in self.rows("SELECT name FROM sqlite_master WHERE type = 'trigger'"):
            name = row[0].replace('"', '""')
            self.execute(f'DROP TRIGGER "{name}"')
        self.execute(sql, args)

    def logical_state(self) -> dict:
        result = {}
        for row in self.rows("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"):
            name = row[0].replace('"', '""')
            result[row[0]] = sorted(
                (tuple(r) for r in self.rows(f'SELECT * FROM "{name}"')), key=repr
            )
        return result

    def retrieved(self, *, mode="lexical", maximum_words=510, query="Alpha"):
        """Run actual retrieval and pure reference conversion after synthetic setup."""
        project = Project(project_id=self.request.scope.project_id, display_name="Synthetic")
        if mode == "lexical":
            hits = RetrievalService(data_root=self.paths.data_root).search(project, query).hits
            origin = CandidateOrigin(mode="lexical")
            candidates = tuple(candidate_ref_from_lexical(hit) for hit in hits)
        else:
            prepared = self.semantic(maximum_words=maximum_words)
            profile = _profile()
            semantic = SemanticRetrievalService(
                data_root=self.paths.data_root,
                profile=profile,
                embedder=_SyntheticQueryEmbedder(profile),
            )
            if mode == "semantic":
                hits = semantic.search(project, query, limit=100).hits
                candidates = tuple(candidate_ref_from_semantic(hit) for hit in hits)
                origin = prepared.origin
            elif mode == "hybrid":
                hits = (
                    HybridRetrievalService(
                        data_root=self.paths.data_root,
                        semantic_service=semantic,
                    )
                    .search(project, query, limit=100)
                    .hits
                )
                candidates = tuple(candidate_ref_from_hybrid(hit) for hit in hits)
                origin = CandidateOrigin(
                    mode="hybrid",
                    fusion_profile_id="rrf-v1",
                    embedding_profile_id=prepared.origin.embedding_profile_id,
                    vector_generation_id=prepared.origin.vector_generation_id,
                )
            else:
                raise ValueError("unsupported synthetic retrieval mode")
        # Select only the explicitly requested generation; older versions remain current.
        candidates = tuple(
            c
            for c in candidates
            if c.parent.document_generation_id == self.request.scope.document_generation_id
        )
        return EvidenceBundleRequest(
            scope=self.request.scope,
            origin=origin,
            candidates=candidates,
            preview_budget=PreviewBudget(),
        )

    def semantic(self, *, maximum_words: int = 510) -> EvidenceBundleRequest:
        profile = _profile()
        repository = EmbeddingRepository(self.paths)
        repository.register_profile(profile, artifact_manifest_sha256="e" * 64)
        result = _builder(
            repository, self.paths, _Embedder(profile), tokenizer_maximum_words=maximum_words
        ).build(project_id=self.request.scope.project_id)
        candidates = []
        for span in self.rows(
            "SELECT * FROM embedding_spans WHERE coverage_status = 'EMBEDDABLE' "
            "ORDER BY start_offset"
        ):
            page = self.rows(
                "SELECT canonical_text FROM pages WHERE page_id = ?", (span["page_id"],)
            )[0]
            text = page[0][span["start_offset"] : span["end_offset"]]
            anchors = self.rows(
                "SELECT page_anchor_id FROM page_anchors WHERE page_id = ? "
                "AND char_start >= ? AND char_end <= ? "
                "ORDER BY char_start, char_end, page_anchor_id",
                (span["page_id"], span["start_offset"], span["end_offset"]),
            )
            candidates.append(
                EvidenceCandidateRef(
                    parent=ChunkCandidateIdentity(
                        project_id=self.request.scope.project_id,
                        document_generation_id=span["document_generation_id"],
                        page_id=span["page_id"],
                        chunk_id=span["chunk_id"],
                    ),
                    reported_rank=len(candidates) + 1,
                    semantic=SemanticRangeRef(
                        embedding_span_id=span["embedding_span_id"],
                        start_offset=span["start_offset"],
                        end_offset=span["end_offset"],
                        expected_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        expected_anchor_ids=tuple(a[0] for a in anchors),
                    ),
                )
            )
        return EvidenceBundleRequest(
            scope=self.request.scope,
            preview_budget=PreviewBudget(),
            origin=CandidateOrigin(
                mode="semantic",
                embedding_profile_id=profile.embedding_profile_id,
                vector_generation_id=result.generation.vector_generation_id,
            ),
            candidates=tuple(candidates),
        )


class _SyntheticQueryEmbedder(_Embedder):
    """Reuse deterministic document vectors for query-time fixture setup only."""

    def embed_queries(self, texts):
        return self.embed_documents(texts)


def database(
    tmp_path: Path,
    *,
    empty: bool = False,
    older: bool = False,
    text: str | None = None,
    project_id: str = "project-one",
) -> Database:
    service = LibraryService(data_root=tmp_path / "data", max_pdf_bytes=1_000_000)
    service.create_project(display_name="Synthetic", project_id=project_id)
    service.create_paper(project_id=project_id, paper_id="paper-one")
    paths = ProjectPaths.create(tmp_path / "data", project_id=project_id)
    version = service.admit_pdf(
        project_id=project_id,
        paper_id="paper-one",
        source_path=write_pdf(tmp_path / "source.pdf", empty=empty, text=text),
    )
    parsed = NativePdfParser(paths).parse(version)
    published = DocumentImportService(service.repository_for_project_id(project_id)).publish(parsed)
    scope = BundleSourceScope(
        project_id=project_id,
        paper_id="paper-one",
        file_version_id=version.file_version_id,
        document_generation_id=published.document_generation_id,
    )
    db = Database(
        paths,
        EvidenceBundleRequest(
            scope=scope,
            origin=CandidateOrigin(mode="lexical"),
            candidates=(),
            preview_budget=PreviewBudget(),
        ),
    )
    candidates = []
    for chunk in db.rows("SELECT * FROM chunks ORDER BY page_id, ordinal"):
        anchors = db.rows(
            "SELECT page_anchor_id FROM page_anchors WHERE page_id = ? "
            "AND char_start >= ? AND char_end <= ? ORDER BY char_start, char_end, page_anchor_id",
            (chunk["page_id"], chunk["start_offset"], chunk["end_offset"]),
        )
        candidates.append(
            EvidenceCandidateRef(
                parent=ChunkCandidateIdentity(
                    project_id=scope.project_id,
                    document_generation_id=scope.document_generation_id,
                    page_id=chunk["page_id"],
                    chunk_id=chunk["chunk_id"],
                ),
                reported_rank=len(candidates) + 1,
                lexical=LexicalRangeRef(
                    start_offset=chunk["start_offset"],
                    end_offset=chunk["end_offset"],
                    expected_text_sha256=hashlib.sha256(chunk["chunk_text"].encode()).hexdigest(),
                    expected_anchor_ids=tuple(a[0] for a in anchors),
                ),
            )
        )
    db.request = db.request.model_copy(update={"candidates": tuple(candidates)})
    if older:
        newer = service.admit_pdf(
            project_id=project_id,
            paper_id="paper-one",
            source_path=write_pdf(tmp_path / "newer.pdf", suffix="newer"),
        )
        DocumentImportService(service.repository_for_project_id(project_id)).publish(
            NativePdfParser(paths).parse(newer)
        )
    return db
