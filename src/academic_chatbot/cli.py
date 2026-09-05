"""Minimal local CLI for the approved Task 1--5 vertical slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from academic_chatbot.documents.import_service import DocumentImportService
from academic_chatbot.documents.native_pdf import NativePdfParser
from academic_chatbot.domain.library import Project
from academic_chatbot.embeddings.artifacts import EmbeddingArtifactError, load_verified_artifacts
from academic_chatbot.embeddings.embedder import OfflineEmbedder, OfflineEmbedderError
from academic_chatbot.embeddings.models import canonical_json_bytes
from academic_chatbot.embeddings.profile import approved_bge_small_en_v15_profile
from academic_chatbot.embeddings.repository import EmbeddingPersistenceError, EmbeddingRepository
from academic_chatbot.embeddings.vector_build import ProjectVectorBuilder, VectorBuildError
from academic_chatbot.library.repository import ProjectRepository
from academic_chatbot.library.service import LibraryService
from academic_chatbot.retrieval.fts import RetrievalQueryError
from academic_chatbot.retrieval.hybrid_service import (
    HybridRetrievalIntegrityError,
    HybridRetrievalService,
)
from academic_chatbot.retrieval.semantic import (
    SemanticArtifactIntegrityError,
    SemanticIndexStaleError,
    SemanticIndexUnavailableError,
    SemanticProfileError,
    SemanticQueryError,
    SemanticRetrievalIntegrityError,
    SemanticRetrievalService,
)
from academic_chatbot.retrieval.service import (
    RetrievalIntegrityError,
    RetrievalService,
    RetrievalStorageError,
)
from academic_chatbot.storage.paths import ProjectPaths


class SemanticIndexBuildError(ValueError):
    """Stable user-facing failure for semantic-index build orchestration."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run local commands and return a clean non-zero code for ordinary errors."""

    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 2
    try:
        if arguments.max_pdf_bytes <= 0:
            raise ValueError("--max-pdf-bytes must be positive")
        library = LibraryService(
            data_root=Path(arguments.data_root), max_pdf_bytes=arguments.max_pdf_bytes
        )
        payload = _dispatch(library, arguments)
    except (
        RetrievalQueryError,
        RetrievalStorageError,
        RetrievalIntegrityError,
        ValueError,
        HybridRetrievalIntegrityError,
        SemanticArtifactIntegrityError,
        SemanticIndexStaleError,
        SemanticIndexUnavailableError,
        SemanticProfileError,
        SemanticQueryError,
        SemanticRetrievalIntegrityError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="academic_chatbot")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--max-pdf-bytes", type=int, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    project = commands.add_parser("project").add_subparsers(dest="project_command", required=True)
    create_project = project.add_parser("create")
    create_project.add_argument("--project-id", required=True)
    create_project.add_argument("--display-name", required=True)
    paper = commands.add_parser("paper").add_subparsers(dest="paper_command", required=True)
    create_paper = paper.add_parser("create")
    create_paper.add_argument("--project-id", required=True)
    create_paper.add_argument("--paper-id", required=True)
    imported = commands.add_parser("import-pdf")
    imported.add_argument("--project-id", required=True)
    imported.add_argument("--paper-id", required=True)
    imported.add_argument("--source", required=True)
    semantic_index = commands.add_parser("semantic-index")
    semantic_build = semantic_index.add_subparsers(dest="semantic_index_command", required=True)
    build = semantic_build.add_parser("build")
    build.add_argument("--project-id", required=True)
    build.add_argument("--embedding-profile-id", required=True)
    build.add_argument("--model-root", required=True)
    search = commands.add_parser("search")
    search.add_argument("--project-id", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--mode", choices=("lexical", "semantic", "hybrid"), default="lexical")
    search.add_argument("--embedding-profile-id")
    search.add_argument("--model-root")
    return parser


def _dispatch(library: LibraryService, arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.command == "project":
        return library.create_project(
            display_name=arguments.display_name, project_id=arguments.project_id
        ).model_dump(mode="json")
    if arguments.command == "paper":
        return library.create_paper(
            project_id=arguments.project_id, paper_id=arguments.paper_id
        ).model_dump(mode="json")
    if arguments.command == "import-pdf":
        file_version = library.admit_pdf(
            project_id=arguments.project_id,
            paper_id=arguments.paper_id,
            source_path=Path(arguments.source),
        )
        paths = ProjectPaths.create(Path(arguments.data_root), project_id=arguments.project_id)
        parsed = NativePdfParser(paths).parse(file_version)
        published = DocumentImportService(
            library.repository_for_project_id(arguments.project_id)
        ).publish(parsed)
        return {
            "file_version": file_version.model_dump(mode="json"),
            "generation": published.model_dump(mode="json"),
        }
    if arguments.command == "semantic-index":
        return _build_semantic_index(library, arguments)
    project = Project(project_id=arguments.project_id, display_name="retrieval")
    if arguments.mode in ("semantic", "hybrid"):
        if not arguments.embedding_profile_id or not arguments.model_root:
            raise ValueError(
                f"{arguments.mode} search requires --embedding-profile-id and --model-root"
            )
        if arguments.mode == "semantic":
            semantic_results = SemanticRetrievalService.open_from_model_root(
                data_root=Path(arguments.data_root),
                project_id=arguments.project_id,
                profile_id=arguments.embedding_profile_id,
                model_root=Path(arguments.model_root),
            ).search(project, arguments.query, limit=arguments.limit)
            return {"mode": "semantic", **semantic_results.model_dump(mode="json")}
        hybrid_results = HybridRetrievalService.open_from_model_root(
            data_root=Path(arguments.data_root),
            project_id=arguments.project_id,
            profile_id=arguments.embedding_profile_id,
            model_root=Path(arguments.model_root),
        ).search(project, arguments.query, limit=arguments.limit)
        return {"mode": "hybrid", **hybrid_results.model_dump(mode="json")}
    return (
        RetrievalService(data_root=Path(arguments.data_root))
        .search(project, arguments.query, limit=arguments.limit)
        .model_dump(mode="json")
    )


def _build_semantic_index(
    library: LibraryService, arguments: argparse.Namespace
) -> dict[str, object]:
    profile = approved_bge_small_en_v15_profile()
    if arguments.embedding_profile_id != profile.embedding_profile_id:
        raise SemanticIndexBuildError(
            "semantic index build does not support this embedding profile"
        )
    paths = ProjectPaths.create(Path(arguments.data_root), project_id=arguments.project_id)
    if not paths.database_path.is_file():
        raise SemanticIndexBuildError("project does not exist")
    if not ProjectRepository(paths).project_exists(arguments.project_id):
        raise SemanticIndexBuildError("project does not exist")
    try:
        artifacts = load_verified_artifacts(Path(arguments.model_root), profile=profile)
    except EmbeddingArtifactError as error:
        if "does not exist" in str(error):
            raise SemanticIndexBuildError("embedding model root is unavailable") from error
        raise SemanticIndexBuildError("embedding artifact verification failed") from error
    manifest_sha256 = hashlib.sha256(
        canonical_json_bytes(artifacts.manifest.canonical_payload())
    ).hexdigest()
    repository = EmbeddingRepository(paths)
    try:
        repository.register_profile(profile, artifact_manifest_sha256=manifest_sha256)
        embedder = OfflineEmbedder.open(Path(arguments.model_root), profile)
        result = ProjectVectorBuilder(
            paths=paths,
            repository=repository,
            profile=profile,
            tokenizer=embedder._tokenizer,
            embedder=embedder,
        ).build(project_id=arguments.project_id)
    except OfflineEmbedderError as error:
        raise SemanticIndexBuildError(
            "embedding tokenizer or runtime initialization failed"
        ) from error
    except EmbeddingPersistenceError as error:
        raise SemanticIndexBuildError(
            "semantic profile registration or publication failed"
        ) from error
    except VectorBuildError as error:
        raise SemanticIndexBuildError("semantic vector build failed") from error
    active = repository.active_generation(
        project_id=arguments.project_id, embedding_profile_id=profile.embedding_profile_id
    )
    if active is None or active.vector_generation_id != result.generation.vector_generation_id:
        raise SemanticIndexBuildError("semantic vector generation did not become active")
    generation = result.generation
    return {
        "project_id": generation.project_id,
        "embedding_profile_id": generation.embedding_profile_id,
        "vector_generation_id": generation.vector_generation_id,
        "source_snapshot_sha256": generation.source_snapshot_sha256,
        "coverage": asdict(generation.coverage),
        "artifact_relative_path": generation.artifact_relative_dir,
        "active": True,
        "current": True,
        "empty": result.empty,
        "reused": result.reused,
    }
