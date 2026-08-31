"""Minimal local CLI for the approved Task 1--5 vertical slice."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from academic_chatbot.documents.import_service import DocumentImportService
from academic_chatbot.documents.native_pdf import NativePdfParser
from academic_chatbot.domain.library import Project
from academic_chatbot.library.service import LibraryService
from academic_chatbot.retrieval.fts import RetrievalQueryError
from academic_chatbot.retrieval.service import (
    RetrievalIntegrityError,
    RetrievalService,
    RetrievalStorageError,
)
from academic_chatbot.storage.paths import ProjectPaths


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
    search = commands.add_parser("search")
    search.add_argument("--project-id", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=10)
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
            project_id=arguments.project_id, paper_id=arguments.paper_id,
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
    project = Project(project_id=arguments.project_id, display_name="retrieval")
    return RetrievalService(data_root=Path(arguments.data_root)).search(
        project, arguments.query, limit=arguments.limit
    ).model_dump(mode="json")
