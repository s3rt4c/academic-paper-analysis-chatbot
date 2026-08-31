"""Atomic publication of Task 3 native-PDF output into the project database."""

from __future__ import annotations

import hashlib

from academic_chatbot.documents.chunking import LEXICAL_CHUNK_PROFILE_ID
from academic_chatbot.documents.models import NativePdfDocument, PublishedDocumentGeneration
from academic_chatbot.library.repository import ProjectRepository

DOCUMENT_GENERATION_PROFILE_ID = "native-lexical-fts-v1"


def processing_profile_id_for(*, lexical_chunk_profile_id: str) -> str:
    """Bind the immutable generation profile to its lexical chunk profile."""

    if not lexical_chunk_profile_id:
        raise ValueError("lexical chunk profile identity must not be empty")
    return f"{DOCUMENT_GENERATION_PROFILE_ID}:{lexical_chunk_profile_id}"


def document_generation_id_for(*, file_version_id: str, processing_profile_id: str) -> str:
    """Return a stable generation identity from logical FileVersion and profile."""

    file_version_bytes = file_version_id.encode("utf-8")
    profile_bytes = processing_profile_id.encode("utf-8")
    identity = (
        len(file_version_bytes).to_bytes(8, "big")
        + file_version_bytes
        + len(profile_bytes).to_bytes(8, "big")
        + profile_bytes
    )
    return "dg-sha256-" + hashlib.sha256(identity).hexdigest()


class DocumentImportService:
    """Persist a validated Task 3 parse without reparsing or normalizing text."""

    def __init__(self, repository: ProjectRepository) -> None:
        self._repository = repository

    def publish(self, parsed: NativePdfDocument) -> PublishedDocumentGeneration:
        lexical_chunk_profile_id = LEXICAL_CHUNK_PROFILE_ID
        processing_profile_id = processing_profile_id_for(
            lexical_chunk_profile_id=lexical_chunk_profile_id
        )
        return self._repository.publish_native_document(
            parsed=parsed,
            document_generation_id=document_generation_id_for(
                file_version_id=parsed.file_version.file_version_id,
                processing_profile_id=processing_profile_id,
            ),
            processing_profile_id=processing_profile_id,
            lexical_chunk_profile_id=lexical_chunk_profile_id,
        )
