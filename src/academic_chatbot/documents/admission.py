"""Streaming, content-addressed admission of untrusted local PDF files."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import uuid
from pathlib import Path

from academic_chatbot.domain.library import FileVersion, file_version_id_for
from academic_chatbot.library.repository import ProjectRepository
from academic_chatbot.storage.atomic import StreamCopyLimitError, atomic_stream_copy
from academic_chatbot.storage.paths import ProjectPaths

_PDF_SIGNATURE = b"%PDF-"


class PdfAdmissionError(ValueError):
    """Raised when an untrusted source cannot be safely admitted as a PDF."""


class PdfSizeLimitError(PdfAdmissionError):
    """Raised when a source exceeds the configured streaming admission limit."""


class PdfIntegrityError(PdfAdmissionError):
    """Raised when source or stored content changes during admission."""


def _file_digest(path: Path, *, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_is_reparse_point(source_stat: os.stat_result) -> bool:
    attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attribute and source_stat.st_file_attributes & attribute)


class PdfAdmissionService:
    """Coordinate safe local bytes publication before file-version registration.

    Path checks mitigate existing symlink/reparse escapes.  A malicious actor
    with write access to the controlled root can still race checks and later
    filesystem operations; the service never treats a file as authoritative
    until its matching database row is registered.
    """

    def __init__(
        self,
        *,
        paths: ProjectPaths,
        repository: ProjectRepository,
        max_pdf_bytes: int,
        chunk_size: int = 1024 * 1024,
    ) -> None:
        if max_pdf_bytes <= 0 or chunk_size <= 0:
            raise ValueError("max_pdf_bytes and chunk_size must be positive")
        self._paths = paths
        self._repository = repository
        self._max_pdf_bytes = max_pdf_bytes
        self._chunk_size = chunk_size

    def admit(self, *, paper_id: str, source_path: Path) -> FileVersion:
        source = source_path
        inspected_source = self._validate_source(source)
        self._paths.transactions_dir.mkdir(parents=True, exist_ok=True)
        staging = self._paths.transactions_dir / f"admission-{uuid.uuid4().hex}.part"
        published_target: Path | None = None
        published_here = False
        try:
            with source.open("rb") as source_handle:
                before = os.fstat(source_handle.fileno())
                self._verify_open_source(source, inspected_source, before)
                if source_handle.read(len(_PDF_SIGNATURE)) != _PDF_SIGNATURE:
                    raise PdfAdmissionError("source does not have a PDF signature")
                source_handle.seek(0)
                try:
                    copied = atomic_stream_copy(
                        source_handle,
                        staging,
                        max_bytes=self._max_pdf_bytes,
                        chunk_size=self._chunk_size,
                    )
                except StreamCopyLimitError as error:
                    raise PdfSizeLimitError(str(error)) from error
                after = os.fstat(source_handle.fileno())
            if not _same_source_state(before, after):
                raise PdfIntegrityError("source changed while it was being admitted")

            with self._repository.file_version_admission_transaction() as connection:
                stored_relative_path = f"originals/sha256/{copied.sha256}.pdf"
                target = self._paths.resolve_relative(stored_relative_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if (
                        target.is_symlink()
                        or _file_digest(target, chunk_size=self._chunk_size) != copied.sha256
                    ):
                        message = "existing content-addressed original does not match its digest"
                        raise PdfIntegrityError(message)
                else:
                    with staging.open("rb") as staged_handle:
                        published = atomic_stream_copy(
                            staged_handle,
                            target,
                            max_bytes=self._max_pdf_bytes,
                            chunk_size=self._chunk_size,
                        )
                    if (
                        published != copied
                        or _file_digest(target, chunk_size=self._chunk_size) != copied.sha256
                    ):
                        target.unlink(missing_ok=True)
                        message = "published original does not match the streamed source digest"
                        raise PdfIntegrityError(message)
                    published_target = target
                    published_here = True

                file_version = FileVersion(
                    file_version_id=file_version_id_for(paper_id=paper_id, sha256=copied.sha256),
                    paper_id=paper_id,
                    sha256=copied.sha256,
                    byte_length=copied.byte_length,
                    stored_relative_path=stored_relative_path,
                )
                try:
                    registered = self._repository.register_file_version_in_transaction(
                        connection=connection, file_version=file_version
                    )
                    if registered.paper_id != paper_id:
                        message = "registered file version does not belong to requested paper"
                        raise PdfAdmissionError(message)
                    return registered
                except BaseException:
                    self._cleanup_after_registration_failure(
                        connection, published_target, published_here, copied.sha256
                    )
                    raise
        finally:
            staging.unlink(missing_ok=True)

    def _validate_source(self, source_path: Path) -> os.stat_result:
        try:
            source_stat = source_path.lstat()
        except FileNotFoundError as error:
            raise PdfAdmissionError(f"source does not exist: {source_path}") from error
        if source_path.is_symlink() or _source_is_reparse_point(source_stat):
            raise PdfAdmissionError("source symlink or reparse point is not admitted")
        if not stat.S_ISREG(source_stat.st_mode):
            raise PdfAdmissionError("source must be a regular file")
        if source_stat.st_size > self._max_pdf_bytes:
            message = f"source exceeds the configured limit of {self._max_pdf_bytes} bytes"
            raise PdfSizeLimitError(message)
        return source_stat

    @staticmethod
    def _verify_open_source(
        source_path: Path,
        inspected_source: os.stat_result,
        opened_source: os.stat_result,
    ) -> None:
        try:
            current_path = source_path.lstat()
        except FileNotFoundError as error:
            raise PdfIntegrityError("source changed before it could be opened") from error
        if (
            source_path.is_symlink()
            or _source_is_reparse_point(current_path)
            or not stat.S_ISREG(current_path.st_mode)
            or not _same_source_state(inspected_source, current_path)
            or not _same_source_state(current_path, opened_source)
        ):
            raise PdfIntegrityError("source changed before it could be opened")

    def _cleanup_after_registration_failure(
        self,
        connection: sqlite3.Connection,
        published_target: Path | None,
        published_here: bool,
        sha256: str,
    ) -> None:
        if not published_here or published_target is None:
            return
        try:
            if not self._repository.any_file_version_references_sha256(
                sha256, connection=connection
            ):
                published_target.unlink(missing_ok=True)
        except BaseException:
            # If SQLite state cannot be determined, retain a non-authoritative
            # immutable orphan rather than deleting a file an ambiguous commit may reference.
            return


def _same_source_state(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        os.path.samestat(before, after)
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )
