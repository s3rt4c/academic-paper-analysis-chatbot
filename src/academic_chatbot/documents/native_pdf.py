"""In-process parsing of admitted PDFs into verified native-text evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, BinaryIO, Literal, cast

import pdfplumber
from pdfminer.pdfexceptions import PDFException
from pdfplumber.utils.exceptions import PdfminerException

from academic_chatbot.documents.models import (
    CanonicalPdfPage,
    CanonicalPdfWord,
    NativePdfDocument,
    NativePdfPage,
)
from academic_chatbot.documents.normalization import (
    PARSER_PROFILE_SHA256,
    NativePdfNormalizationError,
    canonicalize_extracted_words,
    pdfplumber_extract_words_kwargs,
)
from academic_chatbot.domain.library import FileVersion
from academic_chatbot.ports.documents import NativePdfAnchor, PdfAnchorBox
from academic_chatbot.storage.paths import PathEscapeError, ProjectPaths

_STREAM_CHUNK_SIZE = 1024 * 1024


class NativePdfParseError(ValueError):
    """Base class for production native-PDF parsing failures."""


class NativePdfSourceError(NativePdfParseError):
    """Raised when a stored original cannot be safely opened as a regular file."""


class NativePdfIntegrityError(NativePdfParseError):
    """Raised when stored bytes do not match the admitted FileVersion digest."""


class NativePdfMalformedError(NativePdfParseError):
    """Raised when verified bytes cannot be parsed as a PDF."""


class NativePdfExtractionError(NativePdfParseError):
    """Raised when a parseable PDF yields invalid native extraction data."""


class NativePdfParser:
    """Parse one trusted, admitted FileVersion without persistence or OCR."""

    def __init__(self, paths: ProjectPaths, *, stream_chunk_size: int = _STREAM_CHUNK_SIZE) -> None:
        if stream_chunk_size <= 0:
            raise ValueError("stream_chunk_size must be positive")
        self._paths = paths
        self._stream_chunk_size = stream_chunk_size

    def parse(self, file_version: FileVersion) -> NativePdfDocument:
        """Verify and parse the immutable stored original for exactly this FileVersion."""

        source = self._resolve_stored_original(file_version)
        try:
            with source.open("rb") as handle:
                before = os.fstat(handle.fileno())
                self._verify_open_regular_source(source, before)
                actual_digest = _stream_sha256(handle, chunk_size=self._stream_chunk_size)
                after_hash = os.fstat(handle.fileno())
                if not _same_file_state(before, after_hash):
                    raise NativePdfIntegrityError(
                        "stored PDF changed while its digest was verified"
                    )
                if actual_digest != file_version.sha256:
                    raise NativePdfIntegrityError(
                        "stored PDF digest does not match FileVersion.sha256"
                    )
                handle.seek(0)
                pages = self._extract_verified_pages(handle, file_version)
                after_parse = os.fstat(handle.fileno())
                if not _same_file_state(before, after_parse):
                    raise NativePdfIntegrityError("stored PDF changed while it was parsed")
        except NativePdfParseError:
            raise
        except OSError as error:
            raise NativePdfSourceError("stored PDF could not be read") from error
        return NativePdfDocument(
            file_version=file_version,
            source_pdf_sha256=file_version.sha256,
            parser_profile_sha256=PARSER_PROFILE_SHA256,
            pages=tuple(pages),
        )

    def _resolve_stored_original(self, file_version: FileVersion) -> Path:
        try:
            source = self._paths.resolve_relative(file_version.stored_relative_path)
        except PathEscapeError as error:
            raise NativePdfSourceError(
                "FileVersion stored path escapes the project root"
            ) from error
        except OSError as error:
            raise NativePdfSourceError("stored PDF path could not be resolved") from error
        try:
            inspected = source.lstat()
        except FileNotFoundError as error:
            raise NativePdfSourceError("stored PDF does not exist") from error
        except OSError as error:
            raise NativePdfSourceError("stored PDF could not be inspected") from error
        if (
            stat.S_ISLNK(inspected.st_mode)
            or _is_reparse_point(inspected)
            or not stat.S_ISREG(inspected.st_mode)
        ):
            raise NativePdfSourceError(
                "stored PDF must be a regular non-symlink, non-reparse file"
            )
        return source

    @staticmethod
    def _verify_open_regular_source(source: Path, opened: os.stat_result) -> None:
        try:
            current = source.lstat()
        except FileNotFoundError as error:
            raise NativePdfIntegrityError("stored PDF changed before it could be opened") from error
        except OSError as error:
            raise NativePdfIntegrityError("stored PDF could not be re-inspected") from error
        if (
            stat.S_ISLNK(current.st_mode)
            or _is_reparse_point(current)
            or not stat.S_ISREG(current.st_mode)
            or not os.path.samestat(current, opened)
        ):
            raise NativePdfIntegrityError("stored PDF changed before it could be opened")

    @staticmethod
    def _extract_verified_pages(
        handle: BinaryIO, file_version: FileVersion
    ) -> list[NativePdfPage]:
        try:
            if handle.read(5) != b"%PDF-":
                raise NativePdfMalformedError("stored bytes do not have a PDF signature")
            handle.seek(0)
            with pdfplumber.open(cast(Any, handle)) as document:
                return [
                    _native_page_from_pdfplumber(
                        page,
                        physical_page_index=index,
                        file_version=file_version,
                    )
                    for index, page in enumerate(document.pages)
                ]
        except NativePdfParseError:
            raise
        except (PDFException, PdfminerException) as error:
            raise NativePdfMalformedError("stored bytes are not a parseable PDF") from error
        except NativePdfNormalizationError as error:
            raise NativePdfExtractionError("native PDF extraction data is invalid") from error
        except (OSError, TypeError, ValueError) as error:
            raise NativePdfExtractionError("native PDF extraction failed") from error


def build_native_pdf_anchor(
    *,
    file_version_id: str,
    source_pdf_sha256: str,
    physical_page_index: int,
    page_width_points: float,
    page_height_points: float,
    source_page_rotation_degrees: int,
    canonical_page: CanonicalPdfPage,
    word: CanonicalPdfWord,
) -> NativePdfAnchor:
    """Build one exact word anchor from construction offsets, never text search."""

    box = PdfAnchorBox(
        char_start=word.char_start,
        char_end=word.char_end,
        x0=word.x0,
        top=word.top,
        x1=word.x1,
        bottom=word.bottom,
    )
    boxes_payload = [box.model_dump(mode="json")]
    boxes_sha256 = _canonical_sha256(boxes_payload)
    canonical_text_sha256 = _text_sha256(canonical_page.text)
    anchor_text_sha256 = _text_sha256(word.text)
    binding_sha256 = _canonical_sha256(
        {"file_version_id": file_version_id, "pdf_sha256": source_pdf_sha256}
    )
    identity_payload = {
        "file_version_id": file_version_id,
        "pdf_sha256": source_pdf_sha256,
        "parser_profile_sha256": PARSER_PROFILE_SHA256,
        "physical_page_index": physical_page_index,
        "char_start": word.char_start,
        "char_end": word.char_end,
        "anchor_text_sha256": anchor_text_sha256,
        "boxes_sha256": boxes_sha256,
    }
    return NativePdfAnchor(
        evidence_id="ev-sha256-" + _canonical_sha256(identity_payload),
        file_version_id=file_version_id,
        file_version_binding_sha256=binding_sha256,
        source_pdf_sha256=source_pdf_sha256,
        parser_profile_sha256=PARSER_PROFILE_SHA256,
        physical_page_index=physical_page_index,
        display_page_number=physical_page_index + 1,
        printed_page_label=canonical_page.printed_page_label,
        page_width_points=page_width_points,
        page_height_points=page_height_points,
        source_page_rotation_degrees=source_page_rotation_degrees,
        char_start=word.char_start,
        char_end=word.char_end,
        canonical_page_text=canonical_page.text,
        canonical_page_text_sha256=canonical_text_sha256,
        anchor_text=word.text,
        anchor_text_sha256=anchor_text_sha256,
        boxes=(box,),
        boxes_sha256=boxes_sha256,
    )


def _native_page_from_pdfplumber(
    page: pdfplumber.page.Page, *, physical_page_index: int, file_version: FileVersion
) -> NativePdfPage:
    width = _finite_page_dimension(page.width)
    height = _finite_page_dimension(page.height)
    canonical = canonicalize_extracted_words(
        cast(list[dict[str, object]], page.extract_words(**pdfplumber_extract_words_kwargs())),
        page_width_points=width,
        page_height_points=height,
    )
    rotation = page.rotation or 0
    if isinstance(rotation, bool) or not isinstance(rotation, int):
        raise NativePdfExtractionError("PDF page rotation must be an integer")
    anchors = tuple(
        build_native_pdf_anchor(
            file_version_id=file_version.file_version_id,
            source_pdf_sha256=file_version.sha256,
            physical_page_index=physical_page_index,
            page_width_points=width,
            page_height_points=height,
            source_page_rotation_degrees=rotation,
            canonical_page=canonical,
            word=word,
        )
        for word in canonical.words
    )
    quality = _native_text_quality(canonical.text, len(canonical.words))
    return NativePdfPage(
        file_version_id=file_version.file_version_id,
        source_pdf_sha256=file_version.sha256,
        parser_profile_sha256=PARSER_PROFILE_SHA256,
        physical_page_index=physical_page_index,
        display_page_number=physical_page_index + 1,
        printed_page_label=canonical.printed_page_label,
        page_width_points=width,
        page_height_points=height,
        source_page_rotation_degrees=rotation,
        canonical_text=canonical.text,
        canonical_text_sha256=_text_sha256(canonical.text),
        words=canonical.words,
        quality=quality,
        needs_ocr=quality != "adequate_native_text",
        anchors=anchors,
    )


def _finite_page_dimension(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise NativePdfExtractionError("PDF page dimensions must be finite numbers")
    rounded = round(float(value), 6)
    if rounded <= 0.0:
        raise NativePdfExtractionError("PDF page dimensions must be positive")
    return 0.0 if rounded == 0.0 else rounded


def _stream_sha256(handle: BinaryIO, *, chunk_size: int) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def _same_file_state(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        os.path.samestat(before, after)
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )


def _is_reparse_point(metadata: os.stat_result) -> bool:
    attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attribute and metadata.st_file_attributes & attribute)


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _native_text_quality(
    canonical_text: str, word_count: int
) -> Literal["adequate_native_text", "low_native_text", "empty_native_text"]:
    if not canonical_text:
        return "empty_native_text"
    if word_count < 3 or len(canonical_text) < 24:
        return "low_native_text"
    return "adequate_native_text"
