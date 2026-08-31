"""Immutable in-memory native-PDF evidence results."""

from __future__ import annotations

import hashlib
import hmac
import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from academic_chatbot.domain.library import FileVersion
from academic_chatbot.ports.documents import NativePdfAnchor

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CanonicalPdfWord(BaseModel):
    """One normalized extracted word and its page-local source geometry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    x0: float
    top: float
    x1: float
    bottom: float

    @model_validator(mode="after")
    def _validate_word(self) -> Self:
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("canonical word character span must equal its text length")
        coordinates = (self.x0, self.top, self.x1, self.bottom)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("canonical word coordinates must be finite")
        if any(value != round(value, 6) for value in coordinates) or any(
            value == 0.0 and math.copysign(1.0, value) < 0.0
            for value in coordinates
        ):
            raise ValueError("canonical word coordinates must be six-decimal values")
        if self.x0 >= self.x1 or self.top >= self.bottom:
            raise ValueError("canonical word coordinates must not be inverted")
        return self


class CanonicalPdfPage(BaseModel):
    """Canonical native text assembled directly from ordered extracted words."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    words: tuple[CanonicalPdfWord, ...]
    printed_page_label: str | None

    @model_validator(mode="after")
    def _validate_page(self) -> Self:
        expected = " ".join(word.text for word in self.words)
        if self.text != expected:
            raise ValueError("canonical page text must be built from ordered words")
        cursor = 0
        for index, word in enumerate(self.words):
            if word.char_start != cursor or word.char_end != cursor + len(word.text):
                raise ValueError("canonical word offsets must be construction offsets")
            cursor = word.char_end + (1 if index < len(self.words) - 1 else 0)
        return self


class NativePdfPage(BaseModel):
    """One physical PDF page, its canonical text, and verified word anchors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_version_id: str = Field(min_length=1)
    source_pdf_sha256: str = Field(pattern=_SHA256_PATTERN)
    parser_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    physical_page_index: int = Field(ge=0)
    display_page_number: int = Field(ge=1)
    printed_page_label: str | None
    page_width_points: float = Field(gt=0.0)
    page_height_points: float = Field(gt=0.0)
    source_page_rotation_degrees: int
    canonical_text: str
    canonical_text_sha256: str = Field(pattern=_SHA256_PATTERN)
    words: tuple[CanonicalPdfWord, ...]
    quality: Literal["adequate_native_text", "low_native_text", "empty_native_text"]
    needs_ocr: bool
    anchors: tuple[NativePdfAnchor, ...]

    @model_validator(mode="after")
    def _validate_page(self) -> Self:
        if self.display_page_number != self.physical_page_index + 1:
            raise ValueError("display_page_number must equal physical_page_index + 1")
        if not math.isfinite(self.page_width_points) or not math.isfinite(
            self.page_height_points
        ):
            raise ValueError("PDF page dimensions must be finite")
        if self.canonical_text != " ".join(word.text for word in self.words):
            raise ValueError("canonical text must match ordered canonical words")
        if not hmac.compare_digest(
            self.canonical_text_sha256, _text_sha256(self.canonical_text)
        ):
            raise ValueError("canonical_text_sha256 does not match canonical_text")
        expected_quality = _native_text_quality(self.canonical_text, len(self.words))
        if self.quality != expected_quality:
            raise ValueError("native text quality does not match the canonical text")
        if self.needs_ocr != (expected_quality != "adequate_native_text"):
            raise ValueError("needs_ocr must reflect only native text insufficiency")
        if len(self.anchors) != len(self.words):
            raise ValueError("each canonical word must have exactly one native evidence anchor")
        for word, anchor in zip(self.words, self.anchors, strict=True):
            if (
                anchor.file_version_id != self.file_version_id
                or anchor.source_pdf_sha256 != self.source_pdf_sha256
                or anchor.parser_profile_sha256 != self.parser_profile_sha256
                or anchor.physical_page_index != self.physical_page_index
                or anchor.display_page_number != self.display_page_number
                or anchor.canonical_page_text != self.canonical_text
                or anchor.char_start != word.char_start
                or anchor.char_end != word.char_end
                or anchor.anchor_text != word.text
            ):
                raise ValueError("native anchor does not match its parsed page and word")
        return self


class NativePdfDocument(BaseModel):
    """Result of parsing one admitted immutable FileVersion in process memory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_version: FileVersion
    source_pdf_sha256: str = Field(pattern=_SHA256_PATTERN)
    parser_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    pages: tuple[NativePdfPage, ...]

    @model_validator(mode="after")
    def _validate_document(self) -> Self:
        if self.source_pdf_sha256 != self.file_version.sha256:
            raise ValueError("parsed source digest must equal the input FileVersion digest")
        for index, page in enumerate(self.pages):
            if (
                page.file_version_id != self.file_version.file_version_id
                or page.source_pdf_sha256 != self.source_pdf_sha256
                or page.parser_profile_sha256 != self.parser_profile_sha256
                or page.physical_page_index != index
            ):
                raise ValueError("parsed page does not retain input FileVersion lineage")
        return self


def _native_text_quality(canonical_text: str, word_count: int) -> Literal[
    "adequate_native_text", "low_native_text", "empty_native_text"
]:
    """Return the conservative Phase 1A native-text sufficiency heuristic.

    ``needs_ocr`` means only that the native text layer is insufficient for
    this evidence pipeline; it neither proves OCR is required nor predicts
    that OCR would succeed.
    """

    if not canonical_text:
        return "empty_native_text"
    if word_count < 3 or len(canonical_text) < 24:
        return "low_native_text"
    return "adequate_native_text"
