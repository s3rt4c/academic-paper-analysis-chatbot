"""Stable document-anchor boundary shared by Phase 0 feasibility code."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PdfAnchorBox(BaseModel):
    """One source word box and its page-local canonical character span."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    x0: float
    top: float
    x1: float
    bottom: float

    @model_validator(mode="after")
    def _validate_box(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        coordinates = (self.x0, self.top, self.x1, self.bottom)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("PDF anchor box coordinates must be finite")
        if any(value != round(value, 6) for value in coordinates) or any(
            value == 0.0 and math.copysign(1.0, value) < 0.0
            for value in coordinates
        ):
            raise ValueError("PDF anchor box coordinates must be canonical six-decimal values")
        if self.x0 >= self.x1:
            raise ValueError("x0 must be less than x1")
        if self.top >= self.bottom:
            raise ValueError("top must be less than bottom")
        return self


class NativePdfAnchor(BaseModel):
    """Immutable, content-addressed exact-text anchor in a native PDF."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(pattern=r"^ev-sha256-[0-9a-f]{64}$")
    file_version_id: str = Field(min_length=1)
    file_version_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_pdf_sha256: str = Field(pattern=_SHA256_PATTERN)
    parser_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    extraction_method: Literal["native_text"] = "native_text"
    physical_page_index: int = Field(ge=0)
    display_page_number: int = Field(ge=1)
    printed_page_label: str | None
    page_width_points: float = Field(gt=0.0)
    page_height_points: float = Field(gt=0.0)
    source_page_rotation_degrees: int
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    canonical_page_text: str = Field(min_length=1)
    canonical_page_text_sha256: str = Field(pattern=_SHA256_PATTERN)
    anchor_text: str = Field(min_length=1)
    anchor_text_sha256: str = Field(pattern=_SHA256_PATTERN)
    boxes: tuple[PdfAnchorBox, ...] = Field(min_length=1)
    boxes_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_anchor(self) -> Self:
        if not self.file_version_id.strip():
            raise ValueError("file_version_id must not be empty")
        if not math.isfinite(self.page_width_points) or not math.isfinite(
            self.page_height_points
        ):
            raise ValueError("PDF page dimensions must be finite")
        if self.display_page_number != self.physical_page_index + 1:
            raise ValueError("display_page_number must equal physical_page_index + 1")
        if not self.char_start < self.char_end <= len(self.canonical_page_text):
            raise ValueError("anchor offsets must be inside canonical_page_text")
        if self.canonical_page_text[self.char_start : self.char_end] != self.anchor_text:
            raise ValueError("anchor_text must equal the canonical page-text slice")
        if not hmac.compare_digest(
            self.canonical_page_text_sha256,
            _text_sha256(self.canonical_page_text),
        ):
            raise ValueError("canonical_page_text_sha256 does not match canonical_page_text")
        if not hmac.compare_digest(
            self.anchor_text_sha256,
            _text_sha256(self.anchor_text),
        ):
            raise ValueError("anchor_text_sha256 does not match anchor_text")

        previous_end = -1
        for box in self.boxes:
            if box.char_start < previous_end:
                raise ValueError("anchor box spans must be strictly ordered")
            previous_end = box.char_end
            if box.char_end > len(self.canonical_page_text):
                raise ValueError("anchor box spans must be inside canonical_page_text")
            if box.char_end <= self.char_start or box.char_start >= self.char_end:
                raise ValueError("every anchor box span must intersect the anchor range")
            if not 0.0 <= box.x0 < box.x1 <= self.page_width_points:
                raise ValueError("anchor box horizontal coordinates must be page-bound")
            if not 0.0 <= box.top < box.bottom <= self.page_height_points:
                raise ValueError("anchor box vertical coordinates must be page-bound")

        boxes_payload = [box.model_dump(mode="json") for box in self.boxes]
        expected_boxes_sha256 = _canonical_sha256(boxes_payload)
        if not hmac.compare_digest(self.boxes_sha256, expected_boxes_sha256):
            raise ValueError("boxes_sha256 does not match the ordered anchor boxes")

        expected_binding = _canonical_sha256(
            {
                "file_version_id": self.file_version_id,
                "pdf_sha256": self.source_pdf_sha256,
            }
        )
        if not hmac.compare_digest(
            self.file_version_binding_sha256,
            expected_binding,
        ):
            raise ValueError("file_version_binding_sha256 is inconsistent with the source")

        identity_payload = {
            "file_version_id": self.file_version_id,
            "pdf_sha256": self.source_pdf_sha256,
            "parser_profile_sha256": self.parser_profile_sha256,
            "physical_page_index": self.physical_page_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "anchor_text_sha256": self.anchor_text_sha256,
            "boxes_sha256": self.boxes_sha256,
        }
        expected_evidence_id = "ev-sha256-" + _canonical_sha256(identity_payload)
        if not hmac.compare_digest(self.evidence_id, expected_evidence_id):
            raise ValueError("evidence_id is inconsistent with the anchor identity")
        return self


class PdfAnchorLocator(Protocol):
    """Locate one exact native-text occurrence in an immutable PDF version."""

    def locate(
        self,
        source: Path,
        *,
        file_version_id: str,
        needle: str,
    ) -> NativePdfAnchor | None: ...
