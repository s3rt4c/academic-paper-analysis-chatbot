"""Deterministic native-PDF text anchors for the Phase 0 feasibility gate."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Literal, NoReturn, Self, cast

import pdfminer
import pdfplumber
import pypdfium2 as pdfium  # type: ignore[import-untyped]
import reportlab  # type: ignore[import-untyped]
from pdfminer.pdfexceptions import PDFException
from PIL import Image
from PIL import __version__ as pillow_version
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticSerializationError

from academic_chatbot.feasibility.hardware import HardwareFacts
from academic_chatbot.ports.documents import NativePdfAnchor, PdfAnchorBox

REFERENCE_FIXTURE_PROFILE_ID = "reportlab-native-anchor-v1"
REFERENCE_FIXTURE_SIZE_BYTES = 2_470
REFERENCE_FIXTURE_SHA256 = (
    "2d9c30592721d5e27f39c6a047f4e10f2577868d0ac1ef836a81dbdb8180175e"
)
REFERENCE_FILE_VERSION_ID = "fv-phase0-native-anchor-v1"
REFERENCE_NEEDLE = "The anchor sentence reports an accuracy of 91.2 percent."

_FOOTER_LABEL_PATTERN = re.compile(r"^[A-Z]+-[0-9]+$")
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class PdfAnchorOperationalError(ValueError):
    """Stable expected failure at the deterministic PDF-anchor boundary."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_file_bytes(payload: object) -> bytes:
    return _canonical_json_bytes(payload) + b"\n"


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("JSON object keys must be unique.")
        payload[key] = value
    return payload


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON number {value!r} is forbidden.")


def _load_raw_canonical_json_object(
    path: Path,
    *,
    invalid_message: str,
) -> tuple[bytes, dict[str, object]]:
    try:
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8")
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
        if not isinstance(decoded, dict):
            raise ValueError("The canonical JSON root must be an object.")
        payload = dict(decoded)
        if raw != _canonical_json_file_bytes(payload):
            raise ValueError("The JSON file is not in canonical form.")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise PdfAnchorOperationalError(invalid_message) from error
    return raw, payload


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_finite_float(value: object, *, field_group: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"{field_group} must use finite float values")
    return value


class PdfParserProfile(BaseModel):
    """Complete immutable native-text extraction and anchor policy."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    profile_id: Literal["pdfplumber-native-anchor-v1"] = "pdfplumber-native-anchor-v1"
    pdfplumber_version: Literal["0.11.10"] = "0.11.10"
    normalization_profile_id: Literal["nfc-unicode-whitespace-ascii-space-v1"] = (
        "nfc-unicode-whitespace-ascii-space-v1"
    )
    unicode_normalization: Literal["NFC"] = "NFC"
    whitespace_rule: Literal["maximal-unicode-runs-to-ascii-space-and-trim"] = (
        "maximal-unicode-runs-to-ascii-space-and-trim"
    )
    preserve_case: Literal[True] = True
    preserve_punctuation: Literal[True] = True
    preserve_symbols: Literal[True] = True
    preserve_compatibility_characters: Literal[True] = True
    dehyphenate: Literal[False] = False
    empty_normalized_token_policy: Literal[
        "discard-only-empty-normalized-word-tokens"
    ] = "discard-only-empty-normalized-word-tokens"
    match_mode: Literal["exact-overlapping-substring"] = "exact-overlapping-substring"
    occurrence_policy: Literal[
        "find-every-overlapping-exact-normalized-substring"
    ] = "find-every-overlapping-exact-normalized-substring"
    substring_boundary_policy: Literal["may-start-or-end-inside-source-word"] = (
        "may-start-or-end-inside-source-word"
    )
    match_cardinality_policy: Literal["zero-none-one-anchor-many-error"] = (
        "zero-none-one-anchor-many-error"
    )
    anchor_box_policy: Literal[
        "ordered-unmerged-source-word-boxes-intersecting-[char_start,char_end)"
    ] = "ordered-unmerged-source-word-boxes-intersecting-[char_start,char_end)"
    inside_word_box_span_policy: Literal[
        "retain-full-page-local-source-word-span"
    ] = "retain-full-page-local-source-word-span"
    offset_unit: Literal["page-local-unicode-code-point-half-open"] = (
        "page-local-unicode-code-point-half-open"
    )
    x_tolerance: float = Field(default=3.0, ge=3.0, le=3.0)
    y_tolerance: float = Field(default=3.0, ge=3.0, le=3.0)
    x_tolerance_ratio: None = None
    y_tolerance_ratio: None = None
    keep_blank_chars: Literal[False] = False
    use_text_flow: Literal[False] = False
    line_dir: Literal["ttb"] = "ttb"
    char_dir: Literal["ltr"] = "ltr"
    split_at_punctuation: Literal[False] = False
    expand_ligatures: Literal[True] = True
    return_chars: Literal[False] = False
    line_vertical_key: Literal["round6((top+bottom)/2)"] = (
        "round6((top+bottom)/2)"
    )
    line_cluster_tolerance_points: float = Field(default=3.0, ge=3.0, le=3.0)
    line_representative: Literal["first-word-frozen"] = "first-word-frozen"
    candidate_sort_keys: tuple[
        Literal["vertical_key"],
        Literal["x0"],
        Literal["top"],
        Literal["original_extraction_index"],
    ] = ("vertical_key", "x0", "top", "original_extraction_index")
    line_sort_keys: tuple[
        Literal["representative_vertical_key"],
        Literal["minimum_x0"],
        Literal["minimum_original_extraction_index"],
    ] = (
        "representative_vertical_key",
        "minimum_x0",
        "minimum_original_extraction_index",
    )
    word_sort_keys: tuple[
        Literal["x0"],
        Literal["top"],
        Literal["original_extraction_index"],
    ] = ("x0", "top", "original_extraction_index")
    word_joiner_ascii_codepoint: Literal[32] = 32
    coordinate_system: Literal["pdf-points-top-left-origin"] = (
        "pdf-points-top-left-origin"
    )
    coordinate_decimal_places: Literal[6] = 6
    canonicalize_negative_zero: Literal[True] = True
    footer_band_points: float = Field(default=72.0, ge=72.0, le=72.0)
    footer_center_tolerance_points: float = Field(default=18.0, ge=18.0, le=18.0)
    footer_label_regex: Literal[r"^[A-Z]+-[0-9]+$"] = r"^[A-Z]+-[0-9]+$"
    footer_candidate_policy: Literal["exactly-one-standalone-line"] = (
        "exactly-one-standalone-line"
    )
    footer_band_formula: Literal[
        "rounded_line_top>=page_height_points-72.0"
    ] = "rounded_line_top>=page_height_points-72.0"
    footer_center_formula: Literal[
        "abs((rounded_line_x0+rounded_line_x1)/2-page_width_points/2)<=18.0"
    ] = "abs((rounded_line_x0+rounded_line_x1)/2-page_width_points/2)<=18.0"
    footer_cardinality_policy: Literal["zero-or-many-none-one-label"] = (
        "zero-or-many-none-one-label"
    )

    @field_validator(
        "x_tolerance",
        "y_tolerance",
        "line_cluster_tolerance_points",
        "footer_band_points",
        "footer_center_tolerance_points",
        mode="before",
    )
    @classmethod
    def _validate_fixed_float_fields(cls, value: object) -> float:
        return _require_finite_float(value, field_group="Parser profile float fields")


class PdfRenderProfile(BaseModel):
    """Complete immutable PDFium input and byte-hashing policy."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    profile_id: Literal["pdfium-rgba-v1"] = "pdfium-rgba-v1"
    scale: float = Field(default=2.0, ge=2.0, le=2.0)
    dpi: Literal[144] = 144
    additional_rotation_degrees: Literal[0] = 0
    crop: tuple[Literal[0], Literal[0], Literal[0], Literal[0]] = (0, 0, 0, 0)
    may_draw_forms: Literal[False] = False
    draw_annots: Literal[False] = False
    fill_color: tuple[
        Literal[255], Literal[255], Literal[255], Literal[255]
    ] = (255, 255, 255, 255)
    force_bitmap_format: Literal["FPDFBitmap_BGRA"] = "FPDFBitmap_BGRA"
    rev_byteorder: Literal[True] = True
    optimize_mode: None = None
    no_smoothtext: Literal[False] = False
    no_smoothimage: Literal[False] = False
    no_smoothpath: Literal[False] = False
    force_halftone: Literal[False] = False
    limit_image_cache: Literal[False] = False
    extra_flags: Literal[0] = 0
    color_scheme: None = None
    fill_to_stroke: Literal[False] = False
    source_rotation_recorded: Literal[True] = True
    dimension_rule: Literal["ceil-source-page-points-times-scale"] = (
        "ceil-source-page-points-times-scale"
    )
    packed_pixel_mode: Literal["RGBA"] = "RGBA"
    packed_layout: Literal["row-major-tight-rgba8"] = "row-major-tight-rgba8"
    raw_hash_domain_utf8: Literal["pdfium-rgba-v1\0"] = "pdfium-rgba-v1\0"
    raw_hash_dimension_encoding: Literal["uint64be-width-then-height"] = (
        "uint64be-width-then-height"
    )
    png_mode: Literal["RGBA"] = "RGBA"
    png_optimize: Literal[False] = False
    png_compress_level: Literal[9] = 9
    png_metadata_policy: Literal["none"] = "none"

    @field_validator("scale", mode="before")
    @classmethod
    def _validate_scale(cls, value: object) -> float:
        return _require_finite_float(value, field_group="Render profile scale")


class PdfRenderEvidence(BaseModel):
    """Checksums and fixed rendering facts without bitmap or PNG payload bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    renderer_profile_id: Literal["pdfium-rgba-v1"] = "pdfium-rgba-v1"
    physical_page_index: int = Field(ge=0)
    source_page_rotation_degrees: int
    scale: float = Field(default=2.0, ge=2.0, le=2.0)
    dpi: Literal[144] = 144
    additional_rotation_degrees: Literal[0] = 0
    draw_annotations: Literal[False] = False
    draw_forms: Literal[False] = False
    background_rgba: tuple[
        Literal[255], Literal[255], Literal[255], Literal[255]
    ] = (255, 255, 255, 255)
    pixel_mode: Literal["RGBA"] = "RGBA"
    pixel_width: int = Field(gt=0)
    pixel_height: int = Field(gt=0)
    rgba_byte_count: int = Field(gt=0)
    rgba_sha256: str = Field(pattern=_SHA256_PATTERN)
    render_sha256: str = Field(pattern=_SHA256_PATTERN)
    png_byte_count: int = Field(gt=0)
    png_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("scale", mode="before")
    @classmethod
    def _validate_scale(cls, value: object) -> float:
        return _require_finite_float(value, field_group="Render evidence scale")

    @model_validator(mode="after")
    def _validate_byte_count(self) -> Self:
        if self.rgba_byte_count != self.pixel_width * self.pixel_height * 4:
            raise ValueError("rgba_byte_count must equal pixel_width * pixel_height * 4")
        return self


@dataclass(frozen=True, slots=True)
class RenderedPdfPage:
    """In-memory render result whose byte payloads never enter the documents port."""

    physical_page_index: int
    page_width_points: float
    page_height_points: float
    pixel_width: int
    pixel_height: int
    packed_rgba_bytes: bytes
    png_bytes: bytes
    evidence: PdfRenderEvidence


def compute_pdf_parser_profile_sha256(profile: PdfParserProfile) -> str:
    """Hash a validated parser profile as canonical JSON without a newline."""

    validated = PdfParserProfile.model_validate(profile.model_dump(mode="python"))
    return _canonical_sha256(validated.model_dump(mode="json"))


def compute_pdf_render_profile_sha256(profile: PdfRenderProfile) -> str:
    """Hash a validated renderer profile as canonical JSON without a newline."""

    validated = PdfRenderProfile.model_validate(profile.model_dump(mode="python"))
    return _canonical_sha256(validated.model_dump(mode="json"))


DEFAULT_PARSER_PROFILE = PdfParserProfile()
DEFAULT_RENDER_PROFILE = PdfRenderProfile()
DEFAULT_PARSER_PROFILE_SHA256 = compute_pdf_parser_profile_sha256(
    DEFAULT_PARSER_PROFILE
)
DEFAULT_RENDER_PROFILE_SHA256 = compute_pdf_render_profile_sha256(
    DEFAULT_RENDER_PROFILE
)


class PdfAnchorReferenceProfile(BaseModel):
    """Frozen source, request, toolchain, anchor, and render expectations."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    profile_id: Literal["phase0-native-pdf-anchor-v1"] = (
        "phase0-native-pdf-anchor-v1"
    )
    fixture_profile_id: str = Field(default=REFERENCE_FIXTURE_PROFILE_ID, min_length=1)
    fixture_size_bytes: int = Field(default=REFERENCE_FIXTURE_SIZE_BYTES, gt=0)
    fixture_sha256: str = Field(default=REFERENCE_FIXTURE_SHA256, pattern=_SHA256_PATTERN)
    page_count: int = Field(default=2, gt=0)
    file_version_id: str = Field(default=REFERENCE_FILE_VERSION_ID, min_length=1)
    file_version_binding_sha256: str = Field(
        default=(
            "2497249fc3f3b35388554b98ab4c9dbf75b810e32d612c3b8a91380900e87e60"
        ),
        pattern=_SHA256_PATTERN,
    )
    needle: str = Field(default=REFERENCE_NEEDLE, min_length=1)
    physical_page_index: int = Field(default=1, ge=0)
    display_page_number: int = Field(default=2, ge=1)
    printed_page_label: str | None = "A-7"
    parser_profile: PdfParserProfile = Field(default_factory=PdfParserProfile)
    parser_profile_sha256: str = Field(
        default=DEFAULT_PARSER_PROFILE_SHA256,
        pattern=_SHA256_PATTERN,
    )
    renderer_profile: PdfRenderProfile = Field(default_factory=PdfRenderProfile)
    renderer_profile_sha256: str = Field(
        default=DEFAULT_RENDER_PROFILE_SHA256,
        pattern=_SHA256_PATTERN,
    )
    python_version: str = Field(default="3.12.13", min_length=1)
    pdfminer_version: str = Field(default="20260107", min_length=1)
    pdfplumber_version: str = Field(default="0.11.10", min_length=1)
    pypdfium2_version: str = Field(default="5.11.0", min_length=1)
    pdfium_version: str = Field(default="151.0.7920.0", min_length=1)
    pillow_version: str = Field(default="12.3.0", min_length=1)
    reportlab_version: str = Field(default="5.0.0", min_length=1)
    page_width_points: float = Field(default=612.0, gt=0.0)
    page_height_points: float = Field(default=792.0, gt=0.0)
    source_page_rotation_degrees: int = 0
    canonical_page_text_sha256: str = Field(
        default=(
            "1e8db89970b5bc9db55546d506859fd5f565fce71df0edd926a68d5924ca3ebf"
        ),
        pattern=_SHA256_PATTERN,
    )
    char_start: int = Field(default=55, ge=0)
    char_end: int = Field(default=111, gt=0)
    anchor_text_sha256: str = Field(
        default=(
            "13ae5b7b01af4390ac74497e4d6d4a435cc12c2a09584848b9ad04e65897adcf"
        ),
        pattern=_SHA256_PATTERN,
    )
    boxes_sha256: str = Field(
        default=(
            "4fb9553245ae187bfc2fb4e4e31f754c3334234d94c5d8c3bd589b50ff04ede0"
        ),
        pattern=_SHA256_PATTERN,
    )
    evidence_id: str = Field(
        default=(
            "ev-sha256-208ff8ced2f81e9c1f94fb71bff43ce8ce57acac00b8c358c2e2ff9912a7d98a"
        ),
        pattern=r"^ev-sha256-[0-9a-f]{64}$",
    )
    pixel_mode: Literal["RGBA"] = "RGBA"
    pixel_width: int = Field(default=1_224, gt=0)
    pixel_height: int = Field(default=1_584, gt=0)
    rgba_byte_count: int = Field(default=7_755_264, gt=0)
    rgba_sha256: str = Field(
        default=(
            "5981aa587ef5712368840795711ae5c01179ae4b38e1e6820b6787945f42bd4d"
        ),
        pattern=_SHA256_PATTERN,
    )
    render_sha256: str = Field(
        default=(
            "51146fb529d05930636779d894047e637afac3d1b0a87238965fd59075003c32"
        ),
        pattern=_SHA256_PATTERN,
    )
    png_byte_count: int = Field(default=46_285, gt=0)
    png_sha256: str = Field(
        default=(
            "b42698373697c3029da02294b78b1d9fb2624f346f85cfe6464684f9a4baa855"
        ),
        pattern=_SHA256_PATTERN,
    )

    @field_validator("page_width_points", "page_height_points", mode="before")
    @classmethod
    def _validate_page_dimension_floats(cls, value: object) -> float:
        return _require_finite_float(
            value,
            field_group="Reference page dimensions",
        )

    @model_validator(mode="after")
    def _validate_reference_profile(self) -> Self:
        if self.physical_page_index >= self.page_count:
            raise ValueError("physical_page_index must be inside the reference PDF")
        if self.display_page_number != self.physical_page_index + 1:
            raise ValueError("display_page_number must equal physical_page_index + 1")
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        if self.parser_profile_sha256 != compute_pdf_parser_profile_sha256(
            self.parser_profile
        ):
            raise ValueError("parser_profile_sha256 does not match parser_profile")
        if self.renderer_profile_sha256 != compute_pdf_render_profile_sha256(
            self.renderer_profile
        ):
            raise ValueError("renderer_profile_sha256 does not match renderer_profile")

        normalized_needle = " ".join(
            unicodedata.normalize("NFC", self.needle).split()
        )
        if not normalized_needle:
            raise ValueError("needle must not normalize to empty text")
        if self.char_end - self.char_start != len(normalized_needle):
            raise ValueError("reference offsets must span the normalized needle")
        if self.anchor_text_sha256 != _text_sha256(normalized_needle):
            raise ValueError("anchor_text_sha256 does not match the normalized needle")

        expected_binding = _canonical_sha256(
            {
                "file_version_id": self.file_version_id,
                "pdf_sha256": self.fixture_sha256,
            }
        )
        if self.file_version_binding_sha256 != expected_binding:
            raise ValueError("file_version_binding_sha256 is inconsistent")
        identity_payload = {
            "file_version_id": self.file_version_id,
            "pdf_sha256": self.fixture_sha256,
            "parser_profile_sha256": self.parser_profile_sha256,
            "physical_page_index": self.physical_page_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "anchor_text_sha256": self.anchor_text_sha256,
            "boxes_sha256": self.boxes_sha256,
        }
        if self.evidence_id != "ev-sha256-" + _canonical_sha256(identity_payload):
            raise ValueError("evidence_id is inconsistent with the reference anchor")

        expected_pixel_width = math.ceil(
            self.page_width_points * self.renderer_profile.scale
        )
        expected_pixel_height = math.ceil(
            self.page_height_points * self.renderer_profile.scale
        )
        if (self.pixel_width, self.pixel_height) != (
            expected_pixel_width,
            expected_pixel_height,
        ):
            raise ValueError("reference pixel dimensions are inconsistent")
        if self.rgba_byte_count != self.pixel_width * self.pixel_height * 4:
            raise ValueError("reference rgba_byte_count is inconsistent")
        return self


def compute_pdf_anchor_reference_profile_sha256(
    profile: PdfAnchorReferenceProfile,
) -> str:
    """Hash one validated reference profile as canonical JSON."""

    validated = PdfAnchorReferenceProfile.model_validate(
        profile.model_dump(mode="python")
    )
    return _canonical_sha256(validated.model_dump(mode="json"))


DEFAULT_REFERENCE_PROFILE = PdfAnchorReferenceProfile()
DEFAULT_REFERENCE_PROFILE_SHA256 = compute_pdf_anchor_reference_profile_sha256(
    DEFAULT_REFERENCE_PROFILE
)


def _runtime_tool_versions() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "pdfminer_version": str(pdfminer.__version__),
        "pdfplumber_version": str(pdfplumber.__version__),
        "pypdfium2_version": importlib.metadata.version("pypdfium2"),
        "pdfium_version": str(pdfium.PDFIUM_INFO),
        "pillow_version": pillow_version,
        "reportlab_version": str(reportlab.Version),
    }


_UTC_TIMESTAMP_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z$"
)


def _validate_utc_timestamp(value: str) -> str:
    if _UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "measured_at_utc must be an RFC 3339 UTC timestamp ending in Z"
        )
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError("measured_at_utc must be a valid UTC timestamp") from error
    if parsed.utcoffset() != UTC.utcoffset(None):
        raise ValueError("measured_at_utc must use UTC")
    return value


class PdfAnchorReport(BaseModel):
    """Canonical success-only evidence for the deterministic PDF anchor gate."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    report_type: Literal["pdf_anchor"] = "pdf_anchor"
    artifact_kind: Literal["deterministic_correctness"] = (
        "deterministic_correctness"
    )
    verification_status: Literal["verified"] = "verified"
    measured_at_utc: str = Field(min_length=1)
    profile: PdfAnchorReferenceProfile
    profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    parser_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    renderer_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    reference_profile_verified: Literal[True] = True
    pdf_sha256: str = Field(pattern=_SHA256_PATTERN)
    pdf_size_bytes: int = Field(gt=0)
    page_count: int = Field(gt=0)
    file_version_id: str = Field(min_length=1)
    file_version_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    hardware_facts_sha256: str = Field(pattern=_SHA256_PATTERN)
    python_version: str = Field(min_length=1)
    pdfminer_version: str = Field(min_length=1)
    pdfplumber_version: str = Field(min_length=1)
    pypdfium2_version: str = Field(min_length=1)
    pdfium_version: str = Field(min_length=1)
    pillow_version: str = Field(min_length=1)
    reportlab_version: str = Field(min_length=1)
    anchor: NativePdfAnchor
    boxes_sha256: str = Field(pattern=_SHA256_PATTERN)
    render: PdfRenderEvidence
    anchor_integrity_verified: Literal[True] = True
    render_integrity_verified: Literal[True] = True
    report_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("measured_at_utc")
    @classmethod
    def _validate_measurement_time(cls, value: str) -> str:
        return _validate_utc_timestamp(value)

    @model_validator(mode="after")
    def _validate_frozen_reference_evidence(self) -> Self:
        unsigned = self.model_dump(mode="json", exclude={"report_sha256"})
        if not hmac.compare_digest(self.report_sha256, _canonical_sha256(unsigned)):
            raise ValueError(
                "report_sha256 does not match the canonical report payload"
            )

        if self.profile != DEFAULT_REFERENCE_PROFILE:
            raise ValueError("profile must equal the frozen reference profile")
        if self.profile_sha256 != DEFAULT_REFERENCE_PROFILE_SHA256:
            raise ValueError("profile_sha256 does not match the reference profile")
        if self.parser_profile_sha256 != DEFAULT_PARSER_PROFILE_SHA256:
            raise ValueError("parser_profile_sha256 does not match the default profile")
        if self.renderer_profile_sha256 != DEFAULT_RENDER_PROFILE_SHA256:
            raise ValueError(
                "renderer_profile_sha256 does not match the default profile"
            )
        reference = DEFAULT_REFERENCE_PROFILE
        if (
            self.pdf_sha256 != reference.fixture_sha256
            or self.pdf_size_bytes != reference.fixture_size_bytes
            or self.page_count != reference.page_count
            or self.file_version_id != reference.file_version_id
            or self.file_version_binding_sha256
            != reference.file_version_binding_sha256
        ):
            raise ValueError("report source identity does not match the reference")

        reported_versions = {
            key: getattr(self, key) for key in _runtime_tool_versions()
        }
        reference_versions = {
            "python_version": reference.python_version,
            "pdfminer_version": reference.pdfminer_version,
            "pdfplumber_version": reference.pdfplumber_version,
            "pypdfium2_version": reference.pypdfium2_version,
            "pdfium_version": reference.pdfium_version,
            "pillow_version": reference.pillow_version,
            "reportlab_version": reference.reportlab_version,
        }
        if reported_versions != reference_versions:
            raise ValueError("report tool versions do not match the reference profile")
        if reported_versions != _runtime_tool_versions():
            raise ValueError("report tool versions do not match the installed runtime")

        anchor = self.anchor
        if (
            anchor.source_pdf_sha256 != self.pdf_sha256
            or anchor.parser_profile_sha256 != self.parser_profile_sha256
            or anchor.file_version_id != self.file_version_id
            or anchor.file_version_binding_sha256
            != self.file_version_binding_sha256
            or anchor.physical_page_index != reference.physical_page_index
            or anchor.display_page_number != reference.display_page_number
            or anchor.printed_page_label != reference.printed_page_label
            or anchor.page_width_points != reference.page_width_points
            or anchor.page_height_points != reference.page_height_points
            or anchor.source_page_rotation_degrees
            != reference.source_page_rotation_degrees
            or anchor.canonical_page_text_sha256
            != reference.canonical_page_text_sha256
            or anchor.char_start != reference.char_start
            or anchor.char_end != reference.char_end
            or anchor.anchor_text != reference.needle
            or anchor.anchor_text_sha256 != reference.anchor_text_sha256
            or anchor.boxes_sha256 != reference.boxes_sha256
            or self.boxes_sha256 != anchor.boxes_sha256
            or anchor.evidence_id != reference.evidence_id
        ):
            raise ValueError("report anchor does not match the reference profile")

        render = self.render
        if (
            render.renderer_profile_id != reference.renderer_profile.profile_id
            or render.physical_page_index != anchor.physical_page_index
            or render.source_page_rotation_degrees
            != anchor.source_page_rotation_degrees
            or render.scale != reference.renderer_profile.scale
            or render.dpi != reference.renderer_profile.dpi
            or render.additional_rotation_degrees
            != reference.renderer_profile.additional_rotation_degrees
            or render.draw_annotations != reference.renderer_profile.draw_annots
            or render.draw_forms != reference.renderer_profile.may_draw_forms
            or render.background_rgba != reference.renderer_profile.fill_color
            or render.pixel_mode != reference.pixel_mode
            or render.pixel_width != reference.pixel_width
            or render.pixel_height != reference.pixel_height
            or render.rgba_byte_count != reference.rgba_byte_count
            or render.rgba_sha256 != reference.rgba_sha256
            or render.render_sha256 != reference.render_sha256
            or render.png_byte_count != reference.png_byte_count
            or render.png_sha256 != reference.png_sha256
        ):
            raise ValueError("report render does not match the reference profile")
        return self


def _load_canonical_hardware_facts(path: Path) -> tuple[HardwareFacts, str]:
    invalid_message = "Hardware facts file is not canonical."
    raw, _ = _load_raw_canonical_json_object(
        Path(path),
        invalid_message=invalid_message,
    )
    try:
        facts = HardwareFacts.model_validate_json(raw, strict=True)
        if raw != _canonical_json_file_bytes(facts.model_dump(mode="json")):
            raise ValueError("Hardware facts changed during strict validation.")
    except (ValidationError, ValueError) as error:
        raise PdfAnchorOperationalError(invalid_message) from error
    return facts, hashlib.sha256(raw[:-1]).hexdigest()


class AmbiguousAnchorError(PdfAnchorOperationalError):
    """Raised when an exact normalized needle has multiple PDF occurrences."""


@dataclass(frozen=True, slots=True)
class _PdfSnapshot:
    data: bytes
    size_bytes: int
    sha256: str


def _read_pdf_snapshot(source: Path) -> _PdfSnapshot:
    path = Path(source)
    try:
        with path.open("rb") as handle:
            data = handle.read()
    except OSError as error:
        raise PdfAnchorOperationalError("The source PDF could not be read.") from error
    if not data.startswith(b"%PDF-"):
        raise PdfAnchorOperationalError("The source file is not a PDF.")
    return _PdfSnapshot(
        data=data,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _normalize_anchor_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    joiner = chr(DEFAULT_PARSER_PROFILE.word_joiner_ascii_codepoint)
    return joiner.join(normalized.split())


def _round_pdf_value(value: float) -> float:
    if not math.isfinite(value):
        raise PdfAnchorOperationalError("PDF coordinates must be finite.")
    rounded = round(value, DEFAULT_PARSER_PROFILE.coordinate_decimal_places)
    return 0.0 if rounded == 0.0 else rounded


def _copy_packed_rgba(
    bitmap: pdfium.PdfBitmap,
    *,
    pixel_width: int,
    pixel_height: int,
) -> bytes:
    if bitmap.width != pixel_width or bitmap.height != pixel_height:
        raise PdfAnchorOperationalError(
            "PDFium bitmap dimensions do not match the render profile."
        )
    if bitmap.n_channels != 4:
        raise PdfAnchorOperationalError(
            "PDFium bitmap must contain exactly four channels."
        )
    row_byte_count = pixel_width * 4
    if bitmap.stride < row_byte_count:
        raise PdfAnchorOperationalError(
            "PDFium bitmap stride is shorter than one RGBA row."
        )

    buffer_view = memoryview(bitmap.buffer).cast("B")
    try:
        required_byte_count = bitmap.stride * pixel_height
        if len(buffer_view) < required_byte_count:
            raise PdfAnchorOperationalError(
                "PDFium bitmap buffer is shorter than its declared stride."
            )
        return b"".join(
            bytes(
                buffer_view[
                    row_index * bitmap.stride : row_index * bitmap.stride
                    + row_byte_count
                ]
            )
            for row_index in range(pixel_height)
        )
    finally:
        buffer_view.release()


def _encode_metadata_free_png(
    packed_rgba_bytes: bytes,
    *,
    pixel_width: int,
    pixel_height: int,
) -> bytes:
    with BytesIO() as output:
        image = Image.frombytes(
            DEFAULT_RENDER_PROFILE.png_mode,
            (pixel_width, pixel_height),
            packed_rgba_bytes,
        )
        try:
            image.save(
                output,
                format="PNG",
                optimize=DEFAULT_RENDER_PROFILE.png_optimize,
                compress_level=DEFAULT_RENDER_PROFILE.png_compress_level,
            )
            return output.getvalue()
        finally:
            image.close()


def _render_pdf_page_snapshot(
    snapshot: _PdfSnapshot,
    *,
    physical_page_index: int,
) -> RenderedPdfPage:
    if physical_page_index < 0:
        raise PdfAnchorOperationalError(
            "physical_page_index must be non-negative."
        )

    document = pdfium.PdfDocument(snapshot.data)
    page: pdfium.PdfPage | None = None
    bitmap: pdfium.PdfBitmap | None = None
    try:
        if physical_page_index >= len(document):
            raise PdfAnchorOperationalError(
                "physical_page_index is outside the PDF page range."
            )
        page = document[physical_page_index]
        page_width_points = _round_pdf_value(float(page.get_width()))
        page_height_points = _round_pdf_value(float(page.get_height()))
        source_page_rotation_degrees = int(page.get_rotation())
        pixel_width = math.ceil(page_width_points * DEFAULT_RENDER_PROFILE.scale)
        pixel_height = math.ceil(page_height_points * DEFAULT_RENDER_PROFILE.scale)
        bitmap = page.render(
            scale=DEFAULT_RENDER_PROFILE.scale,
            rotation=DEFAULT_RENDER_PROFILE.additional_rotation_degrees,
            crop=DEFAULT_RENDER_PROFILE.crop,
            may_draw_forms=DEFAULT_RENDER_PROFILE.may_draw_forms,
            fill_color=DEFAULT_RENDER_PROFILE.fill_color,
            draw_annots=DEFAULT_RENDER_PROFILE.draw_annots,
            force_bitmap_format=pdfium.raw.FPDFBitmap_BGRA,
            rev_byteorder=DEFAULT_RENDER_PROFILE.rev_byteorder,
            optimize_mode=DEFAULT_RENDER_PROFILE.optimize_mode,
            no_smoothtext=DEFAULT_RENDER_PROFILE.no_smoothtext,
            no_smoothimage=DEFAULT_RENDER_PROFILE.no_smoothimage,
            no_smoothpath=DEFAULT_RENDER_PROFILE.no_smoothpath,
            force_halftone=DEFAULT_RENDER_PROFILE.force_halftone,
            limit_image_cache=DEFAULT_RENDER_PROFILE.limit_image_cache,
            extra_flags=DEFAULT_RENDER_PROFILE.extra_flags,
            color_scheme=DEFAULT_RENDER_PROFILE.color_scheme,
            fill_to_stroke=DEFAULT_RENDER_PROFILE.fill_to_stroke,
        )
        packed_rgba_bytes = _copy_packed_rgba(
            bitmap,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
        )
        png_bytes = _encode_metadata_free_png(
            packed_rgba_bytes,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
        )
        rgba_sha256 = hashlib.sha256(packed_rgba_bytes).hexdigest()
        render_preimage = (
            DEFAULT_RENDER_PROFILE.raw_hash_domain_utf8.encode("utf-8")
            + pixel_width.to_bytes(8, "big")
            + pixel_height.to_bytes(8, "big")
            + packed_rgba_bytes
        )
        evidence = PdfRenderEvidence(
            physical_page_index=physical_page_index,
            source_page_rotation_degrees=source_page_rotation_degrees,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            rgba_byte_count=len(packed_rgba_bytes),
            rgba_sha256=rgba_sha256,
            render_sha256=hashlib.sha256(render_preimage).hexdigest(),
            png_byte_count=len(png_bytes),
            png_sha256=hashlib.sha256(png_bytes).hexdigest(),
        )
        return RenderedPdfPage(
            physical_page_index=physical_page_index,
            page_width_points=page_width_points,
            page_height_points=page_height_points,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            packed_rgba_bytes=packed_rgba_bytes,
            png_bytes=png_bytes,
            evidence=evidence,
        )
    finally:
        try:
            if bitmap is not None:
                bitmap.close()
        finally:
            try:
                if page is not None:
                    page.close()
            finally:
                document.close()


def render_pdf_page(
    source: Path,
    *,
    physical_page_index: int,
) -> RenderedPdfPage:
    """Render one physical page from a single immutable source snapshot."""

    snapshot = _read_pdf_snapshot(Path(source))
    return _render_pdf_page_snapshot(
        snapshot,
        physical_page_index=physical_page_index,
    )


@dataclass(frozen=True, slots=True)
class _CanonicalWord:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    vertical_key: float
    original_extraction_index: int
    char_start: int = 0
    char_end: int = 0


@dataclass(frozen=True, slots=True)
class _CanonicalLine:
    representative_vertical_key: float
    minimum_x0: float
    minimum_original_extraction_index: int
    x0: float
    top: float
    x1: float
    bottom: float
    text: str
    words: tuple[_CanonicalWord, ...]


@dataclass(frozen=True, slots=True)
class _CanonicalPage:
    canonical_text: str
    words: tuple[_CanonicalWord, ...]
    lines: tuple[_CanonicalLine, ...]
    printed_page_label: str | None


@dataclass(frozen=True, slots=True)
class _ParsedPdfPage:
    physical_page_index: int
    page_width_points: float
    page_height_points: float
    source_page_rotation_degrees: int
    canonical: _CanonicalPage


def _word_float(word: Mapping[str, object], key: str) -> float:
    value = word.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PdfAnchorOperationalError(f"Extracted word {key} must be numeric.")
    return _round_pdf_value(float(value))


def _canonicalize_extracted_words(
    extracted_words: Sequence[Mapping[str, object]],
    *,
    page_width_points: float,
    page_height_points: float,
) -> _CanonicalPage:
    """Apply the frozen spatial order, canonical text, and footer rules."""

    width = _round_pdf_value(float(page_width_points))
    height = _round_pdf_value(float(page_height_points))
    if width <= 0.0 or height <= 0.0:
        raise PdfAnchorOperationalError("PDF page dimensions must be positive.")

    candidates: list[_CanonicalWord] = []
    for original_index, extracted in enumerate(extracted_words):
        raw_text = extracted.get("text")
        if not isinstance(raw_text, str):
            raise PdfAnchorOperationalError(
                "Extracted word text must be a string."
            )
        text = _normalize_anchor_text(raw_text)
        if not text:
            continue
        x0 = _word_float(extracted, "x0")
        top = _word_float(extracted, "top")
        x1 = _word_float(extracted, "x1")
        bottom = _word_float(extracted, "bottom")
        if not 0.0 <= x0 < x1 <= width:
            raise PdfAnchorOperationalError(
                "Extracted word horizontal coordinates must be page-bound."
            )
        if not 0.0 <= top < bottom <= height:
            raise PdfAnchorOperationalError(
                "Extracted word vertical coordinates must be page-bound."
            )
        vertical_key = _round_pdf_value((top + bottom) / 2.0)
        candidates.append(
            _CanonicalWord(
                text=text,
                x0=x0,
                top=top,
                x1=x1,
                bottom=bottom,
                vertical_key=vertical_key,
                original_extraction_index=original_index,
            )
        )

    candidates.sort(
        key=lambda word: (
            word.vertical_key,
            word.x0,
            word.top,
            word.original_extraction_index,
        )
    )
    line_groups: list[tuple[float, list[_CanonicalWord]]] = []
    for word in candidates:
        if (
            not line_groups
            or abs(word.vertical_key - line_groups[-1][0])
            > DEFAULT_PARSER_PROFILE.line_cluster_tolerance_points
        ):
            line_groups.append((word.vertical_key, [word]))
        else:
            line_groups[-1][1].append(word)

    line_groups.sort(
        key=lambda group: (
            group[0],
            min(word.x0 for word in group[1]),
            min(word.original_extraction_index for word in group[1]),
        )
    )

    canonical_words: list[_CanonicalWord] = []
    canonical_lines: list[_CanonicalLine] = []
    joiner = chr(DEFAULT_PARSER_PROFILE.word_joiner_ascii_codepoint)
    cursor = 0
    for representative, group in line_groups:
        group.sort(
            key=lambda word: (
                word.x0,
                word.top,
                word.original_extraction_index,
            )
        )
        line_words: list[_CanonicalWord] = []
        for word in group:
            if canonical_words:
                cursor += 1
            ranged_word = replace(
                word,
                char_start=cursor,
                char_end=cursor + len(word.text),
            )
            cursor = ranged_word.char_end
            canonical_words.append(ranged_word)
            line_words.append(ranged_word)
        canonical_lines.append(
            _CanonicalLine(
                representative_vertical_key=representative,
                minimum_x0=min(word.x0 for word in line_words),
                minimum_original_extraction_index=min(
                    word.original_extraction_index for word in line_words
                ),
                x0=min(word.x0 for word in line_words),
                top=min(word.top for word in line_words),
                x1=max(word.x1 for word in line_words),
                bottom=max(word.bottom for word in line_words),
                text=joiner.join(word.text for word in line_words),
                words=tuple(line_words),
            )
        )

    canonical_text = joiner.join(word.text for word in canonical_words)
    footer_candidates = tuple(
        line.text
        for line in canonical_lines
        if line.top >= height - DEFAULT_PARSER_PROFILE.footer_band_points
        and abs(((line.x0 + line.x1) / 2.0) - (width / 2.0))
        <= DEFAULT_PARSER_PROFILE.footer_center_tolerance_points
        and _FOOTER_LABEL_PATTERN.fullmatch(line.text) is not None
    )
    printed_page_label = footer_candidates[0] if len(footer_candidates) == 1 else None
    return _CanonicalPage(
        canonical_text=canonical_text,
        words=tuple(canonical_words),
        lines=tuple(canonical_lines),
        printed_page_label=printed_page_label,
    )


def _extract_pdf_pages(snapshot: _PdfSnapshot) -> tuple[_ParsedPdfPage, ...]:
    extracted_pages: list[_ParsedPdfPage] = []
    with BytesIO(snapshot.data) as stream:
        with pdfplumber.open(stream) as document:
            for physical_page_index, page in enumerate(document.pages):
                width = _round_pdf_value(float(page.width))
                height = _round_pdf_value(float(page.height))
                raw_words = page.extract_words(
                    x_tolerance=DEFAULT_PARSER_PROFILE.x_tolerance,
                    y_tolerance=DEFAULT_PARSER_PROFILE.y_tolerance,
                    x_tolerance_ratio=DEFAULT_PARSER_PROFILE.x_tolerance_ratio,
                    y_tolerance_ratio=DEFAULT_PARSER_PROFILE.y_tolerance_ratio,
                    keep_blank_chars=DEFAULT_PARSER_PROFILE.keep_blank_chars,
                    use_text_flow=DEFAULT_PARSER_PROFILE.use_text_flow,
                    line_dir=DEFAULT_PARSER_PROFILE.line_dir,
                    char_dir=DEFAULT_PARSER_PROFILE.char_dir,
                    split_at_punctuation=(
                        DEFAULT_PARSER_PROFILE.split_at_punctuation
                    ),
                    expand_ligatures=DEFAULT_PARSER_PROFILE.expand_ligatures,
                    return_chars=DEFAULT_PARSER_PROFILE.return_chars,
                )
                canonical = _canonicalize_extracted_words(
                    raw_words,
                    page_width_points=width,
                    page_height_points=height,
                )
                extracted_pages.append(
                    _ParsedPdfPage(
                        physical_page_index=physical_page_index,
                        page_width_points=width,
                        page_height_points=height,
                        source_page_rotation_degrees=int(page.rotation or 0),
                        canonical=canonical,
                    )
                )
    return tuple(extracted_pages)


def _overlapping_occurrences(text: str, needle: str) -> tuple[int, ...]:
    occurrences: list[int] = []
    search_start = 0
    while True:
        occurrence = text.find(needle, search_start)
        if occurrence < 0:
            return tuple(occurrences)
        occurrences.append(occurrence)
        search_start = occurrence + 1


def _locate_snapshot_with_page_count(
    snapshot: _PdfSnapshot,
    *,
    file_version_id: str,
    needle: str,
) -> tuple[NativePdfAnchor | None, int]:
    if not file_version_id.strip():
        raise PdfAnchorOperationalError("file_version_id must not be empty.")
    normalized_needle = _normalize_anchor_text(needle)
    if not normalized_needle:
        raise PdfAnchorOperationalError(
            "The normalized anchor text must not be empty."
        )

    pages = _extract_pdf_pages(snapshot)
    occurrences: list[tuple[_ParsedPdfPage, int]] = []
    for page in pages:
        occurrences.extend(
            (page, start)
            for start in _overlapping_occurrences(
                page.canonical.canonical_text,
                normalized_needle,
            )
        )
    if not occurrences:
        return None, len(pages)
    if len(occurrences) > 1:
        raise AmbiguousAnchorError(
            "The normalized anchor text matched more than once."
        )

    page, char_start = occurrences[0]
    char_end = char_start + len(normalized_needle)
    ordered_boxes = tuple(
        PdfAnchorBox(
            char_start=word.char_start,
            char_end=word.char_end,
            x0=word.x0,
            top=word.top,
            x1=word.x1,
            bottom=word.bottom,
        )
        for word in page.canonical.words
        if word.char_end > char_start and word.char_start < char_end
    )
    boxes_sha256 = _canonical_sha256(
        [box.model_dump(mode="json") for box in ordered_boxes]
    )
    anchor_text = page.canonical.canonical_text[char_start:char_end]
    anchor_text_sha256 = _text_sha256(anchor_text)
    binding_sha256 = _canonical_sha256(
        {
            "file_version_id": file_version_id,
            "pdf_sha256": snapshot.sha256,
        }
    )
    identity_payload = {
        "file_version_id": file_version_id,
        "pdf_sha256": snapshot.sha256,
        "parser_profile_sha256": DEFAULT_PARSER_PROFILE_SHA256,
        "physical_page_index": page.physical_page_index,
        "char_start": char_start,
        "char_end": char_end,
        "anchor_text_sha256": anchor_text_sha256,
        "boxes_sha256": boxes_sha256,
    }
    anchor = NativePdfAnchor(
        evidence_id="ev-sha256-" + _canonical_sha256(identity_payload),
        file_version_id=file_version_id,
        file_version_binding_sha256=binding_sha256,
        source_pdf_sha256=snapshot.sha256,
        parser_profile_sha256=DEFAULT_PARSER_PROFILE_SHA256,
        physical_page_index=page.physical_page_index,
        display_page_number=page.physical_page_index + 1,
        printed_page_label=page.canonical.printed_page_label,
        page_width_points=page.page_width_points,
        page_height_points=page.page_height_points,
        source_page_rotation_degrees=page.source_page_rotation_degrees,
        char_start=char_start,
        char_end=char_end,
        canonical_page_text=page.canonical.canonical_text,
        canonical_page_text_sha256=_text_sha256(page.canonical.canonical_text),
        anchor_text=anchor_text,
        anchor_text_sha256=anchor_text_sha256,
        boxes=ordered_boxes,
        boxes_sha256=boxes_sha256,
    )
    return anchor, len(pages)


def _locate_snapshot(
    snapshot: _PdfSnapshot,
    *,
    file_version_id: str,
    needle: str,
) -> NativePdfAnchor | None:
    anchor, _ = _locate_snapshot_with_page_count(
        snapshot,
        file_version_id=file_version_id,
        needle=needle,
    )
    return anchor


class PdfPlumberAnchorLocator:
    """Locate one exact native-text anchor with the committed parser profile."""

    def locate(
        self,
        source: Path,
        *,
        file_version_id: str,
        needle: str,
    ) -> NativePdfAnchor | None:
        snapshot = _read_pdf_snapshot(Path(source))
        return _locate_snapshot(
            snapshot,
            file_version_id=file_version_id,
            needle=needle,
        )


def _validate_rendered_pdf_page(rendered: RenderedPdfPage) -> PdfRenderEvidence:
    try:
        evidence = PdfRenderEvidence.model_validate(
            rendered.evidence.model_dump(mode="python")
        )
    except ValidationError as error:
        raise PdfAnchorOperationalError(
            "Rendered PDF bytes do not match the render evidence."
        ) from error

    packed = rendered.packed_rgba_bytes
    png = rendered.png_bytes
    render_preimage = (
        DEFAULT_RENDER_PROFILE.raw_hash_domain_utf8.encode("utf-8")
        + rendered.pixel_width.to_bytes(8, "big")
        + rendered.pixel_height.to_bytes(8, "big")
        + packed
    )
    if (
        rendered.physical_page_index != evidence.physical_page_index
        or rendered.pixel_width != evidence.pixel_width
        or rendered.pixel_height != evidence.pixel_height
        or len(packed) != evidence.rgba_byte_count
        or hashlib.sha256(packed).hexdigest() != evidence.rgba_sha256
        or hashlib.sha256(render_preimage).hexdigest() != evidence.render_sha256
        or len(png) != evidence.png_byte_count
        or hashlib.sha256(png).hexdigest() != evidence.png_sha256
    ):
        raise PdfAnchorOperationalError(
            "Rendered PDF bytes do not match the render evidence."
        )
    return evidence


def create_pdf_anchor_report(
    *,
    source: Path,
    file_version_id: str,
    needle: str,
    hardware_facts_path: Path,
    measured_at_utc: str | None = None,
) -> PdfAnchorReport:
    """Build verified success evidence entirely from one immutable PDF snapshot."""

    measurement_time = (
        datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
        if measured_at_utc is None
        else _validate_utc_timestamp(measured_at_utc)
    )
    if file_version_id != DEFAULT_REFERENCE_PROFILE.file_version_id:
        raise PdfAnchorOperationalError(
            "file_version_id does not match the reference profile."
        )
    if needle != DEFAULT_REFERENCE_PROFILE.needle:
        normalized_needle = _normalize_anchor_text(needle)
        if not normalized_needle:
            raise PdfAnchorOperationalError(
                "The normalized anchor text must not be empty."
            )

    _, hardware_facts_sha256 = _load_canonical_hardware_facts(
        Path(hardware_facts_path)
    )
    versions = _runtime_tool_versions()
    expected_versions = {
        "python_version": DEFAULT_REFERENCE_PROFILE.python_version,
        "pdfminer_version": DEFAULT_REFERENCE_PROFILE.pdfminer_version,
        "pdfplumber_version": DEFAULT_REFERENCE_PROFILE.pdfplumber_version,
        "pypdfium2_version": DEFAULT_REFERENCE_PROFILE.pypdfium2_version,
        "pdfium_version": DEFAULT_REFERENCE_PROFILE.pdfium_version,
        "pillow_version": DEFAULT_REFERENCE_PROFILE.pillow_version,
        "reportlab_version": DEFAULT_REFERENCE_PROFILE.reportlab_version,
    }
    if versions != expected_versions:
        raise PdfAnchorOperationalError(
            "Installed PDF tool versions do not match the reference profile."
        )

    snapshot = _read_pdf_snapshot(Path(source))
    if (
        snapshot.size_bytes != DEFAULT_REFERENCE_PROFILE.fixture_size_bytes
        or snapshot.sha256 != DEFAULT_REFERENCE_PROFILE.fixture_sha256
    ):
        raise PdfAnchorOperationalError(
            "The source PDF does not match the reference profile."
        )

    anchor, page_count = _locate_snapshot_with_page_count(
        snapshot,
        file_version_id=file_version_id,
        needle=needle,
    )
    if anchor is None:
        raise PdfAnchorOperationalError(
            "The normalized anchor text was not found."
        )
    if needle != DEFAULT_REFERENCE_PROFILE.needle:
        raise PdfAnchorOperationalError(
            "The requested anchor text does not exactly match the reference profile."
        )
    rendered = _render_pdf_page_snapshot(
        snapshot,
        physical_page_index=anchor.physical_page_index,
    )
    evidence = _validate_rendered_pdf_page(rendered)
    if (
        rendered.page_width_points != anchor.page_width_points
        or rendered.page_height_points != anchor.page_height_points
    ):
        raise PdfAnchorOperationalError(
            "Rendered PDF page dimensions do not match the text anchor."
        )

    unsigned: dict[str, object] = {
        "schema_version": "1.0.0",
        "report_type": "pdf_anchor",
        "artifact_kind": "deterministic_correctness",
        "verification_status": "verified",
        "measured_at_utc": measurement_time,
        "profile": DEFAULT_REFERENCE_PROFILE,
        "profile_sha256": DEFAULT_REFERENCE_PROFILE_SHA256,
        "parser_profile_sha256": DEFAULT_PARSER_PROFILE_SHA256,
        "renderer_profile_sha256": DEFAULT_RENDER_PROFILE_SHA256,
        "reference_profile_verified": True,
        "pdf_sha256": snapshot.sha256,
        "pdf_size_bytes": snapshot.size_bytes,
        "page_count": page_count,
        "file_version_id": file_version_id,
        "file_version_binding_sha256": anchor.file_version_binding_sha256,
        "hardware_facts_sha256": hardware_facts_sha256,
        **versions,
        "anchor": anchor,
        "boxes_sha256": anchor.boxes_sha256,
        "render": evidence,
        "anchor_integrity_verified": True,
        "render_integrity_verified": True,
    }
    unsigned_json = {
        key: (
            value.model_dump(mode="json")
            if isinstance(value, BaseModel)
            else value
        )
        for key, value in unsigned.items()
    }
    return PdfAnchorReport.model_validate(
        {**unsigned, "report_sha256": _canonical_sha256(unsigned_json)}
    )


def _python_mapping(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise PdfAnchorOperationalError("PDF anchor report is not valid.")
    return cast(dict[str, object], value)


def _require_exact_float_fields(
    payload: Mapping[str, object],
    fields: tuple[str, ...],
) -> None:
    if any(type(payload.get(field)) is not float for field in fields):
        raise PdfAnchorOperationalError("PDF anchor report is not valid.")


def _require_original_report_python_types(payload: Mapping[str, object]) -> None:
    profile = _python_mapping(payload.get("profile"))
    parser_profile = _python_mapping(profile.get("parser_profile"))
    renderer_profile = _python_mapping(profile.get("renderer_profile"))
    anchor = _python_mapping(payload.get("anchor"))
    render = _python_mapping(payload.get("render"))

    _require_exact_float_fields(
        parser_profile,
        (
            "x_tolerance",
            "y_tolerance",
            "line_cluster_tolerance_points",
            "footer_band_points",
            "footer_center_tolerance_points",
        ),
    )
    _require_exact_float_fields(renderer_profile, ("scale",))
    _require_exact_float_fields(
        profile,
        ("page_width_points", "page_height_points"),
    )
    _require_exact_float_fields(
        anchor,
        ("page_width_points", "page_height_points"),
    )
    _require_exact_float_fields(render, ("scale",))

    tuple_fields = (
        (parser_profile, "candidate_sort_keys"),
        (parser_profile, "line_sort_keys"),
        (parser_profile, "word_sort_keys"),
        (renderer_profile, "crop"),
        (renderer_profile, "fill_color"),
        (anchor, "boxes"),
        (render, "background_rgba"),
    )
    if any(type(container.get(field)) is not tuple for container, field in tuple_fields):
        raise PdfAnchorOperationalError("PDF anchor report is not valid.")

    boxes = cast(tuple[object, ...], anchor["boxes"])
    for box_value in boxes:
        box = _python_mapping(box_value)
        _require_exact_float_fields(box, ("x0", "top", "x1", "bottom"))


def _revalidate_pdf_anchor_report(report: PdfAnchorReport) -> PdfAnchorReport:
    try:
        original_payload = report.model_dump(mode="python", warnings="error")
        supplied_hash = original_payload.get("report_sha256")
        unsigned = {
            key: value
            for key, value in original_payload.items()
            if key != "report_sha256"
        }
        expected_hash = _canonical_sha256(unsigned)
        if not isinstance(supplied_hash, str) or not hmac.compare_digest(
            supplied_hash,
            expected_hash,
        ):
            raise PdfAnchorOperationalError("PDF anchor report is not valid.")

        original_canonical = _canonical_json_bytes(original_payload)
        _require_original_report_python_types(original_payload)
        validated = PdfAnchorReport.model_validate(original_payload, strict=True)
        validated_payload = validated.model_dump(mode="python", warnings="error")
        if original_canonical != _canonical_json_bytes(validated_payload):
            raise PdfAnchorOperationalError("PDF anchor report is not valid.")
        return validated
    except PdfAnchorOperationalError:
        raise
    except (
        AttributeError,
        TypeError,
        UnicodeError,
        ValidationError,
        ValueError,
        PydanticSerializationError,
    ) as error:
        raise PdfAnchorOperationalError("PDF anchor report is not valid.") from error


def load_pdf_anchor_report(path: Path) -> PdfAnchorReport:
    """Load canonical report bytes, checking their raw self-hash first."""

    raw, payload = _load_raw_canonical_json_object(
        Path(path),
        invalid_message="PDF anchor report file is not canonical.",
    )
    supplied_hash = payload.get("report_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "report_sha256"
    }
    expected_hash = _canonical_sha256(unsigned)
    if not isinstance(supplied_hash, str) or not hmac.compare_digest(
        supplied_hash,
        expected_hash,
    ):
        raise PdfAnchorOperationalError(
            "report_sha256 does not match the raw canonical report payload."
        )
    try:
        report = PdfAnchorReport.model_validate_json(raw, strict=True)
    except ValidationError as error:
        raise PdfAnchorOperationalError(
            "PDF anchor report file is not valid."
        ) from error
    if raw != _canonical_json_file_bytes(report.model_dump(mode="json")):
        raise PdfAnchorOperationalError(
            "PDF anchor report file is not canonical."
        )
    return report


def write_pdf_anchor_report(path: Path, report: PdfAnchorReport) -> None:
    """Atomically publish a fully revalidated canonical report."""

    validated = _revalidate_pdf_anchor_report(report)
    output = Path(path)
    parent = output.parent
    if not parent.is_dir():
        raise PdfAnchorOperationalError(
            "Output parent directory does not exist."
        )
    encoded = _canonical_json_file_bytes(validated.model_dump(mode="json"))

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    handle: BinaryIO | None = None
    try:
        handle = os.fdopen(fd, "wb")
        fd = -1
        written_byte_count = handle.write(encoded)
        if written_byte_count != len(encoded):
            raise PdfAnchorOperationalError(
                "PDF anchor report write was incomplete."
            )
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(temporary, output)
    except BaseException as primary_error:
        try:
            if handle is not None:
                handle.close()
            elif fd >= 0:
                os.close(fd)
        except BaseException as close_error:
            primary_error.add_note(
                "Temporary report handle cleanup failed "
                f"({type(close_error).__name__})."
            )

        unlink_error: BaseException | None = None
        for _ in range(2):
            try:
                temporary.unlink(missing_ok=True)
            except BaseException as cleanup_error:
                unlink_error = cleanup_error
            else:
                unlink_error = None
                break
        if unlink_error is not None:
            primary_error.add_note(
                "Temporary report cleanup failed after two attempts "
                f"({type(unlink_error).__name__})."
            )
        raise


def verify_pdf_anchor_replay(
    *,
    source: Path,
    report: PdfAnchorReport,
) -> None:
    """Replay locator and renderer checks without writing any artifact."""

    validated = _revalidate_pdf_anchor_report(report)
    snapshot = _read_pdf_snapshot(Path(source))
    if (
        snapshot.sha256 != validated.pdf_sha256
        or snapshot.size_bytes != validated.pdf_size_bytes
    ):
        raise PdfAnchorOperationalError(
            "PDF anchor replay source does not match the report."
        )

    try:
        anchor, page_count = _locate_snapshot_with_page_count(
            snapshot,
            file_version_id=validated.file_version_id,
            needle=validated.anchor.anchor_text,
        )
    except AmbiguousAnchorError as error:
        raise PdfAnchorOperationalError(
            "PDF anchor replay text anchor does not match the report."
        ) from error
    if (
        anchor is None
        or page_count != validated.page_count
        or anchor.model_dump(mode="json")
        != validated.anchor.model_dump(mode="json")
    ):
        raise PdfAnchorOperationalError(
            "PDF anchor replay text anchor does not match the report."
        )

    rendered = _render_pdf_page_snapshot(
        snapshot,
        physical_page_index=anchor.physical_page_index,
    )
    try:
        evidence = _validate_rendered_pdf_page(rendered)
    except PdfAnchorOperationalError as error:
        raise PdfAnchorOperationalError(
            "PDF anchor replay render does not match the report."
        ) from error
    if (
        rendered.page_width_points != anchor.page_width_points
        or rendered.page_height_points != anchor.page_height_points
        or evidence.model_dump(mode="json")
        != validated.render.model_dump(mode="json")
    ):
        raise PdfAnchorOperationalError(
            "PDF anchor replay render does not match the report."
        )


class _ArgumentParsingError(Exception):
    pass


class _QuietArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _ArgumentParsingError(message)


def _resolved_path_identity(path: Path) -> str:
    return str(path).replace("\\", "/").casefold()


def _resolve_cli_path(raw_path: str) -> Path:
    try:
        return Path(raw_path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise PdfAnchorOperationalError(
            "Input/output paths could not be resolved."
        ) from error


def _output_aliases_input(output: Path, input_path: Path) -> bool:
    if _resolved_path_identity(output) == _resolved_path_identity(input_path):
        return True
    try:
        return (
            output.exists()
            and input_path.exists()
            and os.path.samefile(output, input_path)
        )
    except OSError as error:
        raise PdfAnchorOperationalError(
            "Input/output path identity could not be checked."
        ) from error


def _write_cli_error(message: str) -> None:
    stable_line = " ".join(message.splitlines()).strip()
    sys.stderr.write((stable_line or "PDF anchor operation failed.") + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic PDF-anchor report CLI without leaking SystemExit."""

    parser = _QuietArgumentParser(add_help=False)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--file-version-id", required=True)
    parser.add_argument("--needle", required=True)
    parser.add_argument("--hardware-facts", required=True)
    parser.add_argument("--output", required=True)
    try:
        arguments = parser.parse_args(None if argv is None else list(argv))
    except _ArgumentParsingError:
        _write_cli_error("Invalid command arguments.")
        return 2

    try:
        source = _resolve_cli_path(arguments.pdf)
        hardware = _resolve_cli_path(arguments.hardware_facts)
        output = _resolve_cli_path(arguments.output)
        if _output_aliases_input(output, source) or _output_aliases_input(
            output,
            hardware,
        ):
            raise PdfAnchorOperationalError(
                "Output path must not alias an input path."
            )
        if not output.parent.is_dir():
            raise PdfAnchorOperationalError(
                "Output parent directory does not exist."
            )
        report = create_pdf_anchor_report(
            source=source,
            file_version_id=arguments.file_version_id,
            needle=arguments.needle,
            hardware_facts_path=hardware,
        )
        write_pdf_anchor_report(output, report)
    except PdfAnchorOperationalError as error:
        _write_cli_error(str(error))
        return 1
    except ValidationError:
        _write_cli_error("PDF anchor operation failed.")
        return 1
    except (PDFException, pdfium.PdfiumError, OSError):
        _write_cli_error("PDF anchor operation failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
