"""Deterministic native-PDF text normalization and spatial ordering."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real

from academic_chatbot.documents.models import CanonicalPdfPage, CanonicalPdfWord

_FOOTER_LABEL_PATTERN = re.compile(r"^[A-Z]+-[0-9]+$")
_COORDINATE_DECIMAL_PLACES = 6
_LINE_CLUSTER_TOLERANCE_POINTS = 3.0
_FOOTER_BAND_POINTS = 72.0
_FOOTER_CENTER_TOLERANCE_POINTS = 18.0

_PDFPLUMBER_EXTRACT_WORDS_KWARGS: dict[str, object] = {
    "x_tolerance": 3.0,
    "y_tolerance": 3.0,
    "x_tolerance_ratio": None,
    "y_tolerance_ratio": None,
    "keep_blank_chars": False,
    "use_text_flow": False,
    "line_dir": "ttb",
    "char_dir": "ltr",
    "split_at_punctuation": False,
    "expand_ligatures": True,
    "return_chars": False,
}

_PARSER_PROFILE_PAYLOAD = {
    "profile_id": "pdfplumber-native-evidence-v1",
    "pdfplumber_version": "0.11.10",
    "pdfplumber_profile": _PDFPLUMBER_EXTRACT_WORDS_KWARGS,
    "normalization": {
        "unicode_normalization": "NFC",
        "whitespace": "maximal-unicode-runs-to-ascii-space-and-trim",
        "preserve_case": True,
        "preserve_punctuation": True,
        "preserve_symbols": True,
        "dehyphenate": False,
        "empty_normalized_tokens": "discard",
    },
    "ordering": {
        "coordinate_decimal_places": _COORDINATE_DECIMAL_PLACES,
        "negative_zero": "canonicalize-to-positive-zero",
        "candidate_sort": ["vertical_key", "x0", "top", "original_index"],
        "line_cluster_tolerance_points": _LINE_CLUSTER_TOLERANCE_POINTS,
        "word_sort": ["x0", "top", "original_index"],
    },
    "printed_label": {
        "candidate_regex": r"^[A-Z]+-[0-9]+$",
        "footer_band_points": _FOOTER_BAND_POINTS,
        "center_tolerance_points": _FOOTER_CENTER_TOLERANCE_POINTS,
        "cardinality": "exactly-one",
    },
}

PARSER_PROFILE_SHA256 = hashlib.sha256(
    json.dumps(
        _PARSER_PROFILE_PAYLOAD,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


class NativePdfNormalizationError(ValueError):
    """Raised when extracted native text or geometry cannot become evidence."""


@dataclass(frozen=True, slots=True)
class _CandidateWord:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    vertical_key: float
    original_index: int
    char_start: int = 0
    char_end: int = 0


@dataclass(frozen=True, slots=True)
class _CanonicalLine:
    representative_vertical_key: float
    x0: float
    top: float
    x1: float
    bottom: float
    text: str
    words: tuple[_CandidateWord, ...]


def normalize_native_text(text: str) -> str:
    """Apply the frozen Phase 1A NFC and Unicode-whitespace text profile."""

    return " ".join(unicodedata.normalize("NFC", text).split())


def pdfplumber_extract_words_kwargs() -> dict[str, object]:
    """Return the locked in-process pdfplumber native-text extraction options."""

    return dict(_PDFPLUMBER_EXTRACT_WORDS_KWARGS)


def canonicalize_extracted_words(
    extracted_words: Sequence[Mapping[str, object]],
    *,
    page_width_points: float,
    page_height_points: float,
) -> CanonicalPdfPage:
    """Build page text and offsets directly from deterministic word construction."""

    width = _round_pdf_value(page_width_points)
    height = _round_pdf_value(page_height_points)
    if width <= 0.0 or height <= 0.0:
        raise NativePdfNormalizationError("PDF page dimensions must be positive")

    candidates: list[_CandidateWord] = []
    for original_index, extracted in enumerate(extracted_words):
        raw_text = extracted.get("text")
        if not isinstance(raw_text, str):
            raise NativePdfNormalizationError("extracted word text must be a string")
        text = normalize_native_text(raw_text)
        if not text:
            continue
        x0 = _word_coordinate(extracted, "x0")
        top = _word_coordinate(extracted, "top")
        x1 = _word_coordinate(extracted, "x1")
        bottom = _word_coordinate(extracted, "bottom")
        if not 0.0 <= x0 < x1 <= width:
            raise NativePdfNormalizationError("extracted word horizontal coordinates exceed page")
        if not 0.0 <= top < bottom <= height:
            raise NativePdfNormalizationError("extracted word vertical coordinates exceed page")
        candidates.append(
            _CandidateWord(
                text=text,
                x0=x0,
                top=top,
                x1=x1,
                bottom=bottom,
                vertical_key=_round_pdf_value((top + bottom) / 2.0),
                original_index=original_index,
            )
        )

    candidates.sort(
        key=lambda word: (word.vertical_key, word.x0, word.top, word.original_index)
    )
    grouped_lines: list[tuple[float, list[_CandidateWord]]] = []
    for word in candidates:
        if (
            not grouped_lines
            or abs(word.vertical_key - grouped_lines[-1][0]) > _LINE_CLUSTER_TOLERANCE_POINTS
        ):
            grouped_lines.append((word.vertical_key, [word]))
        else:
            grouped_lines[-1][1].append(word)

    grouped_lines.sort(
        key=lambda group: (
            group[0],
            min(word.x0 for word in group[1]),
            min(word.original_index for word in group[1]),
        )
    )
    words: list[_CandidateWord] = []
    lines: list[_CanonicalLine] = []
    cursor = 0
    for representative, group in grouped_lines:
        group.sort(key=lambda word: (word.x0, word.top, word.original_index))
        line_words: list[_CandidateWord] = []
        for word in group:
            if words:
                cursor += 1
            ranged_word = replace(word, char_start=cursor, char_end=cursor + len(word.text))
            cursor = ranged_word.char_end
            words.append(ranged_word)
            line_words.append(ranged_word)
        lines.append(
            _CanonicalLine(
                representative_vertical_key=representative,
                x0=min(word.x0 for word in line_words),
                top=min(word.top for word in line_words),
                x1=max(word.x1 for word in line_words),
                bottom=max(word.bottom for word in line_words),
                text=" ".join(word.text for word in line_words),
                words=tuple(line_words),
            )
        )

    printed_page_label = _printed_page_label(lines, page_width=width, page_height=height)
    return CanonicalPdfPage(
        text=" ".join(word.text for word in words),
        words=tuple(
            CanonicalPdfWord(
                text=word.text,
                char_start=word.char_start,
                char_end=word.char_end,
                x0=word.x0,
                top=word.top,
                x1=word.x1,
                bottom=word.bottom,
            )
            for word in words
        ),
        printed_page_label=printed_page_label,
    )


def _round_pdf_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise NativePdfNormalizationError("PDF coordinates must be finite numbers")
    rounded = round(float(value), _COORDINATE_DECIMAL_PLACES)
    return 0.0 if rounded == 0.0 else rounded


def _word_coordinate(extracted: Mapping[str, object], name: str) -> float:
    if name not in extracted:
        raise NativePdfNormalizationError(f"extracted word is missing {name}")
    return _round_pdf_value(extracted[name])


def _printed_page_label(
    lines: Sequence[_CanonicalLine], *, page_width: float, page_height: float
) -> str | None:
    candidates = tuple(
        line.text
        for line in lines
        if line.top >= page_height - _FOOTER_BAND_POINTS
        and abs(((line.x0 + line.x1) / 2.0) - (page_width / 2.0))
        <= _FOOTER_CENTER_TOLERANCE_POINTS
        and _FOOTER_LABEL_PATTERN.fullmatch(line.text) is not None
    )
    return candidates[0] if len(candidates) == 1 else None
