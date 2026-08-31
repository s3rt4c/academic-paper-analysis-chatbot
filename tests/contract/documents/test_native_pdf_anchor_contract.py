from __future__ import annotations

import pytest

from academic_chatbot.documents.native_pdf import build_native_pdf_anchor
from academic_chatbot.documents.normalization import (
    NativePdfNormalizationError,
    canonicalize_extracted_words,
)


def test_repeated_words_keep_their_constructed_occurrence_offsets_in_anchors() -> None:
    """Would fail if anchors rediscovered repeated token positions with a text search."""
    canonical = canonicalize_extracted_words(
        (
            {"text": "same", "x0": 10.0, "top": 10.0, "x1": 30.0, "bottom": 20.0},
            {"text": "same", "x0": 40.0, "top": 10.0, "x1": 60.0, "bottom": 20.0},
        ),
        page_width_points=100.0,
        page_height_points=100.0,
    )

    anchors = tuple(
        build_native_pdf_anchor(
            file_version_id="fv-paper-sha256-test",
            source_pdf_sha256="a" * 64,
            physical_page_index=0,
            page_width_points=100.0,
            page_height_points=100.0,
            source_page_rotation_degrees=0,
            canonical_page=canonical,
            word=word,
        )
        for word in canonical.words
    )

    assert canonical.text == "same same"
    assert [(anchor.char_start, anchor.char_end) for anchor in anchors] == [(0, 4), (5, 9)]
    assert [anchor.anchor_text for anchor in anchors] == ["same", "same"]
    assert anchors[0].evidence_id != anchors[1].evidence_id
    assert all(
        anchor.canonical_page_text[anchor.char_start : anchor.char_end] == anchor.anchor_text
        for anchor in anchors
    )


def test_canonicalization_preserves_nfc_content_and_uses_phase_zero_ordering() -> None:
    """Would fail if normalization cleaned typography or coordinates lost deterministic ordering."""
    canonical = canonicalize_extracted_words(
        (
            {"text": "B", "x0": 40.0, "top": 10.0, "x1": 45.0, "bottom": 20.0},
            {"text": "A\u2003", "x0": 10.0, "top": 10.0, "x1": 15.0, "bottom": 20.0},
            {"text": "co\u00adoperate", "x0": 10.0, "top": 40.0, "x1": 50.0, "bottom": 50.0},
        ),
        page_width_points=100.0,
        page_height_points=100.0,
    )

    assert canonical.text == "A B co\u00adoperate"
    assert [(word.char_start, word.char_end) for word in canonical.words] == [
        (0, 1),
        (2, 3),
        (4, 14),
    ]


def test_invalid_geometry_is_rejected_instead_of_becoming_plausible_evidence() -> None:
    """Would fail if non-finite or inverted source geometry silently reached anchors."""
    with pytest.raises(NativePdfNormalizationError, match="finite"):
        canonicalize_extracted_words(
            (
                {
                    "text": "bad",
                    "x0": float("nan"),
                    "top": 10.0,
                    "x1": 20.0,
                    "bottom": 20.0,
                },
            ),
            page_width_points=100.0,
            page_height_points=100.0,
        )

    with pytest.raises(NativePdfNormalizationError, match="horizontal"):
        canonicalize_extracted_words(
            (
                {"text": "bad", "x0": 30.0, "top": 10.0, "x1": 20.0, "bottom": 20.0},
            ),
            page_width_points=100.0,
            page_height_points=100.0,
        )


def test_ambiguous_printed_page_labels_are_intentionally_omitted() -> None:
    """Would fail if a footer heuristic fabricated one label from several candidates."""
    canonical = canonicalize_extracted_words(
        (
            {"text": "A-6", "x0": 42.0, "top": 28.0, "x1": 58.0, "bottom": 38.0},
            {"text": "B-7", "x0": 42.0, "top": 42.0, "x1": 58.0, "bottom": 52.0},
        ),
        page_width_points=100.0,
        page_height_points=100.0,
    )

    assert canonical.printed_page_label is None
