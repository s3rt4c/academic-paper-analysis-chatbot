"""Generate the deterministic native-text PDF fixture used by Phase 0."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

ANCHOR_SENTENCE = "The anchor sentence reports an accuracy of 91.2 percent."
FIXTURE_PROFILE_ID = "reportlab-native-anchor-v1"


def _configure_metadata(pdf: Canvas) -> None:
    pdf.setTitle("Phase 0 Native PDF Anchor Fixture")
    pdf.setAuthor("Local Academic Paper Chatbot")
    pdf.setSubject("Deterministic native-text anchor and page-render fixture")
    pdf.setCreator(FIXTURE_PROFILE_ID)
    pdf.setKeywords("academic chatbot, deterministic fixture, native PDF text")


def _draw_page(pdf: Canvas, *, page_number: int, printed_label: str) -> None:
    page_width, page_height = letter
    pdf.setFont("Helvetica", 16)
    pdf.drawString(72, page_height - 72, "Deterministic Native PDF Fixture")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(72, page_height - 112, f"Physical page {page_number} of 2.")
    if page_number == 1:
        pdf.drawString(
            72,
            page_height - 148,
            "This page establishes a distinct native-text control sample.",
        )
    else:
        pdf.drawString(72, page_height - 148, ANCHOR_SENTENCE)
        pdf.drawString(
            72,
            page_height - 172,
            "The sentence above is the sole exact anchor in this document.",
        )

    pdf.setFont("Helvetica", 10)
    pdf.drawCentredString(page_width / 2, 36, printed_label)
    pdf.showPage()


def generate_pdf(output: Path) -> None:
    """Write the deterministic two-page PDF fixture to ``output``."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = Canvas(
        str(output),
        pagesize=letter,
        invariant=1,
        pageCompression=0,
    )
    _configure_metadata(pdf)
    _draw_page(pdf, page_number=1, printed_label="A-6")
    _draw_page(pdf, page_number=2, printed_label="A-7")
    pdf.save()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    generate_pdf(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
