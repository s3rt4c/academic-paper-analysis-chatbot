"""Rights-safe PDFs containing only deterministic synthetic text."""

from pathlib import Path
from textwrap import wrap

from reportlab.pdfgen import canvas


def write_pdf(
    path: Path,
    *,
    empty: bool = False,
    suffix: str = "",
    text: str | None = None,
) -> Path:
    pdf = canvas.Canvas(str(path), invariant=1)
    if not empty:
        lines = (
            [
                line
                for paragraph in text.splitlines()
                for line in wrap(
                    paragraph, width=70, break_long_words=False, break_on_hyphens=False
                )
            ]
            if text is not None
            else [
                f"Alpha beta gamma delta synthetic evidence {index} {suffix}" for index in range(6)
            ]
        )
        for index, line in enumerate(lines):
            pdf.drawString(40, 760 - index * 25, line)
    pdf.showPage()
    pdf.save()
    return path
