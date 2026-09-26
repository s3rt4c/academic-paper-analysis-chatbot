"""Rights-safe PDFs containing only deterministic synthetic text."""

from pathlib import Path

from reportlab.pdfgen import canvas


def write_pdf(path: Path, *, empty: bool = False, suffix: str = "") -> Path:
    pdf = canvas.Canvas(str(path), invariant=1)
    if not empty:
        for index in range(6):
            pdf.drawString(
                40, 760 - index * 25, f"Alpha beta gamma delta synthetic evidence {index} {suffix}"
            )
    pdf.showPage()
    pdf.save()
    return path
