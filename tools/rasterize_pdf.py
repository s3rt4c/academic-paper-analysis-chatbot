import os
import sys
from pathlib import Path

from pdf2image import convert_from_path


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: rasterize_pdf.py input.pdf output_dir")
    source = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configured_poppler = os.environ.get("ACADEMIC_CHATBOT_POPPLER_PATH")
    if configured_poppler:
        poppler = Path(configured_poppler).expanduser().resolve()
        if not poppler.is_dir():
            raise SystemExit(
                "ACADEMIC_CHATBOT_POPPLER_PATH must name a directory containing Poppler binaries"
            )
        poppler_path: str | None = str(poppler)
    else:
        # pdf2image resolves pdftoppm from PATH when this is None.
        poppler_path = None
    paths = convert_from_path(
        str(source),
        dpi=144,
        fmt="png",
        thread_count=4,
        output_folder=str(output),
        paths_only=True,
        output_file="page",
        poppler_path=poppler_path,
    )
    numbered = []
    for index, raw in enumerate(sorted(Path(p) for p in paths), start=1):
        destination = output / f"page-{index}.png"
        if destination.exists():
            destination.unlink()
        raw.replace(destination)
        numbered.append(destination)
    print(f"pages={len(numbered)}")


if __name__ == "__main__":
    main()
