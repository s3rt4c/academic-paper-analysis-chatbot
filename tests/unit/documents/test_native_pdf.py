from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from academic_chatbot.documents import native_pdf
from academic_chatbot.documents.native_pdf import (
    NativePdfIntegrityError,
    NativePdfMalformedError,
    NativePdfParser,
    NativePdfSourceError,
)
from academic_chatbot.domain.library import FileVersion
from academic_chatbot.library.service import LibraryService
from academic_chatbot.storage.paths import ProjectPaths

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "pdfs" / "native_anchor.pdf"


def _admitted_fixture(tmp_path: Path) -> tuple[FileVersion, ProjectPaths]:
    service = LibraryService(data_root=tmp_path / "data", max_pdf_bytes=1_000_000)
    project = service.create_project(display_name="Research", project_id="project-1")
    paper = service.create_paper(project_id=project.project_id, paper_id="paper-1")
    file_version = service.admit_pdf(
        project_id=project.project_id,
        paper_id=paper.paper_id,
        source_path=_FIXTURE,
    )
    return file_version, ProjectPaths.create(tmp_path / "data", project_id=project.project_id)


def _stored_file_version(paths: ProjectPaths, *, relative_path: str, data: bytes) -> FileVersion:
    target = paths.resolve_relative(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return FileVersion(
        file_version_id="fv-test-native-pdf",
        paper_id="paper-1",
        sha256=hashlib.sha256(data).hexdigest(),
        byte_length=len(data),
        stored_relative_path=relative_path,
    )


def test_parse_admitted_native_fixture_preserves_file_version_page_and_anchor_lineage(
    tmp_path: Path,
) -> None:
    """Would fail if parsing substitutes a digest-peer FileVersion or page numbering drifts."""
    file_version, paths = _admitted_fixture(tmp_path)

    parsed = NativePdfParser(paths).parse(file_version)

    assert parsed.file_version == file_version
    assert parsed.source_pdf_sha256 == file_version.sha256
    assert [(page.physical_page_index, page.display_page_number) for page in parsed.pages] == [
        (0, 1),
        (1, 2),
    ]
    assert [page.printed_page_label for page in parsed.pages] == ["A-6", "A-7"]
    assert all(page.quality == "adequate_native_text" for page in parsed.pages)
    assert all(not page.needs_ocr for page in parsed.pages)
    assert all(
        anchor.file_version_id == file_version.file_version_id
        for page in parsed.pages
        for anchor in page.anchors
    )
    assert all(
        anchor.canonical_page_text[anchor.char_start : anchor.char_end] == anchor.anchor_text
        for page in parsed.pages
        for anchor in page.anchors
    )


def test_stored_pdf_tampering_fails_before_pdfplumber_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if a changed stored original reached the extractor before integrity validation."""
    file_version, paths = _admitted_fixture(tmp_path)
    stored = paths.resolve_relative(file_version.stored_relative_path)
    stored.write_bytes(stored.read_bytes() + b"tampered")

    def extraction_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("pdfplumber must not receive an integrity-mismatched stored PDF")

    monkeypatch.setattr(native_pdf.pdfplumber, "open", extraction_must_not_run)

    with pytest.raises(NativePdfIntegrityError, match="digest"):
        NativePdfParser(paths).parse(file_version)


def test_malformed_pdf_is_a_parser_failure_not_an_ocr_classification(tmp_path: Path) -> None:
    """Would fail if malformed inputs were converted into a recoverable text-quality result."""
    paths = ProjectPaths.create(tmp_path / "data", project_id="project-1")
    file_version = _stored_file_version(
        paths,
        relative_path="originals/sha256/malformed.pdf",
        data=b"%PDF-1.4\nthis is not a complete PDF",
    )

    with pytest.raises(NativePdfMalformedError):
        NativePdfParser(paths).parse(file_version)


def test_empty_and_low_native_text_are_conservative_ocr_heuristics_without_ocr(
    tmp_path: Path,
) -> None:
    """Would fail if native-text insufficiency were hidden or treated as an extraction failure."""
    paths = ProjectPaths.create(tmp_path / "data", project_id="project-1")
    empty_source = tmp_path / "empty.pdf"
    low_source = tmp_path / "low.pdf"
    for source, text in ((empty_source, None), (low_source, "Two words")):
        pdf = canvas.Canvas(str(source), invariant=1)
        if text is not None:
            pdf.drawString(72, 720, text)
        else:
            pdf.showPage()
        pdf.save()

    empty = NativePdfParser(paths).parse(
        _stored_file_version(
            paths,
            relative_path="originals/sha256/empty.pdf",
            data=empty_source.read_bytes(),
        )
    ).pages[0]
    low = NativePdfParser(paths).parse(
        _stored_file_version(
            paths,
            relative_path="originals/sha256/low.pdf",
            data=low_source.read_bytes(),
        )
    ).pages[0]

    assert (empty.quality, empty.needs_ocr, empty.anchors) == (
        "empty_native_text",
        True,
        (),
    )
    assert (low.quality, low.needs_ocr) == ("low_native_text", True)


def test_native_parser_result_does_not_depend_on_host_python_patch_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if ordinary parser evidence drifted with host Python patch metadata."""
    file_version, paths = _admitted_fixture(tmp_path)
    expected = NativePdfParser(paths).parse(file_version)

    monkeypatch.setattr(platform, "python_version", lambda: "3.12.999")

    assert NativePdfParser(paths).parse(file_version) == expected


def test_parser_rejects_a_windows_reparse_point_before_opening_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if project-relative stored paths could still traverse a reparse point."""
    file_version, paths = _admitted_fixture(tmp_path)
    monkeypatch.setattr(native_pdf, "_is_reparse_point", lambda _: True, raising=False)

    with pytest.raises(NativePdfSourceError, match="reparse"):
        NativePdfParser(paths).parse(file_version)


def test_parser_wraps_stored_file_inspection_permission_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if hostile stored-file inspection errors escaped the parser taxonomy."""
    file_version, paths = _admitted_fixture(tmp_path)
    stored = paths.resolve_relative(file_version.stored_relative_path)
    real_lstat = Path.lstat

    def denied_lstat(path: Path) -> object:
        if path == stored:
            raise PermissionError("synthetic stored-file access denial")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", denied_lstat)

    with pytest.raises(NativePdfSourceError, match="inspected"):
        NativePdfParser(paths).parse(file_version)
