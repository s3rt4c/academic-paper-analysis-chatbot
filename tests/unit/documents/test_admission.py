from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from academic_chatbot.documents import admission
from academic_chatbot.documents.admission import PdfAdmissionError, PdfSizeLimitError
from academic_chatbot.library.repository import ProjectRepository
from academic_chatbot.library.service import LibraryService
from academic_chatbot.storage.paths import ProjectPaths

_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n"


def _service_with_paper(
    tmp_path: Path, *, max_pdf_bytes: int = 1024
) -> tuple[LibraryService, str, str]:
    service = LibraryService(data_root=tmp_path / "data", max_pdf_bytes=max_pdf_bytes)
    project = service.create_project(display_name="Research", project_id="project-1")
    paper = service.create_paper(project_id=project.project_id, paper_id="paper-1")
    return service, project.project_id, paper.paper_id


def test_valid_pdf_is_copied_by_content_hash_without_mutating_the_source(tmp_path: Path) -> None:
    service, project_id, paper_id = _service_with_paper(tmp_path)
    source = tmp_path / "outside" / "input.pdf"
    source.parent.mkdir()
    source.write_bytes(_PDF_BYTES)
    original_source_bytes = source.read_bytes()

    admitted = service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=source)

    expected_digest = hashlib.sha256(_PDF_BYTES).hexdigest()
    paths = ProjectPaths.create(tmp_path / "data", project_id=project_id)
    stored = paths.resolve_relative(admitted.stored_relative_path)
    assert admitted.sha256 == expected_digest
    assert admitted.byte_length == len(_PDF_BYTES)
    assert admitted.stored_relative_path == f"originals/sha256/{expected_digest}.pdf"
    assert stored.read_bytes() == _PDF_BYTES
    assert source.read_bytes() == original_source_bytes


def test_admission_uses_pdf_signature_not_filename_extension(tmp_path: Path) -> None:
    service, project_id, paper_id = _service_with_paper(tmp_path)
    invalid = tmp_path / "not-a-pdf.pdf"
    invalid.write_bytes(b"not pdf")
    valid_without_extension = tmp_path / "valid-document.bin"
    valid_without_extension.write_bytes(_PDF_BYTES)

    with pytest.raises(PdfAdmissionError, match="signature"):
        service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=invalid)

    admitted = service.admit_pdf(
        project_id=project_id, paper_id=paper_id, source_path=valid_without_extension
    )
    assert admitted.sha256 == hashlib.sha256(_PDF_BYTES).hexdigest()


@pytest.mark.parametrize("source_kind", ("missing", "directory"))
def test_admission_rejects_missing_or_nonregular_sources(tmp_path: Path, source_kind: str) -> None:
    service, project_id, paper_id = _service_with_paper(tmp_path)
    source = tmp_path / source_kind
    if source_kind == "directory":
        source.mkdir()

    with pytest.raises(PdfAdmissionError):
        service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=source)


def test_admission_rejects_oversized_input_without_publishing(tmp_path: Path) -> None:
    service, project_id, paper_id = _service_with_paper(tmp_path, max_pdf_bytes=len(_PDF_BYTES) - 1)
    source = tmp_path / "large.pdf"
    source.write_bytes(_PDF_BYTES)

    with pytest.raises(PdfSizeLimitError):
        service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=source)

    paths = ProjectPaths.create(tmp_path / "data", project_id=project_id)
    assert not paths.originals_dir.exists()


def test_admission_streams_without_reading_the_source_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, project_id, paper_id = _service_with_paper(tmp_path)
    source = tmp_path / "streamed.pdf"
    source.write_bytes(_PDF_BYTES)

    def forbid_read_bytes(self: Path) -> bytes:
        if self == source:
            raise AssertionError("admission must stream source bytes")
        return original_read_bytes(self)

    original_read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)

    admitted = service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=source)
    assert admitted.byte_length == len(_PDF_BYTES)


def test_identical_bytes_reuse_one_immutable_file_version(tmp_path: Path) -> None:
    service, project_id, paper_id = _service_with_paper(tmp_path)
    first_source = tmp_path / "first-name.pdf"
    second_source = tmp_path / "second-name.pdf"
    first_source.write_bytes(_PDF_BYTES)
    second_source.write_bytes(_PDF_BYTES)

    first = service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=first_source)
    second = service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=second_source)

    assert second == first
    paths = ProjectPaths.create(tmp_path / "data", project_id=project_id)
    assert list(paths.originals_dir.rglob("*.pdf")) == [
        paths.resolve_relative(first.stored_relative_path)
    ]


def test_identical_bytes_for_another_paper_preserve_each_paper_ownership(tmp_path: Path) -> None:
    service, project_id, paper_id = _service_with_paper(tmp_path)
    other_paper = service.create_paper(project_id=project_id, paper_id="paper-2")
    source = tmp_path / "same.pdf"
    source.write_bytes(_PDF_BYTES)

    first = service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=source)
    second = service.admit_pdf(
        project_id=project_id, paper_id=other_paper.paper_id, source_path=source
    )
    first_reimported = service.admit_pdf(
        project_id=project_id, paper_id=paper_id, source_path=source
    )
    second_reimported = service.admit_pdf(
        project_id=project_id, paper_id=other_paper.paper_id, source_path=source
    )

    assert first.paper_id == paper_id
    assert second.paper_id == other_paper.paper_id
    assert first.file_version_id != second.file_version_id
    assert first.sha256 == second.sha256
    assert first.stored_relative_path == second.stored_relative_path
    assert first_reimported == first
    assert second_reimported == second
    repository = service.repository_for_project_id(project_id)
    assert repository.file_version_count() == 2
    paths = ProjectPaths.create(tmp_path / "data", project_id=project_id)
    assert list(paths.originals_dir.rglob("*.pdf")) == [
        paths.resolve_relative(first.stored_relative_path)
    ]


def test_shared_original_survives_failed_second_paper_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, project_id, paper_id = _service_with_paper(tmp_path)
    other_paper = service.create_paper(project_id=project_id, paper_id="paper-2")
    source = tmp_path / "same.pdf"
    source.write_bytes(_PDF_BYTES)
    first = service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=source)
    paths = ProjectPaths.create(tmp_path / "data", project_id=project_id)
    stored = paths.resolve_relative(first.stored_relative_path)

    def fail_registration(self: ProjectRepository, **kwargs: object) -> object:
        raise RuntimeError("synthetic second-paper database failure")

    monkeypatch.setattr(
        ProjectRepository, "register_file_version_in_transaction", fail_registration
    )

    with pytest.raises(RuntimeError, match="second-paper database failure"):
        service.admit_pdf(project_id=project_id, paper_id=other_paper.paper_id, source_path=source)

    assert stored.read_bytes() == _PDF_BYTES
    assert service.repository_for_project_id(project_id).file_version_count() == 1


def test_same_filename_with_different_bytes_creates_distinct_content_identities(
    tmp_path: Path,
) -> None:
    service, project_id, paper_id = _service_with_paper(tmp_path)
    first_source = tmp_path / "first" / "same.pdf"
    second_source = tmp_path / "second" / "same.pdf"
    first_source.parent.mkdir()
    second_source.parent.mkdir()
    first_source.write_bytes(_PDF_BYTES)
    second_source.write_bytes(_PDF_BYTES.replace(b"1.4", b"1.5"))

    first = service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=first_source)
    second = service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=second_source)

    assert first.sha256 != second.sha256
    assert first.file_version_id != second.file_version_id


def test_copy_failure_does_not_register_a_file_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, project_id, paper_id = _service_with_paper(tmp_path)
    source = tmp_path / "input.pdf"
    source.write_bytes(_PDF_BYTES)

    def fail_copy(*args: object, **kwargs: object) -> object:
        raise OSError("synthetic copy failure")

    monkeypatch.setattr(admission, "atomic_stream_copy", fail_copy)

    with pytest.raises(OSError, match="synthetic copy failure"):
        service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=source)

    assert service.repository_for_project_id(project_id).file_version_count() == 0


def test_source_replaced_before_opening_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, project_id, paper_id = _service_with_paper(tmp_path)
    source = tmp_path / "source.pdf"
    external = tmp_path / "external.pdf"
    source.write_bytes(_PDF_BYTES)
    external.write_bytes(_PDF_BYTES.replace(b"1.4", b"1.5"))
    original_open = Path.open
    swapped = False

    def swap_then_open(self: Path, *args: object, **kwargs: object) -> object:
        nonlocal swapped
        if self == source and not swapped:
            swapped = True
            external.replace(source)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_then_open)

    with pytest.raises(PdfAdmissionError, match="changed"):
        service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=source)


def test_database_failure_removes_a_new_unregistered_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, project_id, paper_id = _service_with_paper(tmp_path)
    source = tmp_path / "input.pdf"
    source.write_bytes(_PDF_BYTES)

    def fail_registration(self: ProjectRepository, **kwargs: object) -> object:
        raise RuntimeError("synthetic database failure")

    monkeypatch.setattr(
        ProjectRepository, "register_file_version_in_transaction", fail_registration
    )

    with pytest.raises(RuntimeError, match="synthetic database failure"):
        service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=source)

    expected = hashlib.sha256(_PDF_BYTES).hexdigest()
    paths = ProjectPaths.create(tmp_path / "data", project_id=project_id)
    assert not paths.resolve_relative(f"originals/sha256/{expected}.pdf").exists()
    assert service.repository_for_project_id(project_id).file_version_count() == 0


def test_source_mutation_during_streaming_copy_is_detected_before_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, project_id, paper_id = _service_with_paper(tmp_path)
    source = tmp_path / "changing.pdf"
    source.write_bytes(_PDF_BYTES)
    original_copy = admission.atomic_stream_copy

    def copy_then_change_source(*args: object, **kwargs: object) -> object:
        copied = original_copy(*args, **kwargs)
        source.write_bytes(_PDF_BYTES + b"changed")
        return copied

    monkeypatch.setattr(admission, "atomic_stream_copy", copy_then_change_source)

    with pytest.raises(PdfAdmissionError, match="changed"):
        service.admit_pdf(project_id=project_id, paper_id=paper_id, source_path=source)

    assert service.repository_for_project_id(project_id).file_version_count() == 0
