from __future__ import annotations

import os
from pathlib import Path

import pytest

from academic_chatbot.settings import ApplicationSettings, SettingsError, default_data_root
from academic_chatbot.storage.paths import PathEscapeError, ProjectPaths


def test_explicit_data_root_is_resolved_at_construction_time(tmp_path: Path) -> None:
    supplied_root = tmp_path / "data-root" / ".." / "data-root"

    settings = ApplicationSettings.create(data_root=supplied_root)

    assert settings.data_root == supplied_root.resolve(strict=False)


def test_default_data_root_uses_supplied_windows_environment_value() -> None:
    root = default_data_root(environment={"LOCALAPPDATA": r"C:\\Users\\example\\AppData\\Local"})

    assert root == Path(r"C:\\Users\\example\\AppData\\Local") / "LocalAcademicPaperChatbot"


def test_default_data_root_fails_clearly_when_localappdata_is_absent() -> None:
    with pytest.raises(SettingsError, match="LOCALAPPDATA"):
        default_data_root(environment={})


def test_project_paths_are_deterministic_and_stay_below_the_data_root(tmp_path: Path) -> None:
    paths = ProjectPaths.create(tmp_path, project_id="project-1")

    assert paths.project_root == tmp_path / "projects" / "project-1"
    assert paths.database_path == tmp_path / "projects" / "project-1" / "project.sqlite3"
    assert paths.resolve_relative("originals/paper.pdf") == paths.originals_dir / "paper.pdf"


@pytest.mark.parametrize(
    "fragment",
    (
        "../outside.txt",
        "..\\outside.txt",
        "/outside.txt",
        r"C:\\outside.txt",
        "C:/outside.txt",
        "originals\\paper.pdf",
    ),
)
def test_project_paths_reject_noncanonical_or_escaping_persisted_paths(
    tmp_path: Path, fragment: str
) -> None:
    paths = ProjectPaths.create(tmp_path, project_id="project-1")

    with pytest.raises(PathEscapeError):
        paths.resolve_relative(fragment)


def test_project_paths_round_trip_canonical_posix_storage_path(tmp_path: Path) -> None:
    paths = ProjectPaths.create(tmp_path, project_id="project-1")
    target = paths.resolve_relative("derivatives/version-1/normalized.json")

    stored = paths.to_relative_posix(target)

    assert stored == "derivatives/version-1/normalized.json"
    assert paths.resolve_relative(stored) == target


def test_project_paths_reject_absolute_path_outside_project_root(tmp_path: Path) -> None:
    paths = ProjectPaths.create(tmp_path, project_id="project-1")
    outside = tmp_path.parent / "outside.txt"

    with pytest.raises(PathEscapeError):
        paths.to_relative_posix(outside)


def test_existing_symlink_escape_is_rejected_when_supported(tmp_path: Path) -> None:
    paths = ProjectPaths.create(tmp_path, project_id="project-1")
    paths.project_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = paths.project_root / "linked"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(PathEscapeError):
        paths.resolve_relative("linked/escape.txt")
