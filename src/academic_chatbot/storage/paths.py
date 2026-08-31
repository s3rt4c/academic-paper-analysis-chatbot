"""Project-local path construction and containment checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath


class PathEscapeError(ValueError):
    """Raised when a path cannot be safely represented below a project root."""


def ensure_path_beneath(*, root: Path, candidate: Path) -> Path:
    """Resolve *candidate* and ensure it remains beneath the resolved *root*.

    Resolution catches existing symlink/reparse-point escapes.  Like every
    filesystem path check, it cannot eliminate a concurrent replacement race
    after this function returns; callers keep creation and publication scoped
    to the checked parent directory.
    """

    resolved_root = root.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        message = f"path escapes the configured root: {candidate}"
        raise PathEscapeError(message) from error
    return resolved_candidate


def _validate_project_id(project_id: str) -> str:
    if not project_id or project_id in {".", ".."}:
        raise PathEscapeError("project identifier must be a non-empty plain path segment")
    pure = PureWindowsPath(project_id)
    if (
        pure.name != project_id
        or pure.drive
        or pure.root
        or "/" in project_id
        or "\\" in project_id
    ):
        raise PathEscapeError("project identifier must not contain path syntax")
    return project_id


def _validate_stored_relative_path(relative_path: str) -> Path:
    if not relative_path or "\\" in relative_path:
        raise PathEscapeError("stored paths must be non-empty canonical POSIX relative paths")
    windows_path = PureWindowsPath(relative_path)
    if windows_path.drive or windows_path.root or windows_path.is_absolute():
        raise PathEscapeError("stored paths must not be absolute or drive-qualified")
    parts = relative_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PathEscapeError("stored paths must not contain empty, current, or parent segments")
    return Path(*parts)


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Canonical locations for one project's private, local files."""

    data_root: Path
    project_id: str

    @classmethod
    def create(cls, data_root: Path, *, project_id: str) -> ProjectPaths:
        root = data_root.resolve(strict=False)
        return cls(data_root=root, project_id=_validate_project_id(project_id))

    @property
    def project_root(self) -> Path:
        return ensure_path_beneath(
            root=self.data_root, candidate=self.data_root / "projects" / self.project_id
        )

    @property
    def database_path(self) -> Path:
        return self.project_root / "project.sqlite3"

    @property
    def originals_dir(self) -> Path:
        return self.project_root / "originals"

    @property
    def derivatives_dir(self) -> Path:
        return self.project_root / "derivatives"

    @property
    def indexes_dir(self) -> Path:
        return self.project_root / "indexes"

    @property
    def audit_dir(self) -> Path:
        return self.project_root / "audit"

    @property
    def cache_dir(self) -> Path:
        return self.project_root / "cache"

    @property
    def transactions_dir(self) -> Path:
        return self.project_root / "transactions"

    @property
    def backups_dir(self) -> Path:
        return self.project_root / "backups"

    def resolve_relative(self, relative_path: str) -> Path:
        fragment = _validate_stored_relative_path(relative_path)
        return ensure_path_beneath(root=self.project_root, candidate=self.project_root / fragment)

    def to_relative_posix(self, path: Path) -> str:
        resolved_path = ensure_path_beneath(root=self.project_root, candidate=path)
        return resolved_path.relative_to(self.project_root).as_posix()
