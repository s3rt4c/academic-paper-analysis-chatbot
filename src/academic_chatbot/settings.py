"""Runtime-resolved local application settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class SettingsError(RuntimeError):
    """Raised when a required local application setting is unavailable."""


def default_data_root(*, environment: Mapping[str, str] | None = None) -> Path:
    """Return the Windows-local application data root without creating it."""

    values = os.environ if environment is None else environment
    local_app_data = values.get("LOCALAPPDATA")
    if not local_app_data:
        message = "LOCALAPPDATA is required to determine the local application data root"
        raise SettingsError(message)
    return Path(local_app_data) / "LocalAcademicPaperChatbot"


@dataclass(frozen=True, slots=True)
class ApplicationSettings:
    """Settings deliberately limited to the local persistent data boundary."""

    data_root: Path

    @classmethod
    def create(
        cls,
        *,
        data_root: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> ApplicationSettings:
        root = default_data_root(environment=environment) if data_root is None else data_root
        return cls(data_root=root.resolve(strict=False))
