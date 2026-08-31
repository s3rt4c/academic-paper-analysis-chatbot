"""Minimal persistent identities for local projects and admitted originals."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class Project(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    project_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)


class Paper(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    paper_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)


class FileVersion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    file_version_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_length: int = Field(ge=0)
    stored_relative_path: str = Field(min_length=1)
