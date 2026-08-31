"""Minimal persistent identities for local projects and admitted originals."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def file_version_id_for(*, paper_id: str, sha256: str) -> str:
    """Return a deterministic logical FileVersion ID for one paper and digest."""

    paper_id_bytes = paper_id.encode("utf-8")
    digest_bytes = sha256.encode("ascii")
    identity = (
        len(paper_id_bytes).to_bytes(8, byteorder="big")
        + paper_id_bytes
        + len(digest_bytes).to_bytes(8, byteorder="big")
        + digest_bytes
    )
    return f"fv-paper-sha256-{hashlib.sha256(identity).hexdigest()}"


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
