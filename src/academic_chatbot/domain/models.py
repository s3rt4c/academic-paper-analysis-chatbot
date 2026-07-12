from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EvidenceSpan(BaseModel, frozen=True):
    evidence_id: str
    paper_id: str
    file_version_id: str
    physical_page_index: int = Field(ge=0)
    printed_page_label: str | None
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction_method: Literal["native_text", "ocr"]
    quality: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_offsets(self) -> "EvidenceSpan":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        return self


class SourceScope(BaseModel, frozen=True):
    project_id: str
    file_version_ids: tuple[str, ...] = Field(min_length=1)
    section_types: tuple[str, ...] = ()
