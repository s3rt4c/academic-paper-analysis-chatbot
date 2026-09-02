"""Immutable, persistence-free Phase 1B embedding contracts."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from enum import StrEnum
from pathlib import PureWindowsPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_EMBEDDING_PROFILE_ID_PATTERN = r"^ep-sha256-[0-9a-f]{64}$"
_REVISION_PATTERN = r"^[0-9a-f]{40}$"

FROZEN_DIMENSION = 384
FROZEN_MAX_SEQUENCE_LENGTH = 512
FROZEN_DOCUMENT_SOURCE_TOKEN_BUDGET = 510
FROZEN_QUERY_SOURCE_TOKEN_BUDGET = 502
FROZEN_QUERY_PREFIX_TOKEN_COUNT = 8
FROZEN_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
FROZEN_SPAN_POLICY_ID = "canonical-word-greedy-v1-zero-overlap"


class EmbeddingContractError(ValueError):
    """Stable expected failure at the embedding-contract boundary."""


class EmbeddingRole(StrEnum):
    DOCUMENT = "document"
    QUERY = "query"


def canonical_json_bytes(payload: object) -> bytes:
    """Return the repository's UTF-8 canonical JSON representation."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_relative_posix_path(value: str) -> str:
    if not value or "\\" in value:
        raise ValueError("artifact paths must be non-empty canonical POSIX relative paths")
    windows_path = PureWindowsPath(value)
    if windows_path.drive or windows_path.root or windows_path.is_absolute():
        raise ValueError("artifact paths must not be absolute, UNC, or drive-qualified")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("artifact paths must not contain empty, current, or parent segments")
    return value


class ArtifactFile(BaseModel):
    """One exact, relative external embedding artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    filename: str = Field(min_length=1)
    byte_size: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, value: str) -> str:
        return _validate_relative_posix_path(value)


class SpecialTokenPolicy(BaseModel):
    """Frozen single-sequence token accounting shared by roles."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    add_special_tokens: bool = True
    document_content_token_budget: int = Field(gt=0)
    query_prefix_token_count: int = Field(ge=0)
    query_source_token_budget: int = Field(gt=0)
    single_sequence_template: str = Field(min_length=1)
    special_token_count: int = Field(ge=0)


class NormalizationProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    dtype: str = Field(min_length=1)
    rule: str = Field(min_length=1)


class TokenizerProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    artifact: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    type: str = Field(min_length=1)

    @field_validator("artifact")
    @classmethod
    def _validate_artifact(cls, value: str) -> str:
        return _validate_relative_posix_path(value)


class OnnxRuntimeCompatibility(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    provider: str = Field(min_length=1)
    requirement: str = Field(min_length=1)


class OnnxTensorContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)


class OnnxOutputContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    batch_axis: str = Field(min_length=1)
    sequence_axis: str = Field(min_length=1)
    dimension: int = Field(gt=0)


class OnnxGraphContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    opset: int = Field(gt=0)
    inputs: tuple[OnnxTensorContract, ...] = Field(min_length=1)
    output: OnnxOutputContract

    @field_validator("inputs", mode="before")
    @classmethod
    def _freeze_json_inputs(cls, value: object) -> object:
        """Accept the canonical JSON arrays, then retain immutable tuples."""

        if isinstance(value, list):
            return tuple(
                {"name": item[0], "dtype": item[1]}
                if isinstance(item, list) and len(item) == 2
                else item
                for item in value
            )
        return value

    @field_validator("output", mode="before")
    @classmethod
    def _decode_json_output(cls, value: object) -> object:
        """Accept the canonical JSON output array without weakening validation."""

        if isinstance(value, list) and len(value) == 5:
            return {
                "name": value[0],
                "dtype": value[1],
                "batch_axis": value[2],
                "sequence_axis": value[3],
                "dimension": value[4],
            }
        return value

    @model_validator(mode="after")
    def _validate_inputs(self) -> Self:
        expected = ("input_ids", "attention_mask", "token_type_ids")
        if tuple(item.name for item in self.inputs) != expected:
            raise ValueError("ONNX inputs must be the frozen BGE input contract")
        return self


class SpanPolicy(BaseModel):
    """The immutable semantics bound by the frozen span-policy ID."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    profile_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    overlap_words: int = Field(ge=0)
    truncation: str = Field(min_length=1)
    single_overbudget_word: str = Field(min_length=1)


class EmbeddingProfile(BaseModel):
    """All host-independent facts that define vector meaning for one profile."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["embedding-profile-v1"] = "embedding-profile-v1"
    model_repository: str = Field(min_length=1)
    model_revision: str = Field(pattern=_REVISION_PATTERN)
    artifacts: tuple[ArtifactFile, ...] = Field(min_length=1)
    artifact_set_sha256: str = Field(pattern=_SHA256_PATTERN)
    dimension: int = Field(gt=0)
    max_sequence_length: int = Field(gt=0)
    document_prefix_utf8: str
    query_prefix_utf8: str
    special_token_policy: SpecialTokenPolicy
    pooling: str = Field(min_length=1)
    normalization: NormalizationProfile
    span_policy: str = Field(min_length=1)
    tokenizer: TokenizerProfile
    onnx_runtime: OnnxRuntimeCompatibility
    onnx_graph: OnnxGraphContract
    onnx_ir_version: int = Field(gt=0)

    @field_validator("artifacts", mode="before")
    @classmethod
    def _freeze_json_artifacts(cls, value: object) -> object:
        """Accept a manifest's JSON list while storing immutable artifact tuples."""

        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_profile(self) -> Self:
        if tuple(item.filename for item in self.artifacts) != tuple(
            sorted(item.filename for item in self.artifacts)
        ):
            raise ValueError("profile artifacts must be ordered by canonical relative filename")
        if len({item.filename for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("profile artifact filenames must be unique")
        if self.tokenizer.artifact not in {item.filename for item in self.artifacts}:
            raise ValueError("tokenizer artifact must be an item in the profile artifact inventory")
        expected_artifact_set = artifact_set_sha256_for(self.artifacts)
        if not hmac.compare_digest(self.artifact_set_sha256, expected_artifact_set):
            raise ValueError("artifact_set_sha256 does not match the canonical artifact inventory")
        if self.dimension != self.onnx_graph.output.dimension:
            raise ValueError("profile dimension must equal the ONNX output dimension")
        policy = self.special_token_policy
        if (
            policy.document_content_token_budget + policy.special_token_count
            != self.max_sequence_length
        ):
            raise ValueError("document token budget must account for the model sequence length")
        if (
            policy.query_source_token_budget
            + policy.query_prefix_token_count
            + policy.special_token_count
            != self.max_sequence_length
        ):
            raise ValueError("query token budget must account for prefix and special tokens")
        return self

    def semantic_identity_payload(self) -> dict[str, object]:
        """Return the exact Task 0 vector-semantic canonical payload.

        ONNX IR is retained on the immutable profile for artifact compatibility,
        but is deliberately absent here: the frozen Task 0 profile identity was
        calculated from graph semantics/opset and not the serialization IR.
        """

        return {
            "schema_version": self.schema_version,
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "artifacts": [item.model_dump(mode="json") for item in self.artifacts],
            "artifact_set_sha256": self.artifact_set_sha256,
            "dimension": self.dimension,
            "max_sequence_length": self.max_sequence_length,
            "document_prefix_utf8": self.document_prefix_utf8,
            "query_prefix_utf8": self.query_prefix_utf8,
            "special_token_policy": self.special_token_policy.model_dump(mode="json"),
            "pooling": self.pooling,
            "normalization": self.normalization.model_dump(mode="json"),
            "span_policy": self.span_policy,
            "tokenizer": self.tokenizer.model_dump(mode="json"),
            "onnx_runtime": self.onnx_runtime.model_dump(mode="json"),
            "onnx_graph": {
                "opset": self.onnx_graph.opset,
                "inputs": [[item.name, item.dtype] for item in self.onnx_graph.inputs],
                "output": [
                    self.onnx_graph.output.name,
                    self.onnx_graph.output.dtype,
                    self.onnx_graph.output.batch_axis,
                    self.onnx_graph.output.sequence_axis,
                    self.onnx_graph.output.dimension,
                ],
            },
        }

    @computed_field(return_type=str)  # type: ignore[prop-decorator]
    @property
    def embedding_profile_id(self) -> str:
        return (
            "ep-sha256-"
            + hashlib.sha256(canonical_json_bytes(self.semantic_identity_payload())).hexdigest()
        )

    def role_prefix(self, role: EmbeddingRole) -> str:
        return (
            self.document_prefix_utf8 if role is EmbeddingRole.DOCUMENT else self.query_prefix_utf8
        )

    def role_source_budget(self, role: EmbeddingRole) -> int:
        if role is EmbeddingRole.DOCUMENT:
            return self.special_token_policy.document_content_token_budget
        return self.special_token_policy.query_source_token_budget


class EmbeddingArtifactManifest(BaseModel):
    """Strict in-memory representation of the future external manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    manifest_schema_version: Literal["embedding-artifact-manifest-v1"]
    embedding_profile: EmbeddingProfile
    embedding_profile_id: str = Field(pattern=_EMBEDDING_PROFILE_ID_PATTERN)
    declared_license: str = Field(min_length=1)
    artifacts: tuple[ArtifactFile, ...] = Field(min_length=1)

    @field_validator("artifacts", mode="before")
    @classmethod
    def _freeze_json_artifacts(cls, value: object) -> object:
        """Accept a manifest's JSON list while storing immutable artifact tuples."""

        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        if not hmac.compare_digest(
            self.embedding_profile_id, self.embedding_profile.embedding_profile_id
        ):
            raise ValueError("manifest embedding_profile_id does not match its semantic profile")
        if len({artifact.filename for artifact in self.artifacts}) != len(self.artifacts):
            raise ValueError("manifest artifact filenames must be unique")
        if self.artifacts != self.embedding_profile.artifacts:
            raise ValueError("manifest artifacts must exactly match the profile inventory")
        return self

    def canonical_payload(self) -> dict[str, object]:
        """Return the portable future manifest representation with no local path."""

        profile = self.embedding_profile
        return {
            "manifest_schema_version": self.manifest_schema_version,
            "embedding_profile": profile.model_dump(mode="json", exclude={"embedding_profile_id"}),
            "embedding_profile_id": self.embedding_profile_id,
            "declared_license": self.declared_license,
            "artifacts": [item.model_dump(mode="json") for item in self.artifacts],
        }


class EmbeddingSpanIdentity(BaseModel):
    """A deterministic occurrence identity; no persistence or span splitting."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    document_generation_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    embedding_profile_id: str = Field(pattern=_EMBEDDING_PROFILE_ID_PATTERN)

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.end_offset <= self.start_offset:
            raise ValueError("embedding span offsets must form a non-empty half-open range")
        return self

    @computed_field(return_type=str)  # type: ignore[prop-decorator]
    @property
    def embedding_span_id(self) -> str:
        payload = self.model_dump(mode="json", exclude={"embedding_span_id"})
        return "embedding-span-sha256-" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def artifact_set_sha256_for(artifacts: tuple[ArtifactFile, ...]) -> str:
    """Hash exactly the canonical ordered artifact inventory."""

    payload = [artifact.model_dump(mode="json") for artifact in artifacts]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_embedding_profile_id(value: str) -> str:
    """Validate a profile ID supplied at a non-profile boundary."""

    if re.fullmatch(_EMBEDDING_PROFILE_ID_PATTERN, value) is None:
        raise EmbeddingContractError("embedding profile ID is not a canonical ep-sha256 identifier")
    return value
