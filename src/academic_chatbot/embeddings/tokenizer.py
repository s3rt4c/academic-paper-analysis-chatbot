"""Verified local tokenization for the frozen Phase 1B embedding profile."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers import __version__ as TOKENIZERS_VERSION

from academic_chatbot.embeddings.artifacts import VerifiedEmbeddingArtifacts
from academic_chatbot.embeddings.models import (
    FROZEN_MAX_SEQUENCE_LENGTH,
    EmbeddingContractError,
    EmbeddingProfile,
    EmbeddingRole,
)


class EmbeddingInputError(EmbeddingContractError):
    """Raised when an embedding caller provides invalid meaningful input."""


class EmbeddingInputTooLongError(EmbeddingInputError):
    """Raised instead of silently truncating a document or query."""


class TokenizerAdapterError(EmbeddingContractError):
    """Raised when the verified tokenizer cannot satisfy the frozen profile."""


class _Encoding(Protocol):
    ids: Sequence[int]
    attention_mask: Sequence[int]
    type_ids: Sequence[int]


class _Tokenizer(Protocol):
    def no_truncation(self) -> None: ...

    def token_to_id(self, value: str) -> int | None: ...

    def encode(self, value: str, *, add_special_tokens: bool) -> _Encoding: ...


@dataclass(frozen=True, slots=True)
class PreparedEmbeddingBatch:
    """The exact three int64 tensors accepted by the frozen ONNX graph."""

    input_ids: np.ndarray
    attention_mask: np.ndarray
    token_type_ids: np.ndarray

    def __post_init__(self) -> None:
        tensors = (self.input_ids, self.attention_mask, self.token_type_ids)
        if not tensors[0].size or any(item.dtype != np.dtype(np.int64) for item in tensors):
            raise TokenizerAdapterError("prepared embedding tensors must be non-empty int64 arrays")
        if any(item.ndim != 2 or item.shape != tensors[0].shape for item in tensors):
            raise TokenizerAdapterError(
                "prepared embedding tensors must have matching rank-two shapes"
            )
        if tensors[0].shape[1] > FROZEN_MAX_SEQUENCE_LENGTH:
            raise TokenizerAdapterError("prepared embedding tensors must not exceed 512 positions")


@dataclass(frozen=True, slots=True)
class VerifiedTokenizerAdapter:
    """Role-explicit tokenizer loaded from a Task 1 verified local artifact set."""

    profile: EmbeddingProfile
    _tokenizer: _Tokenizer
    _pad_token_id: int

    @classmethod
    def open(cls, artifacts: VerifiedEmbeddingArtifacts) -> VerifiedTokenizerAdapter:
        profile = artifacts.manifest.embedding_profile
        if profile.tokenizer.runtime != f"tokenizers=={TOKENIZERS_VERSION}":
            raise TokenizerAdapterError(
                "installed tokenizers runtime does not match the embedding profile"
            )
        try:
            tokenizer = Tokenizer.from_file(
                str(artifacts.artifact_path(profile.tokenizer.artifact))
            )
            tokenizer.no_truncation()
            pad_token_id = tokenizer.token_to_id("[PAD]")
        except Exception as error:
            raise TokenizerAdapterError("verified local tokenizer could not be opened") from error
        if pad_token_id is None:
            raise TokenizerAdapterError("verified tokenizer does not expose the required PAD token")
        return cls(profile=profile, _tokenizer=tokenizer, _pad_token_id=pad_token_id)

    def prepare(self, role: EmbeddingRole, texts: Sequence[str]) -> PreparedEmbeddingBatch:
        if not isinstance(role, EmbeddingRole):
            raise TokenizerAdapterError("embedding role must be an explicit EmbeddingRole value")
        if not texts:
            raise EmbeddingInputError("embedding input batch must not be empty")
        encodings = tuple(self._encode(role, value) for value in texts)
        maximum = max(len(item.ids) for item in encodings)
        if maximum > self.profile.max_sequence_length:
            raise TokenizerAdapterError("prepared sequence exceeds the embedding profile maximum")
        input_ids = np.full((len(encodings), maximum), self._pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((len(encodings), maximum), dtype=np.int64)
        token_type_ids = np.zeros((len(encodings), maximum), dtype=np.int64)
        for index, encoding in enumerate(encodings):
            ids = np.asarray(encoding.ids, dtype=np.int64)
            mask = np.asarray(encoding.attention_mask, dtype=np.int64)
            types = np.asarray(encoding.type_ids, dtype=np.int64)
            if ids.ndim != 1 or mask.shape != ids.shape or types.shape != ids.shape:
                raise TokenizerAdapterError("tokenizer returned inconsistent token fields")
            input_ids[index, : ids.size] = ids
            attention_mask[index, : mask.size] = mask
            token_type_ids[index, : types.size] = types
        return PreparedEmbeddingBatch(input_ids, attention_mask, token_type_ids)

    def _encode(self, role: EmbeddingRole, text: str) -> _Encoding:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingInputError("embedding input text must not be empty or whitespace-only")
        try:
            source = self._tokenizer.encode(text, add_special_tokens=False)
        except Exception as error:
            raise TokenizerAdapterError("verified tokenizer rejected embedding input") from error
        budget = self.profile.role_source_budget(role)
        if len(source.ids) > budget:
            label = "document" if role is EmbeddingRole.DOCUMENT else "query"
            raise EmbeddingInputTooLongError(f"{label} exceeds embedding profile token budget")
        prepared_text = self.profile.role_prefix(role) + text
        try:
            complete = self._tokenizer.encode(prepared_text, add_special_tokens=True)
        except Exception as error:
            raise TokenizerAdapterError(
                "verified tokenizer could not prepare embedding input"
            ) from error
        if len(complete.ids) > self.profile.max_sequence_length:
            label = "document" if role is EmbeddingRole.DOCUMENT else "query"
            raise EmbeddingInputTooLongError(f"{label} exceeds embedding profile token budget")
        if not complete.ids:
            raise TokenizerAdapterError("verified tokenizer produced an empty sequence")
        return complete
