from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from academic_chatbot.embeddings.artifacts import VerifiedEmbeddingArtifacts
from academic_chatbot.embeddings.models import (
    EmbeddingArtifactManifest,
    EmbeddingProfile,
    EmbeddingRole,
)
from academic_chatbot.embeddings.tokenizer import (
    EmbeddingInputError,
    EmbeddingInputTooLongError,
    TokenizerAdapterError,
    VerifiedTokenizerAdapter,
)


@dataclass(frozen=True)
class _Encoding:
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class _Tokenizer:
    def __init__(self, profile: EmbeddingProfile) -> None:
        self.profile = profile
        self.truncation_disabled = False

    def no_truncation(self) -> None:
        self.truncation_disabled = True

    def token_to_id(self, value: str) -> int | None:
        return 0 if value == "[PAD]" else None

    def encode(self, value: str, *, add_special_tokens: bool) -> _Encoding:
        prefix = self.profile.query_prefix_utf8
        if value == prefix:
            token_count = 8
        elif value == "document-510":
            token_count = 510
        elif value == "document-511":
            token_count = 511
        elif value == "query-502":
            token_count = 502
        elif value == "query-503":
            token_count = 503
        elif value == prefix + "query-502":
            token_count = 510
        elif value == prefix + "query-503":
            token_count = 511
        else:
            token_count = len(value.split())
        ids = list(range(1000, 1000 + token_count))
        if add_special_tokens:
            ids = [101, *ids, 102]
        return _Encoding(ids=ids, attention_mask=[1] * len(ids), type_ids=[0] * len(ids))


@pytest.fixture
def adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> VerifiedTokenizerAdapter:
    from academic_chatbot.embeddings import tokenizer as module

    profile = EmbeddingProfile.model_validate(frozen_profile_payload)
    manifest = EmbeddingArtifactManifest(
        manifest_schema_version="embedding-artifact-manifest-v1",
        embedding_profile=profile,
        embedding_profile_id=profile.embedding_profile_id,
        declared_license="MIT",
        artifacts=profile.artifacts,
    )
    artifacts = VerifiedEmbeddingArtifacts(
        manifest=manifest,
        _artifact_paths=MappingProxyType({"tokenizer.json": tmp_path / "tokenizer.json"}),
    )
    fake = _Tokenizer(profile)
    monkeypatch.setattr(module, "Tokenizer", type("Loader", (), {"from_file": lambda _: fake}))
    result = VerifiedTokenizerAdapter.open(artifacts)
    assert fake.truncation_disabled
    return result


def test_document_and_query_use_exact_role_preprocessing(adapter: VerifiedTokenizerAdapter) -> None:
    document = adapter.prepare(EmbeddingRole.DOCUMENT, ["same input"])
    query = adapter.prepare(EmbeddingRole.QUERY, ["same input"])

    assert document.input_ids.dtype == np.dtype(np.int64)
    assert query.input_ids.dtype == np.dtype(np.int64)
    assert document.input_ids[0].tolist() != query.input_ids[0].tolist()
    assert document.input_ids[0, 0] == 101
    assert query.input_ids[0, 0] == 101


def test_unicode_preparation_is_deterministic(adapter: VerifiedTokenizerAdapter) -> None:
    first = adapter.prepare(EmbeddingRole.DOCUMENT, ["café 中文 evidence"])
    second = adapter.prepare(EmbeddingRole.DOCUMENT, ["café 中文 evidence"])

    assert np.array_equal(first.input_ids, second.input_ids)
    assert np.array_equal(first.attention_mask, second.attention_mask)
    assert np.array_equal(first.token_type_ids, second.token_type_ids)


def test_prepare_rejects_a_non_embedding_role(adapter: VerifiedTokenizerAdapter) -> None:
    with pytest.raises(TokenizerAdapterError, match="role"):
        adapter.prepare("document", ["evidence"])  # type: ignore[arg-type]


def test_document_budget_accepts_510_and_rejects_511(adapter: VerifiedTokenizerAdapter) -> None:
    assert adapter.prepare(EmbeddingRole.DOCUMENT, ["document-510"]).input_ids.shape == (1, 512)
    with pytest.raises(EmbeddingInputTooLongError, match="document exceeds"):
        adapter.prepare(EmbeddingRole.DOCUMENT, ["document-511"])


def test_query_budget_accepts_actual_complete_sequence_and_rejects_overage(
    adapter: VerifiedTokenizerAdapter,
) -> None:
    assert adapter.prepare(EmbeddingRole.QUERY, ["query-502"]).input_ids.shape == (1, 512)
    with pytest.raises(EmbeddingInputTooLongError, match="query exceeds"):
        adapter.prepare(EmbeddingRole.QUERY, ["query-503"])


def test_batch_padding_and_invalid_members_fail_before_preparation(
    adapter: VerifiedTokenizerAdapter,
) -> None:
    batch = adapter.prepare(EmbeddingRole.DOCUMENT, ["one", "one two"])
    assert batch.input_ids.shape == batch.attention_mask.shape == batch.token_type_ids.shape
    assert (
        batch.input_ids.dtype
        == batch.attention_mask.dtype
        == batch.token_type_ids.dtype
        == np.int64
    )
    assert batch.attention_mask[0, -1] == 0
    with pytest.raises(EmbeddingInputError, match="empty"):
        adapter.prepare(EmbeddingRole.DOCUMENT, ["one", " "])


def test_open_rejects_a_tokenizers_runtime_version_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> None:
    from academic_chatbot.embeddings import tokenizer as module

    profile = EmbeddingProfile.model_validate(frozen_profile_payload)
    manifest = EmbeddingArtifactManifest(
        manifest_schema_version="embedding-artifact-manifest-v1",
        embedding_profile=profile,
        embedding_profile_id=profile.embedding_profile_id,
        declared_license="MIT",
        artifacts=profile.artifacts,
    )
    artifacts = VerifiedEmbeddingArtifacts(
        manifest, MappingProxyType({"tokenizer.json": tmp_path / "x"})
    )
    monkeypatch.setattr(module, "TOKENIZERS_VERSION", "different")
    with pytest.raises(TokenizerAdapterError, match="runtime"):
        VerifiedTokenizerAdapter.open(artifacts)
