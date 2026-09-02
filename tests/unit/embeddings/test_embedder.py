from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from academic_chatbot.embeddings.artifacts import (
    EmbeddingArtifactError,
    VerifiedEmbeddingArtifacts,
)
from academic_chatbot.embeddings.embedder import (
    EmbeddingOutputError,
    OfflineEmbedder,
    RuntimeContractError,
)
from academic_chatbot.embeddings.models import EmbeddingArtifactManifest, EmbeddingProfile
from academic_chatbot.embeddings.tokenizer import EmbeddingInputTooLongError, PreparedEmbeddingBatch


@dataclass(frozen=True)
class _Node:
    name: str
    type: str
    shape: list[int | str | None]


class _Session:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.calls: list[dict[str, np.ndarray]] = []

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def get_inputs(self) -> list[_Node]:
        return [
            _Node("input_ids", "tensor(int64)", ["batch", "sequence"]),
            _Node("attention_mask", "tensor(int64)", ["batch", "sequence"]),
            _Node("token_type_ids", "tensor(int64)", ["batch", "sequence"]),
        ]

    def get_outputs(self) -> list[_Node]:
        return [_Node("last_hidden_state", "tensor(float)", ["batch", "sequence", 384])]

    def run(self, names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append(inputs)
        assert names == ["last_hidden_state"]
        return [self.output]


class _TokenizerAdapter:
    def prepare(self, role: object, texts: object) -> object:
        return _prepared(1, 4)


@pytest.fixture
def artifacts(
    tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> VerifiedEmbeddingArtifacts:
    profile = EmbeddingProfile.model_validate(frozen_profile_payload)
    manifest = EmbeddingArtifactManifest(
        manifest_schema_version="embedding-artifact-manifest-v1",
        embedding_profile=profile,
        embedding_profile_id=profile.embedding_profile_id,
        declared_license="MIT",
        artifacts=profile.artifacts,
    )
    return VerifiedEmbeddingArtifacts(
        manifest=manifest,
        _artifact_paths=MappingProxyType(
            {
                "tokenizer.json": tmp_path / "tokenizer.json",
                "onnx/model.onnx": tmp_path / "model.onnx",
            }
        ),
    )


def test_embedder_cls_pools_and_l2_normalizes_float32(
    monkeypatch: pytest.MonkeyPatch, artifacts: VerifiedEmbeddingArtifacts
) -> None:
    from academic_chatbot.embeddings import embedder as module

    hidden = np.zeros((1, 4, 384), dtype=np.float32)
    hidden[0, 0, :2] = [3.0, 4.0]
    hidden[0, 1, 0] = 99.0
    session = _Session(hidden)
    monkeypatch.setattr(module, "_create_verified_cpu_session", lambda *_: session)
    monkeypatch.setattr(module.VerifiedTokenizerAdapter, "open", lambda _: _TokenizerAdapter())
    embedder = OfflineEmbedder._open_verified(artifacts)

    result = embedder.embed_documents(["source"])

    assert result.dtype == np.float32
    assert result.shape == (1, 384)
    assert np.allclose(result[0, :2], [0.6, 0.8])
    assert np.linalg.norm(result[0]) == pytest.approx(1.0)
    assert session.calls[0]["input_ids"].dtype == np.int64


@pytest.mark.parametrize(
    "value",
    [np.zeros((1, 4, 384), dtype=np.float32), np.full((1, 4, 384), np.nan, dtype=np.float32)],
)
def test_embedder_rejects_zero_or_nonfinite_cls(
    monkeypatch: pytest.MonkeyPatch, artifacts: VerifiedEmbeddingArtifacts, value: np.ndarray
) -> None:
    from academic_chatbot.embeddings import embedder as module

    monkeypatch.setattr(module, "_create_verified_cpu_session", lambda *_: _Session(value))
    monkeypatch.setattr(module.VerifiedTokenizerAdapter, "open", lambda _: _TokenizerAdapter())
    with pytest.raises(EmbeddingOutputError):
        OfflineEmbedder._open_verified(artifacts).embed_documents(["source"])


def test_embedder_rejects_runtime_graph_contract(
    monkeypatch: pytest.MonkeyPatch, artifacts: VerifiedEmbeddingArtifacts
) -> None:
    from academic_chatbot.embeddings import embedder as module

    session = _Session(np.ones((1, 4, 384), dtype=np.float32))
    monkeypatch.setattr(
        session, "get_outputs", lambda: [_Node("wrong", "tensor(float)", [1, 4, 384])]
    )
    monkeypatch.setattr(module, "_create_verified_cpu_session", lambda *_: session)
    monkeypatch.setattr(module.VerifiedTokenizerAdapter, "open", lambda _: _TokenizerAdapter())
    with pytest.raises(RuntimeContractError, match="output"):
        OfflineEmbedder._open_verified(artifacts)


@pytest.mark.parametrize(
    "node",
    [
        _Node("wrong_input", "tensor(int64)", ["batch", "sequence"]),
        _Node("input_ids", "tensor(float)", ["batch", "sequence"]),
    ],
)
def test_embedder_rejects_wrong_input_contract(
    monkeypatch: pytest.MonkeyPatch, artifacts: VerifiedEmbeddingArtifacts, node: _Node
) -> None:
    from academic_chatbot.embeddings import embedder as module

    session = _Session(np.ones((1, 4, 384), dtype=np.float32))
    monkeypatch.setattr(
        session,
        "get_inputs",
        lambda: [node, *_Session(np.ones((1, 4, 384), dtype=np.float32)).get_inputs()[1:]],
    )
    monkeypatch.setattr(module, "_create_verified_cpu_session", lambda *_: session)
    monkeypatch.setattr(module.VerifiedTokenizerAdapter, "open", lambda _: _TokenizerAdapter())
    with pytest.raises(RuntimeContractError, match="input"):
        OfflineEmbedder._open_verified(artifacts)


def test_public_open_reverifies_before_the_onnx_path_is_opened(
    monkeypatch: pytest.MonkeyPatch, artifacts: VerifiedEmbeddingArtifacts
) -> None:
    from academic_chatbot.embeddings import embedder as module

    calls: list[object] = [artifacts, EmbeddingArtifactError("artifact changed after verification")]

    def load(*_args: object, **_kwargs: object) -> VerifiedEmbeddingArtifacts:
        next_value = calls.pop(0)
        if isinstance(next_value, Exception):
            raise next_value
        return next_value

    monkeypatch.setattr(module, "load_verified_artifacts", load)
    monkeypatch.setattr(module.VerifiedTokenizerAdapter, "open", lambda _: _TokenizerAdapter())
    monkeypatch.setattr(
        module,
        "_create_verified_cpu_session",
        lambda _: pytest.fail("runtime must not open after failed re-verification"),
    )
    with pytest.raises(EmbeddingArtifactError, match="changed"):
        OfflineEmbedder.open(Path("model-root"), artifacts.manifest.embedding_profile)


def _prepared(batch_size: int, sequence_length: int) -> object:
    from academic_chatbot.embeddings.tokenizer import PreparedEmbeddingBatch

    values = np.ones((batch_size, sequence_length), dtype=np.int64)
    return PreparedEmbeddingBatch(values, values, values)


def test_prepared_batch_rejects_over_512_positions() -> None:
    from academic_chatbot.embeddings.tokenizer import TokenizerAdapterError

    values = np.ones((1, 513), dtype=np.int64)
    with pytest.raises(TokenizerAdapterError, match="512"):
        PreparedEmbeddingBatch(values, values, values)


def test_invalid_batch_member_prevents_onnx_inference(
    artifacts: VerifiedEmbeddingArtifacts,
) -> None:
    class RejectingTokenizer:
        def prepare(self, role: object, texts: object) -> PreparedEmbeddingBatch:
            raise EmbeddingInputTooLongError("document exceeds embedding profile token budget")

    session = _Session(np.ones((1, 4, 384), dtype=np.float32))
    embedder = OfflineEmbedder(artifacts.manifest.embedding_profile, RejectingTokenizer(), session)  # type: ignore[arg-type]
    with pytest.raises(EmbeddingInputTooLongError):
        embedder.embed_documents(["valid", "over budget"])
    assert session.calls == []
