"""CPU-only local ONNX inference for verified embedding artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from academic_chatbot.embeddings.artifacts import (
    VerifiedEmbeddingArtifacts,
    load_verified_artifacts,
)
from academic_chatbot.embeddings.models import (
    EmbeddingContractError,
    EmbeddingProfile,
    EmbeddingRole,
)
from academic_chatbot.embeddings.tokenizer import PreparedEmbeddingBatch, VerifiedTokenizerAdapter


class OfflineEmbedderError(EmbeddingContractError):
    """Stable service-boundary error for local embedding inference."""


class RuntimeContractError(OfflineEmbedderError):
    """Raised when the verified ONNX graph differs from the frozen profile."""


class EmbeddingOutputError(OfflineEmbedderError):
    """Raised when ONNX output cannot safely become an embedding vector."""


class _Node(Protocol):
    name: str
    type: str
    shape: Sequence[int | str | None]


class _Session(Protocol):
    def get_providers(self) -> Sequence[str]: ...

    def get_inputs(self) -> Sequence[_Node]: ...

    def get_outputs(self) -> Sequence[_Node]: ...

    def run(
        self, output_names: Sequence[str], input_feed: dict[str, np.ndarray]
    ) -> Sequence[object]: ...


@dataclass(frozen=True, slots=True)
class OfflineEmbedder:
    """One immutable profile-bound, CPU-only, offline embedding runtime."""

    profile: EmbeddingProfile
    _tokenizer: VerifiedTokenizerAdapter
    _session: _Session

    @classmethod
    def open(cls, model_root: Path, profile: EmbeddingProfile) -> OfflineEmbedder:
        """Verify all local artifacts before constructing tokenizer or ONNX runtime state."""

        tokenizer_artifacts = load_verified_artifacts(model_root, profile=profile)
        tokenizer = VerifiedTokenizerAdapter.open(tokenizer_artifacts)
        # Tokenizers and ONNX Runtime accept pathnames rather than verified
        # handles. Re-verification narrows the unavoidable pathname TOCTOU
        # window before the second third-party consumer opens the model.
        runtime_artifacts = load_verified_artifacts(model_root, profile=profile)
        session = _create_verified_cpu_session(runtime_artifacts)
        _validate_session_contract(session, profile)
        return cls(profile=profile, _tokenizer=tokenizer, _session=session)

    @classmethod
    def _open_verified(cls, artifacts: VerifiedEmbeddingArtifacts) -> OfflineEmbedder:
        profile = artifacts.manifest.embedding_profile
        tokenizer = VerifiedTokenizerAdapter.open(artifacts)
        session = _create_verified_cpu_session(artifacts)
        _validate_session_contract(session, profile)
        return cls(profile=profile, _tokenizer=tokenizer, _session=session)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(EmbeddingRole.DOCUMENT, texts)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(EmbeddingRole.QUERY, texts)

    def _embed(self, role: EmbeddingRole, texts: Sequence[str]) -> np.ndarray:
        prepared = self._tokenizer.prepare(role, texts)
        return self._infer(prepared)

    def _infer(self, prepared: PreparedEmbeddingBatch) -> np.ndarray:
        try:
            outputs = self._session.run(
                [self.profile.onnx_graph.output.name],
                {
                    "input_ids": prepared.input_ids,
                    "attention_mask": prepared.attention_mask,
                    "token_type_ids": prepared.token_type_ids,
                },
            )
        except Exception as error:
            raise OfflineEmbedderError("local ONNX embedding inference failed") from error
        if len(outputs) != 1 or not isinstance(outputs[0], np.ndarray):
            raise EmbeddingOutputError("local ONNX runtime returned an invalid output collection")
        hidden = outputs[0]
        expected_shape = (
            prepared.input_ids.shape[0],
            prepared.input_ids.shape[1],
            self.profile.dimension,
        )
        if hidden.shape != expected_shape or hidden.dtype != np.dtype(np.float32):
            raise EmbeddingOutputError(
                "local ONNX output does not match the frozen hidden-state shape"
            )
        if not np.isfinite(hidden).all():
            raise EmbeddingOutputError("local ONNX output contains non-finite values")
        pooled = np.asarray(hidden[:, 0, :], dtype=np.float32)
        norms = np.sqrt(np.sum(pooled * pooled, axis=1, dtype=np.float32), dtype=np.float32)
        if np.any(~np.isfinite(norms)) or np.any(norms <= np.float32(0.0)):
            raise EmbeddingOutputError("local ONNX CLS output has an invalid L2 norm")
        normalized = np.asarray(pooled / norms[:, np.newaxis], dtype=np.float32)
        if normalized.shape != (prepared.input_ids.shape[0], self.profile.dimension):
            raise EmbeddingOutputError("normalized embedding output has an invalid shape")
        if not np.isfinite(normalized).all() or not np.allclose(
            np.sum(normalized * normalized, axis=1, dtype=np.float32),
            np.ones(normalized.shape[0], dtype=np.float32),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise EmbeddingOutputError(
                "normalized embedding output is not finite unit-length float32"
            )
        return normalized


def _create_verified_cpu_session(artifacts: VerifiedEmbeddingArtifacts) -> _Session:
    profile = artifacts.manifest.embedding_profile
    if profile.onnx_runtime.provider != "CPUExecutionProvider":
        raise RuntimeContractError("embedding profile does not require CPUExecutionProvider")
    if profile.onnx_runtime.requirement != f"onnxruntime=={ort.__version__}":
        raise RuntimeContractError("installed ONNX Runtime does not match the embedding profile")
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeContractError("CPUExecutionProvider is not available")
    try:
        session = ort.InferenceSession(
            str(artifacts.artifact_path("onnx/model.onnx")),
            providers=["CPUExecutionProvider"],
        )
    except Exception as error:
        raise RuntimeContractError("verified local ONNX model could not be opened") from error
    if tuple(session.get_providers()) != ("CPUExecutionProvider",):
        raise RuntimeContractError("ONNX session did not bind exactly CPUExecutionProvider")
    return cast(_Session, session)


def _validate_session_contract(session: _Session, profile: EmbeddingProfile) -> None:
    expected_inputs = tuple(item.name for item in profile.onnx_graph.inputs)
    inputs = tuple(session.get_inputs())
    if tuple(item.name for item in inputs) != expected_inputs or len(inputs) != 3:
        raise RuntimeContractError("ONNX graph inputs do not match the embedding profile")
    if any(item.type != "tensor(int64)" or len(item.shape) != 2 for item in inputs):
        raise RuntimeContractError(
            "ONNX graph input tensors do not match the frozen int64 contract"
        )
    outputs = tuple(session.get_outputs())
    if len(outputs) != 1 or outputs[0].name != profile.onnx_graph.output.name:
        raise RuntimeContractError("ONNX graph output does not match the embedding profile")
    output = outputs[0]
    if (
        output.type != "tensor(float)"
        or len(output.shape) != 3
        or output.shape[-1] != profile.dimension
    ):
        raise RuntimeContractError(
            "ONNX graph output dimension or dtype does not match the profile"
        )
