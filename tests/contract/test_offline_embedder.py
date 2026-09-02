from __future__ import annotations

import os
import socket
from pathlib import Path

import numpy as np
import pytest

from academic_chatbot.embeddings.embedder import OfflineEmbedder
from academic_chatbot.embeddings.models import EmbeddingProfile, EmbeddingRole
from academic_chatbot.embeddings.tokenizer import EmbeddingInputTooLongError
from tests.unit.embeddings.conftest import frozen_profile_payload as _frozen_profile_payload

_REAL_ARTIFACT_ROOT = "ACADEMIC_CHATBOT_BGE_ARTIFACT_ROOT"
_DOCUMENT_IDS = [101, 3128, 11135, 3350, 2442, 3961, 4987, 2000, 2049, 6635, 3120, 2846, 1012, 102]
_QUERY_PREFIX_IDS = [5050, 2023, 6251, 2005, 6575, 7882, 13768, 1024]


@pytest.fixture
def frozen_profile_payload() -> dict[str, object]:
    return _frozen_profile_payload.__wrapped__()  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not os.environ.get(_REAL_ARTIFACT_ROOT), reason="requires pinned local BGE artifact root"
)
def test_opt_in_real_bge_embedder_is_verified_cpu_only_and_offline(
    monkeypatch: pytest.MonkeyPatch, frozen_profile_payload: dict[str, object]
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    embedder = OfflineEmbedder.open(
        Path(os.environ[_REAL_ARTIFACT_ROOT]),
        EmbeddingProfile.model_validate(frozen_profile_payload),
    )
    prepared_document = embedder._tokenizer.prepare(
        EmbeddingRole.DOCUMENT,
        ["Native PDF evidence must remain attached to its exact source range."],
    )
    prepared_query = embedder._tokenizer.prepare(
        EmbeddingRole.QUERY, ["How is evidence lineage preserved?"]
    )
    document_at_limit = "a " * 510
    document_over_limit = "a " * 511
    query_at_limit = "a " * 502
    query_over_limit = "a " * 503
    document = embedder.embed_documents(
        ["Native PDF evidence must remain attached to its exact source range."]
    )
    query = embedder.embed_queries(["How is evidence lineage preserved?"])

    assert document.shape == query.shape == (1, 384)
    assert prepared_document.input_ids[0].tolist() == _DOCUMENT_IDS
    assert prepared_query.input_ids[0, 1:9].tolist() == _QUERY_PREFIX_IDS
    assert prepared_query.input_ids.shape == (1, 16)
    document_source_ids = embedder._tokenizer._tokenizer.encode(
        document_at_limit, add_special_tokens=False
    ).ids
    query_source_ids = embedder._tokenizer._tokenizer.encode(
        query_at_limit, add_special_tokens=False
    ).ids
    assert len(document_source_ids) == 510
    assert len(query_source_ids) == 502
    assert embedder._tokenizer.prepare(
        EmbeddingRole.DOCUMENT, [document_at_limit]
    ).input_ids.shape == (1, 512)
    assert embedder._tokenizer.prepare(EmbeddingRole.QUERY, [query_at_limit]).input_ids.shape == (
        1,
        512,
    )
    with pytest.raises(EmbeddingInputTooLongError):
        embedder._tokenizer.prepare(EmbeddingRole.DOCUMENT, [document_over_limit])
    with pytest.raises(EmbeddingInputTooLongError):
        embedder._tokenizer.prepare(EmbeddingRole.QUERY, [query_over_limit])
    assert document.dtype == query.dtype == np.float32
    assert np.isfinite(document).all() and np.isfinite(query).all()
    assert np.linalg.norm(document[0]) == pytest.approx(1.0)
    assert np.linalg.norm(query[0]) == pytest.approx(1.0)
