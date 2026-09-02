from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from academic_chatbot.embeddings.artifacts import EmbeddingArtifactPaths
from academic_chatbot.embeddings.models import (
    ArtifactFile,
    EmbeddingProfile,
    EmbeddingSpanIdentity,
    artifact_set_sha256_for,
)
from academic_chatbot.embeddings.profile import canonical_profile_bytes, embedding_profile_id_for

_FROZEN_PROFILE_ID = "ep-sha256-3f8fd2dbcff088eb61b2ef1ecbc6de57644a425722a586fef32059516146a929"


def test_frozen_task0_profile_reproduces_its_exact_identity(
    frozen_profile_payload: dict[str, object],
) -> None:
    profile = EmbeddingProfile.model_validate(frozen_profile_payload)

    assert embedding_profile_id_for(profile) == _FROZEN_PROFILE_ID
    assert profile.embedding_profile_id == _FROZEN_PROFILE_ID
    assert canonical_profile_bytes(profile).startswith(b'{"artifact_set_sha256"')


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("model_revision",), "a" * 40),
        (("artifacts", 4, "sha256"), "b" * 64),
        (("artifacts", 7, "sha256"), "c" * 64),
        (("query_prefix_utf8",), "different prefix"),
        (("pooling",), "mean_pooling"),
        (("normalization", "rule"), "none"),
        (("dimension",), 768),
        (("special_token_policy", "document_content_token_budget"), 509),
        (("span_policy",), "canonical-word-greedy-v2"),
    ],
)
def test_semantic_profile_changes_change_identity(
    mutable_frozen_profile_payload: dict[str, object],
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    cursor: object = mutable_frozen_profile_payload
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]

    if path == ("dimension",):
        mutable_frozen_profile_payload["onnx_graph"]["output"][4] = replacement  # type: ignore[index]
    if path == ("special_token_policy", "document_content_token_budget"):
        mutable_frozen_profile_payload["max_sequence_length"] = 511
        mutable_frozen_profile_payload["special_token_policy"]["query_source_token_budget"] = 501  # type: ignore[index]

    mutable_frozen_profile_payload["artifact_set_sha256"] = artifact_set_sha256_for(
        tuple(
            ArtifactFile.model_validate(item)
            for item in mutable_frozen_profile_payload["artifacts"]
        )  # type: ignore[arg-type]
    )

    assert embedding_profile_id_for(
        EmbeddingProfile.model_validate(mutable_frozen_profile_payload)
    ) != ("ep-sha256-3f8fd2dbcff088eb61b2ef1ecbc6de57644a425722a586fef32059516146a929")


def test_profile_has_no_installation_path_or_host_fields(
    frozen_profile_payload: dict[str, object],
) -> None:
    profile = EmbeddingProfile.model_validate(frozen_profile_payload)

    assert "path" not in profile.model_fields_set
    assert "hostname" not in profile.model_fields_set
    assert "username" not in profile.model_fields_set


def test_local_model_roots_do_not_change_semantic_profile_identity(
    tmp_path: Path, frozen_profile_payload: dict[str, object]
) -> None:
    profile = EmbeddingProfile.model_validate(frozen_profile_payload)

    first = EmbeddingArtifactPaths.create(tmp_path / "alice-host", profile=profile)
    second = EmbeddingArtifactPaths.create(tmp_path / "bob-host", profile=profile)

    assert first.profile.embedding_profile_id == second.profile.embedding_profile_id


def test_profile_rejects_inconsistent_budget_or_dimension(
    mutable_frozen_profile_payload: dict[str, object],
) -> None:
    mutable_frozen_profile_payload["dimension"] = 385

    with pytest.raises(ValidationError, match="dimension"):
        EmbeddingProfile.model_validate(mutable_frozen_profile_payload)


@pytest.mark.parametrize("artifact", ["C:/tokenizer.json", "../tokenizer.json", "missing.json"])
def test_profile_rejects_unsafe_or_uninventoried_tokenizer_artifact(
    mutable_frozen_profile_payload: dict[str, object], artifact: str
) -> None:
    mutable_frozen_profile_payload["tokenizer"]["artifact"] = artifact  # type: ignore[index]

    with pytest.raises(ValidationError, match="artifact"):
        EmbeddingProfile.model_validate(mutable_frozen_profile_payload)


def test_span_identity_is_deterministic_and_occurrence_specific() -> None:
    common = {
        "document_generation_id": "dg-1",
        "chunk_id": "chunk-1",
        "page_id": "page-1",
        "start_offset": 10,
        "end_offset": 20,
        "embedding_profile_id": _FROZEN_PROFILE_ID,
    }

    first = EmbeddingSpanIdentity.model_validate(common)
    repeated = EmbeddingSpanIdentity.model_validate(common)
    other_occurrence = EmbeddingSpanIdentity.model_validate(
        {**common, "start_offset": 30, "end_offset": 40}
    )

    assert first.embedding_span_id == repeated.embedding_span_id
    assert first.embedding_span_id != other_occurrence.embedding_span_id


@pytest.mark.parametrize(
    "payload",
    [
        {
            "document_generation_id": "",
            "chunk_id": "c",
            "page_id": "p",
            "start_offset": 0,
            "end_offset": 1,
            "embedding_profile_id": _FROZEN_PROFILE_ID,
        },
        {
            "document_generation_id": "d",
            "chunk_id": "c",
            "page_id": "p",
            "start_offset": 1,
            "end_offset": 1,
            "embedding_profile_id": _FROZEN_PROFILE_ID,
        },
        {
            "document_generation_id": "d",
            "chunk_id": "c",
            "page_id": "p",
            "start_offset": 0,
            "end_offset": 1,
            "embedding_profile_id": "not-a-profile",
        },
    ],
)
def test_span_identity_rejects_invalid_lineage_or_range(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        EmbeddingSpanIdentity.model_validate(payload)
