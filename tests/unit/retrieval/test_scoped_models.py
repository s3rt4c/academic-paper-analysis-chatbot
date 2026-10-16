"""Frozen scoped wire admission rejects coercion and contradictory policies."""

import importlib

import pytest
from pydantic import ValidationError


def models():
    assert importlib.util.find_spec("academic_chatbot.retrieval.scoped_models"), (
        "source-scoped contracts are absent"
    )
    return importlib.import_module("academic_chatbot.retrieval.scoped_models")


def request(**updates):
    value = dict(
        scope=dict(
            project_id="p", paper_id="paper", file_version_id="fv", document_generation_id="dg"
        ),
        query="  Exact query Ã¼  ",
        mode="lexical",
        policy=dict(
            profile_id="source-scoped-baseline-v1",
            lexical_policy_id="lexical-natural-language-v1",
            semantic_query_policy_id=None,
            fusion_profile_id=None,
        ),
        semantic_identity=None,
    )
    value.update(updates)
    return models().ScopedRetrievalRequest.model_validate(value)


def test_request_exact_query_and_canonical_binding():
    item = request()
    assert item.query == "  Exact query Ã¼  "
    assert item.limit == 10
    assert len(item.fingerprint) == 64
    assert item.fingerprint != request(limit=11).fingerprint


@pytest.mark.parametrize(
    "updates",
    [
        dict(limit=True),
        dict(limit=0),
        dict(limit=101),
        dict(limit="10"),
        dict(query=" "),
        dict(query="Ã¼" * 2049),
        dict(query="\ud800"),
        dict(unknown=True),
        dict(mode="semantic"),
        dict(mode="hybrid"),
        dict(
            scope=dict(
                project_id="../p",
                paper_id="paper",
                file_version_id="fv",
                document_generation_id="dg",
            )
        ),
        dict(
            semantic_identity=dict(
                embedding_profile_id="e",
                embedding_profile_sha256="a" * 64,
                vector_generation_id="v",
                source_snapshot_sha256="b" * 64,
                vector_store_manifest_sha256="c" * 64,
            )
        ),
    ],
)
def test_request_rejects_invalid(updates):
    m = models()
    with pytest.raises(ValidationError):
        request(**updates)
    assert m.ScopedRetrievalError("INVALID_REQUEST").code == "INVALID_REQUEST"


def test_result_channels_are_consistent_and_unknown_fields_rejected():
    m = models()
    r = request()
    value = dict(
        request_fingerprint=r.fingerprint,
        source=r.scope,
        query=r.query,
        limit=r.limit,
        policy=r.policy,
        semantic_identity=None,
        channels=dict(
            lexical=dict(state="empty", requested_limit=10, returned_count=0),
            semantic=dict(state="not_run", requested_limit=None, returned_count=0),
        ),
        hits=(),
    )
    result = m.ScopedLexicalResult(**value)
    assert result.retrieval_completeness == "not_assessed"
    for changes in [
        dict(hits=(object(),)),
        dict(extra=1),
        dict(
            channels=dict(
                lexical=dict(state="nonempty", requested_limit=10, returned_count=0),
                semantic=dict(state="not_run", requested_limit=None, returned_count=0),
            )
        ),
    ]:
        with pytest.raises(ValidationError):
            m.ScopedLexicalResult(**(value | changes))


@pytest.mark.parametrize(
    "updates",
    [
        {"raw_bm25_score": float("nan")},
        {"raw_bm25_score": float("inf")},
        {"raw_bm25_score": True},
        {"raw_bm25_score": "nan"},
        {"rank": True},
        {"paper_id": "foreign"},
        {"chunk_id": "../unsafe"},
        {"end_offset": 0},
    ],
)
def test_result_revalidates_raw_nested_legacy_packets(updates):
    from tests.fixtures.evidence_bundle.contracts import lexical_hit

    m = models()
    hit = lexical_hit().model_copy(update={"rank": 1})
    scope = dict(
        project_id=hit.project_id,
        paper_id=hit.paper_id,
        file_version_id=hit.file_version_id,
        document_generation_id=hit.document_generation_id,
    )
    r = request(scope=scope, query="Alpha", limit=1)
    with pytest.raises(ValidationError):
        m.ScopedLexicalResult(
            request_fingerprint=r.fingerprint,
            source=r.scope,
            query=r.query,
            limit=1,
            policy=r.policy,
            semantic_identity=None,
            channels=dict(
                lexical=dict(state="nonempty", requested_limit=1, returned_count=1),
                semantic=dict(state="not_run", requested_limit=None, returned_count=0),
            ),
            hits=(hit.model_dump() | updates,),
        )


def test_result_revalidates_constructed_legacy_anchor_hashes():
    from tests.fixtures.evidence_bundle.contracts import lexical_hit

    m = models()
    hit = lexical_hit().model_copy(update={"rank": 1})
    hit = hit.model_copy(
        update={
            "anchors": (
                hit.anchors[0].model_copy(update={"anchor_text_sha256": "0" * 64}),
                *hit.anchors[1:],
            )
        }
    )
    r = request(
        scope=dict(
            project_id=hit.project_id,
            paper_id=hit.paper_id,
            file_version_id=hit.file_version_id,
            document_generation_id=hit.document_generation_id,
        ),
        limit=1,
    )
    with pytest.raises(ValidationError):
        m.ScopedLexicalResult(
            request_fingerprint=r.fingerprint,
            source=r.scope,
            query=r.query,
            limit=1,
            policy=r.policy,
            semantic_identity=None,
            channels=dict(
                lexical=dict(state="nonempty", requested_limit=1, returned_count=1),
                semantic=dict(state="not_run", requested_limit=None, returned_count=0),
            ),
            hits=(hit,),
        )


def test_hybrid_result_rejects_trace_profile_and_channel_rank_mismatch():
    from tests.fixtures.evidence_bundle.contracts import PROFILE, VECTOR, hybrid_hit

    m = models()
    hit = hybrid_hit()
    scope = dict(
        project_id="project-1",
        paper_id="paper-1",
        file_version_id="file-1",
        document_generation_id="generation-1",
    )
    identity = m.SemanticIdentity(
        embedding_profile_id=PROFILE,
        vector_generation_id=VECTOR,
        embedding_profile_sha256="a" * 64,
        source_snapshot_sha256="b" * 64,
        vector_store_manifest_sha256="c" * 64,
    )
    r = m.ScopedRetrievalRequest(
        scope=scope,
        query="Alpha",
        mode="hybrid",
        limit=10,
        policy=m.policy_for("hybrid"),
        semantic_identity=identity,
    )
    for packet in (
        hit,
        hit.model_copy(
            update={"trace": hit.trace.model_copy(update={"fusion_profile_id": "other"})}
        ),
    ):
        with pytest.raises(ValidationError):
            m.ScopedHybridResult(
                request_fingerprint=r.fingerprint,
                source=r.scope,
                query=r.query,
                limit=10,
                policy=r.policy,
                semantic_identity=identity,
                channels=dict(
                    lexical=dict(state="nonempty", requested_limit=50, returned_count=1),
                    semantic=dict(state="nonempty", requested_limit=50, returned_count=1),
                ),
                hits=(packet,),
            )


def test_hybrid_result_rejects_empty_channels_and_forged_rational_vote():
    from tests.fixtures.evidence_bundle.contracts import PROFILE, VECTOR, hybrid_hit

    m = models()
    hit = hybrid_hit()
    identity = m.SemanticIdentity(
        embedding_profile_id=PROFILE,
        vector_generation_id=VECTOR,
        embedding_profile_sha256="a" * 64,
        source_snapshot_sha256="b" * 64,
        vector_store_manifest_sha256="c" * 64,
    )
    r = m.ScopedRetrievalRequest(
        scope=dict(
            project_id="project-1",
            paper_id="paper-1",
            file_version_id="file-1",
            document_generation_id="generation-1",
        ),
        query="Alpha",
        mode="hybrid",
        limit=10,
        policy=m.policy_for("hybrid"),
        semantic_identity=identity,
    )
    for packet, count in (
        (hit, 0),
        (
            hit.model_copy(
                update={
                    "trace": hit.trace.model_copy(
                        update={
                            "fusion_score": hit.trace.fusion_score.model_copy(
                                update={"numerator": 1, "denominator": 2}
                            )
                        }
                    )
                }
            ),
            50,
        ),
    ):
        with pytest.raises(ValidationError):
            m.ScopedHybridResult(
                source=r.scope,
                query=r.query,
                limit=r.limit,
                policy=r.policy,
                request_fingerprint=r.fingerprint,
                semantic_identity=identity,
                hits=(packet,),
                channels=dict(
                    lexical=dict(
                        state="empty" if count == 0 else "nonempty",
                        requested_limit=50,
                        returned_count=count,
                    ),
                    semantic=dict(
                        state="empty" if count == 0 else "nonempty",
                        requested_limit=50,
                        returned_count=count,
                    ),
                ),
            )
