"""Pure packet conversion without retrieval, files, vectors or SQLite."""

from hashlib import sha256
from importlib import import_module

import pytest

from tests.fixtures.evidence_bundle.contracts import hybrid_hit, lexical_hit, semantic_hit


def anchor_id(page_id, evidence_id):
    page, evidence = page_id.encode("utf-8"), evidence_id.encode("ascii")
    return (
        "page-anchor-sha256-"
        + sha256(
            len(page).to_bytes(8, "big") + page + len(evidence).to_bytes(8, "big") + evidence
        ).hexdigest()
    )


@pytest.mark.parametrize("channel", ["lexical", "semantic"])
def test_reference_conversion_preserves_existing_ids(channel):
    module = import_module("academic_chatbot.evidence.references")
    hit = lexical_hit() if channel == "lexical" else semantic_hit()
    before = hit.model_dump()
    ref = getattr(module, "candidate_ref_from_" + channel)(hit)
    contribution = getattr(ref, channel)
    assert ref.parent.page_id == hit.page_id and ref.parent.chunk_id == hit.chunk_id
    assert ref.reported_rank == hit.rank
    assert (contribution.start_offset, contribution.end_offset) == (
        hit.start_offset,
        hit.end_offset,
    )
    text = hit.chunk_text if channel == "lexical" else hit.embedding_span_text
    assert contribution.expected_text_sha256 == sha256(text.encode("utf-8")).hexdigest()
    assert contribution.expected_anchor_ids == tuple(
        anchor_id(hit.page_id, a.evidence_id) for a in hit.anchors
    )
    if channel == "semantic":
        assert contribution.embedding_span_id == hit.embedding_span_id
    assert not {"text", "boxes", "score", "printed_page_label"} & contribution.model_dump().keys()
    assert hit.model_dump() == before


@pytest.mark.parametrize("equal", [False, True])
def test_hybrid_conversion_retains_both_ranges(equal):
    module = import_module("academic_chatbot.evidence.references")
    hit = hybrid_hit(equal=equal)
    ref = module.candidate_ref_from_hybrid(hit)
    assert ref.parent == hit.identity
    assert ref.reported_rank == hit.trace.fusion_rank
    assert ref.lexical is not None and ref.semantic is not None
    assert (ref.lexical.start_offset == ref.semantic.start_offset) is equal
    assert (
        ref.semantic.embedding_span_id == hit.semantic_contribution.semantic_hit.embedding_span_id
    )


def test_same_words_distinct_occurrences_remain_distinct():
    module = import_module("academic_chatbot.evidence.references")
    first = lexical_hit()
    second = first.model_copy(update={"page_id": "another-page", "chunk_id": "another-chunk"})
    a, b = module.candidate_ref_from_lexical(first), module.candidate_ref_from_lexical(second)
    assert a.parent != b.parent
    assert a.lexical.expected_text_sha256 == b.lexical.expected_text_sha256
    assert a.lexical.expected_anchor_ids != b.lexical.expected_anchor_ids


def test_conversion_has_no_io(monkeypatch):
    import builtins
    import socket
    import sqlite3

    module = import_module("academic_chatbot.evidence.references")
    lexical, semantic, hybrid = lexical_hit(), semantic_hit(), hybrid_hit()

    def forbidden(*args, **kwargs):
        raise AssertionError("I/O is forbidden during pure conversion")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    assert module.candidate_ref_from_lexical(lexical).lexical is not None
    assert module.candidate_ref_from_semantic(semantic).semantic is not None
    assert module.candidate_ref_from_hybrid(hybrid).reported_rank == 1
