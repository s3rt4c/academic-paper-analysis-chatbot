"""Small deterministic packets; no document, database or model I/O."""

from hashlib import sha256

from academic_chatbot.documents.native_pdf import build_native_pdf_anchor
from academic_chatbot.documents.normalization import canonicalize_extracted_words
from academic_chatbot.embeddings.models import EmbeddingSpanIdentity
from academic_chatbot.retrieval.hybrid_models import (
    ChunkCandidateIdentity,
    ExactRationalScore,
    HybridChannelMembership,
    HybridLexicalContribution,
    HybridParentChunkContext,
    HybridRankingTrace,
    HybridRetrievalHit,
    HybridSemanticContribution,
)
from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.service import RetrievalHit

PROFILE = "ep-sha256-" + "a" * 64
VECTOR = "vector-generation-sha256-" + "b" * 64


def scope_data():
    return dict(
        project_id="project-1",
        paper_id="paper-1",
        file_version_id="file-1",
        document_generation_id="generation-1",
    )


def parent_data():
    return dict(
        project_id="project-1",
        document_generation_id="generation-1",
        page_id="page-1",
        chunk_id="chunk-1",
    )


def range_data():
    return dict(
        start_offset=0,
        end_offset=5,
        expected_text_sha256=sha256(b"alpha").hexdigest(),
        expected_anchor_ids=("page-anchor-sha256-" + "c" * 64,),
    )


def request_data():
    return dict(
        schema_version="evidence-bundle-request-v1",
        scope=scope_data(),
        concern_profile_id="stated-study-objective-v1",
        origin=dict(mode="lexical", ordering="caller_asserted_retrieval_order"),
        candidates=(
            dict(parent=parent_data(), reported_rank=1, lexical=range_data(), semantic=None),
        ),
        preview_budget=dict(
            profile_id="bundle-preview-bytes-v1", max_content_bytes=16384, max_entries=8
        ),
    )


def lexical_hit():
    canonical = canonicalize_extracted_words(
        (
            {"text": "alpha", "x0": 1.0, "top": 1.0, "x1": 10.0, "bottom": 5.0},
            {"text": "beta", "x0": 11.0, "top": 1.0, "x1": 20.0, "bottom": 5.0},
        ),
        page_width_points=100.0,
        page_height_points=100.0,
    )
    anchors = tuple(
        build_native_pdf_anchor(
            file_version_id="file-1",
            source_pdf_sha256="d" * 64,
            physical_page_index=0,
            page_width_points=100.0,
            page_height_points=100.0,
            source_page_rotation_degrees=0,
            canonical_page=canonical,
            word=word,
        )
        for word in canonical.words
    )
    return RetrievalHit(
        **scope_data(),
        page_id="page-1",
        chunk_id="chunk-1",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_ordinal=0,
        chunk_text="alpha beta",
        start_offset=0,
        end_offset=10,
        rank=2,
        raw_bm25_score=-1.5,
        anchors=anchors,
    )


def semantic_hit(*, equal=False):
    lexical = lexical_hit()
    start, end = (0, 10) if equal else (6, 10)
    span = EmbeddingSpanIdentity(
        **{key: value for key, value in parent_data().items() if key != "project_id"},
        start_offset=start,
        end_offset=end,
        embedding_profile_id=PROFILE,
    )
    return SemanticRetrievalHit(
        **scope_data(),
        page_id="page-1",
        chunk_id="chunk-1",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        embedding_span_id=span.embedding_span_id,
        embedding_profile_id=PROFILE,
        vector_generation_id=VECTOR,
        start_offset=start,
        end_offset=end,
        embedding_span_text=lexical.chunk_text[start:end],
        rank=3,
        raw_semantic_score=0.75,
        anchors=lexical.anchors if equal else (lexical.anchors[1],),
    )


def hybrid_hit(*, equal=False):
    lexical, semantic = lexical_hit(), semantic_hit(equal=equal)
    identity = ChunkCandidateIdentity(**parent_data())
    parent = HybridParentChunkContext(
        identity=identity,
        paper_id="paper-1",
        file_version_id="file-1",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_ordinal=0,
        start_offset=0,
        end_offset=10,
        chunk_text=lexical.chunk_text,
    )
    return HybridRetrievalHit(
        identity=identity,
        parent_chunk=parent,
        lexical_contribution=HybridLexicalContribution(lexical_hit=lexical),
        semantic_contribution=HybridSemanticContribution(semantic_hit=semantic),
        trace=HybridRankingTrace(
            fusion_profile_id="rrf-v1",
            fusion_score=ExactRationalScore(numerator=1, denominator=41),
            fusion_rank=1,
            channel_membership=HybridChannelMembership.BOTH,
            lexical_rank=2,
            semantic_rank=3,
        ),
    )


def payload_data():
    return dict(
        scope={**scope_data(), "source_pdf_sha256": "d" * 64},
        origin=dict(mode="lexical", ordering="caller_asserted_retrieval_order"),
        entries=(),
        context=(),
        dispositions=(),
        coverage=dict(
            status="no_flagged_native_gaps",
            pages=(),
            candidate_page_ids=(),
            packed_page_ids=(),
            flagged_page_ids=(),
            excluded_page_ids=(),
        ),
        budget=dict(
            profile_id="bundle-preview-bytes-v1",
            max_content_bytes=16384,
            used_content_bytes=77,
            max_entries=8,
            used_entries=0,
        ),
        state="insufficient_evidence",
        insufficient_reason="no_candidates",
        content_preview=dict(
            concern_profile_id="stated-study-objective-v1", evidence=(), context=()
        ),
    )


def resolved_range_data():
    hit = lexical_hit()
    anchors = []
    ids = []
    for anchor in hit.anchors:
        page, evidence = hit.page_id.encode(), anchor.evidence_id.encode("ascii")
        identifier = (
            "page-anchor-sha256-"
            + sha256(
                len(page).to_bytes(8, "big") + page + len(evidence).to_bytes(8, "big") + evidence
            ).hexdigest()
        )
        ids.append(identifier)
        anchors.append(
            dict(
                page_anchor_id=identifier,
                evidence_id=anchor.evidence_id,
                char_start=anchor.char_start,
                char_end=anchor.char_end,
                anchor_text=anchor.anchor_text,
                anchor_text_sha256=anchor.anchor_text_sha256,
                boxes_sha256=anchor.boxes_sha256,
                boxes=anchor.boxes,
            )
        )
    source = dict(
        scope={**scope_data(), "source_pdf_sha256": "d" * 64},
        parent=parent_data(),
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        canonical_page_text_sha256=hit.anchors[0].canonical_page_text_sha256,
        parser_profile_sha256=hit.anchors[0].parser_profile_sha256,
        processing_profile_id="chunk-profile-1",
        page_width_points=100.0,
        page_height_points=100.0,
        source_page_rotation_degrees=0,
        start_offset=0,
        end_offset=10,
        text_sha256=sha256(b"alpha beta").hexdigest(),
        anchor_ids=tuple(ids),
        provenance=(dict(channel="lexical"),),
    )
    return dict(
        source=source,
        text="alpha beta",
        anchors=tuple(anchors),
        contributions=(dict(channel="lexical"),),
    )
