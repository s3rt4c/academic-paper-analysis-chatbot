"""Pure conversion of retrieval packets into text-free caller assertions."""

from hashlib import sha256

from academic_chatbot.evidence.models import EvidenceCandidateRef, LexicalRangeRef, SemanticRangeRef
from academic_chatbot.library.repository import _page_anchor_id_for
from academic_chatbot.retrieval.hybrid_models import ChunkCandidateIdentity, HybridRetrievalHit
from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.service import RetrievalHit


def _parent(hit: RetrievalHit | SemanticRetrievalHit) -> ChunkCandidateIdentity:
    return ChunkCandidateIdentity(
        project_id=hit.project_id,
        document_generation_id=hit.document_generation_id,
        page_id=hit.page_id,
        chunk_id=hit.chunk_id,
    )


def _anchor_ids(hit: RetrievalHit | SemanticRetrievalHit) -> tuple[str, ...]:
    return tuple(
        _page_anchor_id_for(page_id=hit.page_id, evidence_id=anchor.evidence_id)
        for anchor in hit.anchors
    )


def candidate_ref_from_lexical(hit: RetrievalHit) -> EvidenceCandidateRef:
    """Preserve the whole lexical chunk; no scope selection or storage access."""
    return EvidenceCandidateRef(
        parent=_parent(hit),
        reported_rank=hit.rank,
        lexical=LexicalRangeRef(
            start_offset=hit.start_offset,
            end_offset=hit.end_offset,
            expected_text_sha256=sha256(hit.chunk_text.encode("utf-8")).hexdigest(),
            expected_anchor_ids=_anchor_ids(hit),
        ),
    )


def candidate_ref_from_semantic(hit: SemanticRetrievalHit) -> EvidenceCandidateRef:
    """Preserve the semantic span, never its non-citable parent context."""
    return EvidenceCandidateRef(
        parent=_parent(hit),
        reported_rank=hit.rank,
        semantic=SemanticRangeRef(
            start_offset=hit.start_offset,
            end_offset=hit.end_offset,
            expected_text_sha256=sha256(hit.embedding_span_text.encode("utf-8")).hexdigest(),
            expected_anchor_ids=_anchor_ids(hit),
            embedding_span_id=hit.embedding_span_id,
        ),
    )


def candidate_ref_from_hybrid(hit: HybridRetrievalHit) -> EvidenceCandidateRef:
    """Keep both channel assertions, including equal ranges; use caller fusion rank."""
    lexical = (
        candidate_ref_from_lexical(hit.lexical_contribution.lexical_hit).lexical
        if hit.lexical_contribution is not None
        else None
    )
    semantic = (
        candidate_ref_from_semantic(hit.semantic_contribution.semantic_hit).semantic
        if hit.semantic_contribution is not None
        else None
    )
    return EvidenceCandidateRef(
        parent=hit.identity, reported_rank=hit.trace.fusion_rank, lexical=lexical, semantic=semantic
    )
