"""Strict source-scoped acquisition values; document text remains untrusted."""

from __future__ import annotations

import hashlib
import math
from enum import StrEnum
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

from academic_chatbot.embeddings.models import canonical_json_bytes
from academic_chatbot.evidence.models import (
    MAX_CHUNK_ANCHORS,
    MAX_PAGE_TEXT_BYTES,
    MAX_RESOLVED_TEXT_BYTES,
    BundleSourceScope,
    EvidenceValue,
    Identifier,
    Sha256,
    SourceText,
)
from academic_chatbot.retrieval.hybrid_fusion import (
    RRF_V1_K,
    RRF_V1_LEXICAL_WEIGHT,
    RRF_V1_SEMANTIC_WEIGHT,
)
from academic_chatbot.retrieval.hybrid_models import HybridRetrievalHit
from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.service import RetrievalHit

Mode = Literal["lexical", "semantic", "hybrid"]
Limit = Annotated[int, Field(strict=True, ge=1, le=100)]


class ScopedRetrievalErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    SOURCE_NOT_CURRENT = "SOURCE_NOT_CURRENT"
    SOURCE_SCOPE_AMBIGUOUS = "SOURCE_SCOPE_AMBIGUOUS"
    UNSAFE_PATH = "UNSAFE_PATH"
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
    STORAGE_INTEGRITY = "STORAGE_INTEGRITY"
    VECTOR_UNAVAILABLE = "VECTOR_UNAVAILABLE"
    VECTOR_NOT_CURRENT = "VECTOR_NOT_CURRENT"
    VECTOR_INTEGRITY = "VECTOR_INTEGRITY"
    PROFILE_MISMATCH = "PROFILE_MISMATCH"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INTERNAL_INVARIANT_FAILURE = "INTERNAL_INVARIANT_FAILURE"


_MESSAGES = {
    ScopedRetrievalErrorCode.INVALID_REQUEST: "Scoped search request is invalid.",
    ScopedRetrievalErrorCode.SOURCE_NOT_FOUND: "Selected source was not found.",
    ScopedRetrievalErrorCode.SOURCE_NOT_CURRENT: "Selected source is not current.",
    ScopedRetrievalErrorCode.SOURCE_SCOPE_AMBIGUOUS: "Selected source is ambiguous.",
    ScopedRetrievalErrorCode.UNSAFE_PATH: "Storage path is unsafe.",
    ScopedRetrievalErrorCode.STORAGE_UNAVAILABLE: "Source storage is unavailable.",
    ScopedRetrievalErrorCode.STORAGE_INTEGRITY: "Source storage is invalid.",
    ScopedRetrievalErrorCode.VECTOR_UNAVAILABLE: "Semantic index is unavailable.",
    ScopedRetrievalErrorCode.VECTOR_NOT_CURRENT: "Semantic index is not current.",
    ScopedRetrievalErrorCode.VECTOR_INTEGRITY: "Semantic index is invalid.",
    ScopedRetrievalErrorCode.PROFILE_MISMATCH: "Embedding profile does not match.",
    ScopedRetrievalErrorCode.RESOURCE_LIMIT: "Scoped search resource limit exceeded.",
    ScopedRetrievalErrorCode.INTERNAL_INVARIANT_FAILURE: "Scoped search invariant failed.",
}


class ScopedRetrievalError(ValueError):
    def __init__(self, code: ScopedRetrievalErrorCode | str) -> None:
        self.code = ScopedRetrievalErrorCode(code)
        self.message = _MESSAGES[self.code]
        super().__init__(self.message)


class SemanticIdentity(EvidenceValue):
    embedding_profile_id: Identifier
    embedding_profile_sha256: Sha256
    vector_generation_id: Identifier
    source_snapshot_sha256: Sha256
    vector_store_manifest_sha256: Sha256


class ScopedRetrievalPolicy(EvidenceValue):
    profile_id: Literal["source-scoped-baseline-v1"] = "source-scoped-baseline-v1"
    lexical_policy_id: Literal["lexical-natural-language-v1"] | None = None
    semantic_query_policy_id: None = None
    fusion_profile_id: Literal["rrf-v1"] | None = None


def policy_for(mode: Mode) -> ScopedRetrievalPolicy:
    return ScopedRetrievalPolicy(
        lexical_policy_id="lexical-natural-language-v1" if mode != "semantic" else None,
        fusion_profile_id="rrf-v1" if mode == "hybrid" else None,
    )


def _query(value: str) -> str:
    if not value.strip() or len(value.encode("utf-8", errors="strict")) > 4096:
        raise ValueError("invalid scoped query")
    return value


def _policy(mode: Mode, policy: ScopedRetrievalPolicy, identity: SemanticIdentity | None) -> None:
    if policy != policy_for(mode) or (identity is None) != (mode == "lexical"):
        raise ValueError("contradictory scoped policy")


class ScopedRetrievalRequest(EvidenceValue):
    schema_version: Literal["source-scoped-retrieval-request-v1"] = (
        "source-scoped-retrieval-request-v1"
    )
    scope: BundleSourceScope
    query: SourceText
    mode: Mode
    limit: Limit = 10
    policy: ScopedRetrievalPolicy
    semantic_identity: SemanticIdentity | None

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        return _query(value)

    @model_validator(mode="after")
    def _validate_policy(self) -> Self:
        _policy(self.mode, self.policy, self.semantic_identity)
        return self

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


class ScopedChannel(EvidenceValue):
    state: Literal["not_run", "empty", "nonempty"]
    requested_limit: Limit | None
    returned_count: Annotated[int, Field(strict=True, ge=0, le=100)]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.state == "not_run":
            if self.requested_limit is not None or self.returned_count != 0:
                raise ValueError("unused channel has results")
        elif (
            self.requested_limit is None
            or self.returned_count > self.requested_limit
            or (self.state == "empty") != (self.returned_count == 0)
        ):
            raise ValueError("contradictory channel state")
        return self


class ScopedChannels(EvidenceValue):
    lexical: ScopedChannel
    semantic: ScopedChannel


_identifier_adapter = TypeAdapter(Identifier)
_digest_adapter = TypeAdapter(Sha256)


def _nested(value: object, key: str = "") -> int:
    """Validate raw nested legacy values before their permissive parsers coerce them."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    if isinstance(value, dict):
        return sum(_nested(item, name) for name, item in value.items())
    if isinstance(value, (tuple, list)):
        if key == "anchors" and not 1 <= len(value) <= MAX_CHUNK_ANCHORS:
            raise ValueError("anchor count exceeds bound")
        return sum(_nested(item, key) for item in value)
    if key.endswith("_id"):
        _identifier_adapter.validate_python(value)
    elif key.endswith("_sha256"):
        _digest_adapter.validate_python(value)
    elif (
        key
        in {
            "rank",
            "fusion_rank",
            "lexical_rank",
            "semantic_rank",
            "start_offset",
            "end_offset",
            "char_start",
            "char_end",
            "physical_page_index",
            "display_page_number",
            "chunk_ordinal",
            "numerator",
            "denominator",
            "source_page_rotation_degrees",
        }
        and value is not None
    ):
        if type(value) is not int or value < 0:
            raise ValueError("invalid nested integer")
    if key in {
        "raw_bm25_score",
        "raw_semantic_score",
        "x0",
        "top",
        "x1",
        "bottom",
        "page_width_points",
        "page_height_points",
    }:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError("invalid nested numeric value")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite nested score")
    if isinstance(value, str):
        length = len(value.encode("utf-8", errors="strict"))
        if key == "canonical_page_text" and length > MAX_PAGE_TEXT_BYTES:
            raise ValueError("page text exceeds bound")
        if key in {"canonical_page_text", "chunk_text", "embedding_span_text", "anchor_text"}:
            return length
    return 0


def validate_hit_scope(
    hit: RetrievalHit | SemanticRetrievalHit | HybridRetrievalHit,
    scope: BundleSourceScope,
    identity: SemanticIdentity | None,
) -> None:
    packets: list[RetrievalHit | SemanticRetrievalHit]
    if isinstance(hit, HybridRetrievalHit):
        parent = hit.parent_chunk
        if (
            parent.identity.project_id != scope.project_id
            or parent.identity.document_generation_id != scope.document_generation_id
            or parent.paper_id != scope.paper_id
            or parent.file_version_id != scope.file_version_id
        ):
            raise ValueError("foreign parent")
        packets = []
        if hit.lexical_contribution:
            packets.append(hit.lexical_contribution.lexical_hit)
        if hit.semantic_contribution:
            packets.append(hit.semantic_contribution.semantic_hit)
    else:
        packets = [hit]
    for packet in packets:
        if any(getattr(packet, name) != getattr(scope, name) for name in type(scope).model_fields):
            raise ValueError("foreign evidence")
        if not 0 <= packet.start_offset < packet.end_offset:
            raise ValueError("invalid evidence range")
        if packet.display_page_number != packet.physical_page_index + 1:
            raise ValueError("invalid page number")
        if isinstance(packet, SemanticRetrievalHit):
            if (
                identity is None
                or packet.embedding_profile_id != identity.embedding_profile_id
                or packet.vector_generation_id != identity.vector_generation_id
            ):
                raise ValueError("foreign vector identity")
            text = packet.embedding_span_text
        else:
            text = packet.chunk_text
        for anchor in packet.anchors:
            if (
                anchor.file_version_id != scope.file_version_id
                or anchor.physical_page_index != packet.physical_page_index
                or anchor.display_page_number != packet.display_page_number
                or anchor.printed_page_label != packet.printed_page_label
                or not packet.start_offset
                <= anchor.char_start
                < anchor.char_end
                <= packet.end_offset
                or anchor.canonical_page_text[packet.start_offset : packet.end_offset] != text
            ):
                raise ValueError("foreign anchor")


class _ScopedResult(EvidenceValue):
    schema_version: Literal["source-scoped-retrieval-result-v1"] = (
        "source-scoped-retrieval-result-v1"
    )
    mode: Mode
    request_fingerprint: Sha256
    source: BundleSourceScope
    query: SourceText
    limit: Limit
    policy: ScopedRetrievalPolicy
    semantic_identity: SemanticIdentity | None
    channels: ScopedChannels
    hits: tuple[RetrievalHit | SemanticRetrievalHit | HybridRetrievalHit, ...]
    retrieval_completeness: Literal["not_assessed"] = "not_assessed"

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        return _query(value)

    @field_validator("hits", mode="before", check_fields=False)
    @classmethod
    def _raw_hits(cls, value: object) -> object:
        if _nested(value) > MAX_RESOLVED_TEXT_BYTES:
            raise ValueError("cumulative text exceeds bound")
        if isinstance(value, tuple):
            return tuple(
                item.model_dump(mode="python") if isinstance(item, BaseModel) else item
                for item in value
            )
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        _policy(self.mode, self.policy, self.semantic_identity)
        hits = self.hits
        if len(hits) > self.limit:
            raise ValueError("too many hits")
        expected_depth = min(100, max(50, self.limit * 5)) if self.mode == "hybrid" else self.limit
        for name in ("lexical", "semantic"):
            channel = getattr(self.channels, name)
            runs = self.mode == name or self.mode == "hybrid"
            if runs != (channel.state != "not_run"):
                raise ValueError("wrong channel ran")
            if runs and channel.requested_limit != expected_depth:
                raise ValueError("wrong channel cap")
            if runs and self.mode != "hybrid" and channel.returned_count != len(hits):
                raise ValueError("wrong channel count")
        if self.mode == "hybrid" and bool(hits) != bool(
            self.channels.lexical.returned_count + self.channels.semantic.returned_count
        ):
            raise ValueError("hybrid evidence contradicts channel counts")
        request = ScopedRetrievalRequest(
            scope=self.source,
            query=self.query,
            mode=self.mode,
            limit=self.limit,
            policy=self.policy,
            semantic_identity=self.semantic_identity,
        )
        if request.fingerprint != self.request_fingerprint:
            raise ValueError("wrong request binding")
        for rank, hit in enumerate(hits, 1):
            actual = hit.trace.fusion_rank if isinstance(hit, HybridRetrievalHit) else hit.rank
            if actual != rank:
                raise ValueError("non-contiguous ranks")
            validate_hit_scope(hit, self.source, self.semantic_identity)
            if isinstance(hit, HybridRetrievalHit):
                if hit.trace.fusion_profile_id != self.policy.fusion_profile_id:
                    raise ValueError("wrong fusion profile")
                score = Fraction(0, 1)
                for channel, contribution, weight in (
                    (self.channels.lexical, hit.lexical_contribution, RRF_V1_LEXICAL_WEIGHT),
                    (self.channels.semantic, hit.semantic_contribution, RRF_V1_SEMANTIC_WEIGHT),
                ):
                    if contribution is not None:
                        packet = (
                            contribution.lexical_hit
                            if hasattr(contribution, "lexical_hit")
                            else contribution.semantic_hit
                        )
                        if channel.state != "nonempty" or packet.rank > channel.returned_count:
                            raise ValueError("contribution exceeds channel count")
                        score += Fraction(weight, RRF_V1_K + packet.rank)
                if (hit.trace.fusion_score.numerator, hit.trace.fusion_score.denominator) != (
                    score.numerator,
                    score.denominator,
                ):
                    raise ValueError("wrong exact fusion vote")
        return self


class ScopedLexicalResult(_ScopedResult):
    mode: Literal["lexical"] = "lexical"
    hits: tuple[RetrievalHit, ...]


class ScopedSemanticResult(_ScopedResult):
    mode: Literal["semantic"] = "semantic"
    hits: tuple[SemanticRetrievalHit, ...]


class ScopedHybridResult(_ScopedResult):
    mode: Literal["hybrid"] = "hybrid"
    hits: tuple[HybridRetrievalHit, ...]


ScopedRetrievalResult = Annotated[
    ScopedLexicalResult | ScopedSemanticResult | ScopedHybridResult, Field(discriminator="mode")
]
