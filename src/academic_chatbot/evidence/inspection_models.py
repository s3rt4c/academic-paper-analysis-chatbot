"""Strict inspection envelopes wrapping the existing, single evidence authority."""

from __future__ import annotations

from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from academic_chatbot.evidence.models import (
    BundleSourceScope,
    CandidateOrigin,
    EvidenceBundle,
    EvidenceBundleError,
    EvidenceBundleRequest,
    EvidenceErrorCode,
    EvidenceValue,
    PreviewBudget,
    PreviewState,
    ResolvedBundleSource,
    Sha256,
    SourceText,
)
from academic_chatbot.evidence.resolver import EvidenceResolutionError
from academic_chatbot.evidence.serialization import EvidencePreparationError, canonical_bundle_bytes
from academic_chatbot.retrieval.scoped_models import (
    Limit,
    Mode,
    ScopedChannels,
    ScopedRetrievalError,
    ScopedRetrievalErrorCode,
    ScopedRetrievalPolicy,
    ScopedRetrievalRequest,
    SemanticIdentity,
    policy_for,
)


class EvidenceInspectionRequest(EvidenceValue):
    schema_version: Literal["evidence-inspection-request-v1"] = "evidence-inspection-request-v1"
    retrieval_request: ScopedRetrievalRequest
    preview_budget: PreviewBudget = Field(default_factory=PreviewBudget)

    @property
    def fingerprint(self) -> str:
        from academic_chatbot.evidence.inspection_serialization import (
            canonical_inspection_request_bytes,
        )

        return sha256(canonical_inspection_request_bytes(self)).hexdigest()


class InspectionRetrievalIdentity(EvidenceValue):
    retrieval_request_fingerprint: Sha256
    mode: Mode
    limit: Limit
    policy: ScopedRetrievalPolicy
    semantic_identity: SemanticIdentity | None
    channels: ScopedChannels

    @model_validator(mode="after")
    def _channels(self) -> Self:
        if self.policy != policy_for(self.mode) or (self.semantic_identity is None) != (
            self.mode == "lexical"
        ):
            raise ValueError("contradictory retrieval policy")
        depth = min(100, max(50, self.limit * 5)) if self.mode == "hybrid" else self.limit
        for name in ("lexical", "semantic"):
            channel = getattr(self.channels, name)
            runs = self.mode in (name, "hybrid")
            if runs != (channel.state != "not_run") or (runs and channel.requested_limit != depth):
                raise ValueError("contradictory retrieval channel")
        return self


_REJECTED = frozenset(
    {
        "INVALID_REQUEST",
        "UNSAFE_PATH",
        "SOURCE_NOT_FOUND",
        "SOURCE_SCOPE_MISMATCH",
        "SOURCE_NOT_CURRENT",
        "SOURCE_SCOPE_AMBIGUOUS",
        "GENERATION_NOT_CURRENT",
        "PROFILE_MISMATCH",
    }
)
_MESSAGES: dict[str, set[str]] = {}
for _code in ScopedRetrievalErrorCode:
    _MESSAGES.setdefault(_code.value, set()).add(ScopedRetrievalError(_code).message)
for _evidence_code in EvidenceErrorCode:
    _MESSAGES.setdefault(_evidence_code.value, set()).add(
        EvidenceBundleError(code=_evidence_code).message
    )


class InspectionError(EvidenceValue):
    code: SourceText
    message: SourceText

    @model_validator(mode="after")
    def _safe(self) -> Self:
        if self.message not in _MESSAGES.get(self.code, set()):
            raise ValueError("unknown or unsafe inspection error")
        return self


class _Envelope(EvidenceValue):
    schema_version: Literal["evidence-inspection-result-v1"] = "evidence-inspection-result-v1"
    inspection_profile_id: Literal["source-scoped-inspection-v1"] = "source-scoped-inspection-v1"
    request_fingerprint: Sha256 | None


class InspectionFailure(_Envelope):
    outcome: Literal["rejected", "failed"]
    error: InspectionError

    @model_validator(mode="after")
    def _outcome(self) -> Self:
        if (self.outcome == "rejected") != (self.error.code in _REJECTED):
            raise ValueError("wrong failure outcome")
        return self


class InspectionSuccess(_Envelope):
    outcome: Literal["ready", "partial", "insufficient"]
    request_fingerprint: Sha256
    query: SourceText
    source: ResolvedBundleSource
    retrieval_identity: InspectionRetrievalIdentity
    bundle_fingerprint: Sha256
    bundle: EvidenceBundle

    @model_validator(mode="after")
    def _binding(self) -> Self:
        states = {
            "ready": PreviewState.PREVIEW_READY,
            "partial": PreviewState.PREVIEW_PARTIAL,
            "insufficient": PreviewState.INSUFFICIENT_EVIDENCE,
        }
        if self.bundle.state != states[self.outcome] or self.bundle.scope != self.source:
            raise ValueError("bundle outcome/source mismatch")
        if self.bundle_fingerprint != self.bundle.fingerprint:
            raise ValueError("bundle fingerprint mismatch")
        canonical_bundle_bytes(self.bundle)
        identity = self.retrieval_identity
        candidates = tuple(d.reference for d in self.bundle.dispositions)
        count = len(candidates)
        if count > identity.limit:
            raise ValueError("bundle exceeds retrieval cap")
        if identity.mode != "hybrid":
            if getattr(identity.channels, identity.mode).returned_count != count:
                raise ValueError("bundle contradicts retrieval count")
        else:
            total = sum(
                getattr(identity.channels, name).returned_count for name in ("lexical", "semantic")
            )
            if bool(count) != bool(total) or count > total:
                raise ValueError("bundle contradicts hybrid channel counts")
            for name in ("lexical", "semantic"):
                retained = sum(getattr(candidate, name) is not None for candidate in candidates)
                if retained > getattr(identity.channels, name).returned_count:
                    raise ValueError("bundle contribution contradicts channel count")
        scope = BundleSourceScope(
            **{name: getattr(self.source, name) for name in BundleSourceScope.model_fields}
        )
        retrieval = ScopedRetrievalRequest(
            scope=scope,
            query=self.query,
            mode=identity.mode,
            limit=identity.limit,
            policy=identity.policy,
            semantic_identity=identity.semantic_identity,
        )
        if retrieval.fingerprint != identity.retrieval_request_fingerprint:
            raise ValueError("retrieval fingerprint mismatch")
        semantic = identity.semantic_identity
        origin = CandidateOrigin(
            mode=identity.mode,
            fusion_profile_id=identity.policy.fusion_profile_id,
            embedding_profile_id=semantic.embedding_profile_id if semantic else None,
            vector_generation_id=semantic.vector_generation_id if semantic else None,
        )
        if self.bundle.origin != origin:
            raise ValueError("bundle origin mismatch")
        budget = PreviewBudget(
            **{name: getattr(self.bundle.budget, name) for name in PreviewBudget.model_fields}
        )
        EvidenceBundleRequest(
            scope=scope, origin=origin, candidates=candidates, preview_budget=budget
        )
        if (
            EvidenceInspectionRequest(
                retrieval_request=retrieval, preview_budget=budget
            ).fingerprint
            != self.request_fingerprint
        ):
            raise ValueError("inspection request fingerprint mismatch")
        return self


EvidenceInspectionResult = Annotated[
    InspectionSuccess | InspectionFailure, Field(discriminator="outcome")
]


def inspection_failure(
    error: Exception, *, request_fingerprint: str | None = None
) -> InspectionFailure:
    """Retain existing safe evidence messages; discard internal details and input positions."""
    if isinstance(error, (EvidenceResolutionError, EvidencePreparationError)):
        try:
            safe = EvidenceBundleError(code=EvidenceErrorCode(error.error.code))
            code, message = safe.code.value, safe.message
        except (ValueError, TypeError):
            fallback = ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")
            code, message = fallback.code.value, fallback.message
    else:
        from academic_chatbot.retrieval.scoped_service import safe_error

        scoped = safe_error(error)
        code, message = scoped.code.value, scoped.message
    return InspectionFailure(
        outcome="rejected" if code in _REJECTED else "failed",
        request_fingerprint=request_fingerprint,
        error=InspectionError(code=code, message=message),
    )
