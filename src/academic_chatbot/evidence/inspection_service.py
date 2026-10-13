"""Generation-free composition of scoped retrieval and the existing EvidenceBundle authority."""

from __future__ import annotations

from typing import Literal, Protocol, cast

from pydantic import TypeAdapter, ValidationError

from academic_chatbot.evidence.inspection_models import (
    EvidenceInspectionRequest,
    EvidenceInspectionResult,
    InspectionRetrievalIdentity,
    InspectionSuccess,
    inspection_failure,
)
from academic_chatbot.evidence.inspection_serialization import canonical_inspection_result_bytes
from academic_chatbot.evidence.models import (
    BundleSourceScope,
    CandidateOrigin,
    EvidenceBundle,
    EvidenceBundleRequest,
    EvidenceCandidateRef,
    PreviewBudget,
    PreviewState,
)
from academic_chatbot.evidence.references import (
    candidate_ref_from_hybrid,
    candidate_ref_from_lexical,
    candidate_ref_from_semantic,
)
from academic_chatbot.evidence.serialization import canonical_bundle_bytes
from academic_chatbot.retrieval.hybrid_models import HybridRetrievalHit
from academic_chatbot.retrieval.scoped_models import (
    ScopedRetrievalError,
    ScopedRetrievalRequest,
    ScopedRetrievalResult,
)
from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.service import RetrievalHit

LiteralSuccess = Literal["ready", "partial", "insufficient"]


class _ScopedRetrieval(Protocol):
    def search_scoped(self, request: ScopedRetrievalRequest) -> ScopedRetrievalResult: ...
    def recheck_current(self, request: ScopedRetrievalRequest) -> None: ...


class _Bundles(Protocol):
    def build(self, request: EvidenceBundleRequest) -> EvidenceBundle: ...


def _candidate(
    hit: RetrievalHit | SemanticRetrievalHit | HybridRetrievalHit,
) -> EvidenceCandidateRef:
    if isinstance(hit, RetrievalHit):
        return candidate_ref_from_lexical(hit)
    if isinstance(hit, SemanticRetrievalHit):
        return candidate_ref_from_semantic(hit)
    if isinstance(hit, HybridRetrievalHit):
        return candidate_ref_from_hybrid(hit)
    raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")


def _validate_bundle(
    value: EvidenceBundle, request: EvidenceBundleRequest
) -> tuple[EvidenceBundle, bytes]:
    bundle = EvidenceBundle.model_validate(value)
    raw = canonical_bundle_bytes(bundle)
    scope = BundleSourceScope(
        **{name: getattr(bundle.scope, name) for name in BundleSourceScope.model_fields}
    )
    budget = PreviewBudget(
        **{name: getattr(bundle.budget, name) for name in PreviewBudget.model_fields}
    )
    if (
        scope != request.scope
        or bundle.origin != request.origin
        or budget != request.preview_budget
        or tuple(d.reference for d in bundle.dispositions) != request.candidates
        or tuple(d.input_position for d in bundle.dispositions)
        != tuple(range(len(request.candidates)))
    ):
        raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")
    return bundle, raw


class EvidenceInspectionService:
    def __init__(self, *, retrieval_service: _ScopedRetrieval, bundle_service: _Bundles) -> None:
        self._retrieval = retrieval_service
        self._bundles = bundle_service

    def inspect(self, request: EvidenceInspectionRequest) -> EvidenceInspectionResult:
        try:
            request = EvidenceInspectionRequest.model_validate(request)
            fingerprint = request.fingerprint
        except (ValidationError, ValueError, TypeError) as error:
            code = (
                error
                if isinstance(error, ScopedRetrievalError)
                else ScopedRetrievalError("INVALID_REQUEST")
            )
            return inspection_failure(code)
        except Exception as error:
            return inspection_failure(error)
        admitted_fingerprint = None
        try:
            scoped = request.retrieval_request
            result: ScopedRetrievalResult = TypeAdapter(ScopedRetrievalResult).validate_python(
                self._retrieval.search_scoped(scoped)
            )
            if (
                result.request_fingerprint != scoped.fingerprint
                or result.source != scoped.scope
                or result.query != scoped.query
                or result.mode != scoped.mode
                or result.limit != scoped.limit
                or result.policy != scoped.policy
                or result.semantic_identity != scoped.semantic_identity
            ):
                raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")
            admitted_fingerprint = fingerprint
            semantic = scoped.semantic_identity
            bundle_request = EvidenceBundleRequest(
                scope=scoped.scope,
                origin=CandidateOrigin(
                    mode=scoped.mode,
                    fusion_profile_id=scoped.policy.fusion_profile_id,
                    embedding_profile_id=semantic.embedding_profile_id if semantic else None,
                    vector_generation_id=semantic.vector_generation_id if semantic else None,
                ),
                candidates=tuple(_candidate(hit) for hit in result.hits),
                preview_budget=request.preview_budget,
            )
            # Reuse scoped authority to guard the DB family before each resolver opens it.
            self._retrieval.recheck_current(scoped)
            first, first_bytes = _validate_bundle(
                self._bundles.build(bundle_request), bundle_request
            )
            self._retrieval.recheck_current(scoped)
            second, second_bytes = _validate_bundle(
                self._bundles.build(bundle_request), bundle_request
            )
            if first.fingerprint != second.fingerprint or first_bytes != second_bytes:
                raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")
            self._retrieval.recheck_current(scoped)
            outcome = {
                PreviewState.PREVIEW_READY: "ready",
                PreviewState.PREVIEW_PARTIAL: "partial",
                PreviewState.INSUFFICIENT_EVIDENCE: "insufficient",
            }[second.state]
            envelope = InspectionSuccess(
                outcome=cast("LiteralSuccess", outcome),
                request_fingerprint=fingerprint,
                query=scoped.query,
                source=second.scope,
                retrieval_identity=InspectionRetrievalIdentity(
                    retrieval_request_fingerprint=result.request_fingerprint,
                    mode=result.mode,
                    limit=result.limit,
                    policy=result.policy,
                    semantic_identity=result.semantic_identity,
                    channels=result.channels,
                ),
                bundle_fingerprint=second.fingerprint,
                bundle=second,
            )
            canonical_inspection_result_bytes(envelope)
            self._retrieval.recheck_current(scoped)
            return envelope
        except ValidationError:
            return inspection_failure(
                ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE"),
                request_fingerprint=admitted_fingerprint,
            )
        except Exception as error:
            return inspection_failure(error, request_fingerprint=admitted_fingerprint)
