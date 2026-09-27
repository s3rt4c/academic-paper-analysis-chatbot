"""One authoritative resolution followed by pure, generation-disabled preparation."""

from __future__ import annotations

from pydantic import ValidationError

from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.packing import (
    derive_bundle_state,
    normalize_candidate_groups,
    pack_candidate_groups,
)
from academic_chatbot.evidence.resolver import EvidenceReadResolver
from academic_chatbot.evidence.serialization import (
    EvidencePreparationError,
    canonical_bundle_bytes,
    seal_bundle,
)


class EvidenceBundleService:
    def __init__(self, *, resolver: EvidenceReadResolver) -> None:
        self._resolver = resolver

    def build(self, request: m.EvidenceBundleRequest) -> m.EvidenceBundle:
        try:
            request = m.EvidenceBundleRequest.model_validate(request)
        except (ValidationError, TypeError, ValueError):
            raise EvidencePreparationError(m.EvidenceErrorCode.INVALID_REQUEST) from None
        resolved = self._resolver.resolve(request)
        packed = pack_candidate_groups(
            normalize_candidate_groups(resolved), resolved.request.preview_budget
        )
        coverage = resolved.coverage.model_copy(
            update={
                "packed_page_ids": tuple(
                    dict.fromkeys(e.source.parent.page_id for e in packed.entries)
                )
            }
        )
        omitted = sum(
            d.omission is not None and d.omission.reason != m.OmissionReason.DUPLICATE_CANDIDATE
            for d in packed.dispositions
        )
        state, reason = derive_bundle_state(
            len(packed.entries), omitted, coverage, candidate_count=len(resolved.groups)
        )
        payload = m.EvidenceBundlePayload(
            scope=resolved.resolved_scope,
            origin=resolved.request.origin,
            entries=packed.entries,
            context=packed.context,
            coverage=coverage,
            dispositions=packed.dispositions,
            budget=m.EvidenceBudgetUsage(
                **resolved.request.preview_budget.model_dump(),
                used_content_bytes=packed.used_content_bytes,
                used_entries=len(packed.entries),
            ),
            state=state,
            insufficient_reason=reason,
            content_preview=packed.content_preview,
        )
        bundle = seal_bundle(payload)
        canonical_bundle_bytes(bundle)
        return bundle
