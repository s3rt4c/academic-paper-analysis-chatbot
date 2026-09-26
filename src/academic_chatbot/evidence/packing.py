"""Pure occurrence normalization and atomic, caller-ordered preview packing."""

from __future__ import annotations

from dataclasses import dataclass

from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.budget import measure_proposed_preview, render_content_preview
from academic_chatbot.evidence.serialization import EvidencePreparationError


@dataclass(frozen=True)
class NormalizedCandidates:
    resolved: m.ResolvedEvidenceInput
    groups: tuple[m.ResolvedCandidateGroup, ...]
    duplicates: tuple[m.CandidateDisposition, ...]


@dataclass(frozen=True)
class PackingResult:
    entries: tuple[m.EvidenceBundleEntry, ...]
    context: tuple[m.EvidenceContext, ...]
    dispositions: tuple[m.CandidateDisposition, ...]
    content_preview: m.ContentPreview
    used_content_bytes: int


def _reference_key(reference: m.EvidenceCandidateRef) -> tuple[object, ...]:
    parent = reference.parent
    return (
        parent.project_id,
        parent.document_generation_id,
        parent.page_id,
        parent.chunk_id,
        reference.lexical is not None,
        reference.semantic.embedding_span_id if reference.semantic else None,
    )


def normalize_candidate_groups(resolved: m.ResolvedEvidenceInput) -> NormalizedCandidates:
    seen: dict[tuple[object, ...], m.ResolvedCandidateGroup] = {}
    groups = []
    duplicates = []
    previous_rank = 0
    for group in resolved.groups:
        key = _reference_key(group.reference)
        first = seen.get(key)
        if first is not None:
            if (
                group.reference != first.reference
                or group.ranges != first.ranges
                or group.context != first.context
            ):
                raise EvidencePreparationError(
                    m.EvidenceErrorCode.CONTRADICTORY_DUPLICATE, group.input_position
                )
            duplicates.append(
                m.CandidateDisposition(
                    input_position=group.input_position,
                    reference=group.reference,
                    omission=m.EvidenceOmission(
                        reason=m.OmissionReason.DUPLICATE_CANDIDATE,
                        duplicate_of_input_position=first.input_position,
                    ),
                )
            )
            continue
        seen[key] = group
        if group.reference.reported_rank <= previous_rank:
            raise EvidencePreparationError(m.EvidenceErrorCode.INVALID_ORDER, group.input_position)
        previous_rank = group.reference.reported_rank
        groups.append(group)
    return NormalizedCandidates(resolved, tuple(groups), tuple(duplicates))


def _coalesced_ranges(group: m.ResolvedCandidateGroup) -> tuple[m.ResolvedEvidenceRange, ...]:
    if len(group.ranges) != 2:
        return group.ranges
    first, second = group.ranges
    left, right = first.source, second.source
    if (left.parent, left.start_offset, left.end_offset) != (
        right.parent,
        right.start_offset,
        right.end_offset,
    ):
        return group.ranges
    if (
        first.text != second.text
        or first.anchors != second.anchors
        or left.model_dump(exclude={"provenance"}) != right.model_dump(exclude={"provenance"})
    ):
        raise EvidencePreparationError(
            m.EvidenceErrorCode.CONTRADICTORY_DUPLICATE, group.input_position
        )
    provenance = (*first.contributions, *second.contributions)
    return (
        m.ResolvedEvidenceRange(
            source=left.model_copy(update={"provenance": provenance}),
            text=first.text,
            anchors=first.anchors,
            contributions=provenance,
        ),
    )


def pack_candidate_groups(
    normalized: NormalizedCandidates, budget: m.PreviewBudget
) -> PackingResult:
    entries: tuple[m.EvidenceBundleEntry, ...] = ()
    contexts: tuple[m.EvidenceContext, ...] = ()
    dispositions = {d.input_position: d for d in normalized.duplicates}
    used = measure_proposed_preview(entries, contexts)
    for group in normalized.groups:
        ranges = _coalesced_ranges(group)
        additions = tuple(
            m.EvidenceBundleEntry(**item.model_dump(), citation_label=f"E{len(entries) + i + 1}")
            for i, item in enumerate(ranges)
        )
        labels = tuple(e.citation_label for e in additions)
        new_context = (
            ()
            if group.context is None
            else (m.EvidenceContext(**group.context.model_dump(), related_citation_labels=labels),)
        )
        proposed_entries = (*entries, *additions)
        proposed_contexts = (*contexts, *new_context)
        size = measure_proposed_preview(proposed_entries, proposed_contexts)
        bytes_failed = size > budget.max_content_bytes
        entries_failed = len(proposed_entries) > budget.max_entries
        if bytes_failed or entries_failed:
            violations: list[m.OmissionReason] = []
            if bytes_failed:
                violations.append(m.OmissionReason.PREVIEW_BYTES_EXCEEDED)
            if entries_failed:
                violations.append(m.OmissionReason.ENTRY_LIMIT_EXCEEDED)
            omission = m.EvidenceOmission.model_validate(
                {
                    "reason": m.OmissionReason.ENTRY_LIMIT_EXCEEDED
                    if entries_failed
                    else m.OmissionReason.PREVIEW_BYTES_EXCEEDED,
                    "violated_constraints": tuple(reason.value for reason in violations),
                }
            )
            dispositions[group.input_position] = m.CandidateDisposition(
                input_position=group.input_position, reference=group.reference, omission=omission
            )
            continue
        entries, contexts, used = proposed_entries, proposed_contexts, size
        dispositions[group.input_position] = m.CandidateDisposition(
            input_position=group.input_position, reference=group.reference, citation_labels=labels
        )
    return PackingResult(
        entries,
        contexts,
        tuple(dispositions[g.input_position] for g in normalized.resolved.groups),
        render_content_preview(entries, contexts),
        used,
    )


def derive_bundle_state(
    entry_count: int,
    budget_omission_count: int,
    coverage: m.EvidenceCoverage,
    *,
    candidate_count: int,
) -> tuple[m.PreviewState, m.InsufficientReason | None]:
    if entry_count == 0:
        reason = (
            m.InsufficientReason.NO_CANDIDATES
            if candidate_count == 0
            else m.InsufficientReason.BUDGET_TOO_SMALL
        )
        return m.PreviewState.INSUFFICIENT_EVIDENCE, reason
    if budget_omission_count or coverage.status != m.CoverageStatus.NO_FLAGGED_NATIVE_GAPS:
        return m.PreviewState.PREVIEW_PARTIAL, None
    return m.PreviewState.PREVIEW_READY, None
