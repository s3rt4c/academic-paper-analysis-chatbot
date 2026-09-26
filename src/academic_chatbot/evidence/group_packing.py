"""Atomic packing for the opt-in standalone/group evidence representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from academic_chatbot.evidence import groups as g
from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.budget import measure_proposed_preview, render_content_preview


@dataclass(frozen=True)
class GroupPackingResult:
    entries: tuple[m.EvidenceBundleEntry, ...]
    context: tuple[m.EvidenceContext, ...]
    logical_units: tuple[g.EvidenceGroupLogicalUnitType, ...]
    dispositions: tuple[g.DispositionType, ...]
    content_preview: m.ContentPreview
    used_content_bytes: int


def _entries_for_ranges(
    ranges: tuple[m.ResolvedEvidenceRange, ...], start_index: int
) -> tuple[m.EvidenceBundleEntry, ...]:
    return tuple(
        m.EvidenceBundleEntry(
            **value.model_dump(mode="python"), citation_label=f"E{start_index + offset + 1}"
        )
        for offset, value in enumerate(ranges)
    )


def _contexts_for_ranges(
    contexts: tuple[m.ResolvedContext, ...], labels: tuple[m.CitationLabel, ...]
) -> tuple[m.EvidenceContext, ...]:
    return tuple(
        m.EvidenceContext(
            **context.model_dump(mode="python"), related_citation_labels=labels
        )
        for context in contexts
    )


def _violations(
    *, entries_failed: bool, bytes_failed: bool
) -> tuple[Literal["entry_limit_exceeded", "preview_bytes_exceeded"], ...]:
    values: list[Literal["entry_limit_exceeded", "preview_bytes_exceeded"]] = []
    if entries_failed:
        values.append("entry_limit_exceeded")
    if bytes_failed:
        values.append("preview_bytes_exceeded")
    return tuple(values)


def _standalone_proposal(
    unit: g.ResolvedStandaloneUnit, entry_count: int
) -> tuple[
    tuple[m.EvidenceBundleEntry, ...],
    tuple[m.EvidenceContext, ...],
    tuple[m.CitationLabel, ...],
]:
    entries = _entries_for_ranges(unit.ranges, entry_count)
    labels = tuple(entry.citation_label for entry in entries)
    contexts = (
        ()
        if unit.context is None
        else _contexts_for_ranges((unit.context,), labels)
    )
    return entries, contexts, labels


def _group_proposal(
    unit: g.ResolvedEvidenceGroup, entry_count: int
) -> tuple[
    tuple[m.EvidenceBundleEntry, ...],
    tuple[m.EvidenceContext, ...],
    tuple[m.CitationLabel, ...],
]:
    ranges = tuple(child.range for child in unit.children)
    entries = _entries_for_ranges(ranges, entry_count)
    labels = tuple(entry.citation_label for entry in entries)
    contexts = tuple(
        m.EvidenceContext(
            **child.context.model_dump(mode="python"), related_citation_labels=(label,)
        )
        for child, label in zip(unit.children, labels, strict=True)
        if child.context is not None
    )
    return entries, contexts, labels


def pack_evidence_group_units(
    units: tuple[g.ResolvedEvidenceUnit, ...], budget: m.PreviewBudget
) -> GroupPackingResult:
    entries: tuple[m.EvidenceBundleEntry, ...] = ()
    contexts: tuple[m.EvidenceContext, ...] = ()
    logical_units: list[g.EvidenceGroupLogicalUnitType] = []
    dispositions: list[g.DispositionType] = []
    used = measure_proposed_preview(entries, contexts)

    for unit in units:
        if isinstance(unit, g.ResolvedStandaloneUnit):
            additions, added_context, labels = _standalone_proposal(unit, len(entries))
            proposed_entries = (*entries, *additions)
            proposed_context = (*contexts, *added_context)
            proposed_bytes = measure_proposed_preview(proposed_entries, proposed_context)
            entries_failed = len(proposed_entries) > budget.max_entries
            bytes_failed = proposed_bytes > budget.max_content_bytes
            if entries_failed or bytes_failed:
                violations = _violations(
                    entries_failed=entries_failed, bytes_failed=bytes_failed
                )
                disposition = g.StandaloneDisposition(
                    input_position=unit.input_position,
                    reference=unit.reference,
                    omission=m.EvidenceOmission(
                        reason=(
                            m.OmissionReason.ENTRY_LIMIT_EXCEEDED
                            if entries_failed
                            else m.OmissionReason.PREVIEW_BYTES_EXCEEDED
                        ),
                        violated_constraints=violations,
                    ),
                )
                dispositions.append(disposition)
                continue
            entries, contexts, used = proposed_entries, proposed_context, proposed_bytes
            logical_units.append(
                g.StandaloneLogicalUnit(
                    input_position=unit.input_position, citation_labels=labels
                )
            )
            dispositions.append(
                g.StandaloneDisposition(
                    input_position=unit.input_position,
                    reference=unit.reference,
                    citation_labels=labels,
                )
            )
            continue

        additions, added_context, labels = _group_proposal(unit, len(entries))
        proposed_entries = (*entries, *additions)
        proposed_context = (*contexts, *added_context)
        proposed_bytes = measure_proposed_preview(proposed_entries, proposed_context)
        entries_failed = len(proposed_entries) > budget.max_entries
        bytes_failed = proposed_bytes > budget.max_content_bytes
        member_positions: tuple[m.InputPosition, m.InputPosition] = (
            unit.children[0].input_position,
            unit.children[1].input_position,
        )
        if entries_failed or bytes_failed:
            violations = _violations(entries_failed=entries_failed, bytes_failed=bytes_failed)
            dispositions.append(
                g.EvidenceGroupDisposition(
                    group_id=unit.group_id,
                    grouping_policy_id=unit.grouping_policy_id,
                    member_input_positions=member_positions,
                    omission=g.EvidenceGroupOmission(
                        reason=(
                            "entry_limit_exceeded"
                            if entries_failed
                            else "preview_bytes_exceeded"
                        ),
                        violated_constraints=violations,
                    ),
                )
            )
            continue
        entries, contexts, used = proposed_entries, proposed_context, proposed_bytes
        logical_units.append(
            g.EvidenceGroupLogicalUnit(
                group_id=unit.group_id,
                grouping_policy_id=unit.grouping_policy_id,
                children=(
                    g.GroupChildLogicalUnit(
                        citation_label=labels[0], child_identity=unit.children[0].identity
                    ),
                    g.GroupChildLogicalUnit(
                        citation_label=labels[1], child_identity=unit.children[1].identity
                    ),
                ),
            )
        )
        dispositions.append(
            g.EvidenceGroupDisposition(
                group_id=unit.group_id,
                grouping_policy_id=unit.grouping_policy_id,
                member_input_positions=member_positions,
                citation_labels=labels,
            )
        )

    return GroupPackingResult(
        entries=entries,
        context=contexts,
        logical_units=tuple(logical_units),
        dispositions=tuple(dispositions),
        content_preview=render_content_preview(entries, contexts),
        used_content_bytes=used,
    )
