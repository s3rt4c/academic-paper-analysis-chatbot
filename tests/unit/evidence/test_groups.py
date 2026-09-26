"""Synthetic unit coverage for the opt-in evidence-group representation."""

from hashlib import sha256

import pytest
from pydantic import ValidationError

from academic_chatbot.evidence import groups
from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.budget import measure_proposed_preview
from academic_chatbot.evidence.group_packing import pack_evidence_group_units
from tests.fixtures.evidence_bundle.contracts import request_data, resolved_range_data


def resolved_range(
    *, start: int = 0, end: int = 10, text: str = "alpha beta", chunk: str = "chunk-1"
):
    base = m.ResolvedEvidenceRange.model_validate(resolved_range_data())
    source = base.source.model_copy(
        update={
            "parent": base.source.parent.model_copy(update={"chunk_id": chunk}),
            "start_offset": start,
            "end_offset": end,
            "text_sha256": sha256(text.encode()).hexdigest(),
        }
    )
    return m.ResolvedEvidenceRange(
        source=source,
        text=text,
        anchors=base.anchors,
        contributions=base.contributions,
    )


def candidate_request(*, rank: int = 1, chunk: str = "chunk-1") -> m.EvidenceCandidateRef:
    data = request_data()["candidates"][0]
    data = {
        **data,
        "reported_rank": rank,
        "parent": {**data["parent"], "chunk_id": chunk},
    }
    return m.EvidenceCandidateRef.model_validate(data)


def scope_coverage() -> m.EvidenceCoverage:
    return m.EvidenceCoverage(
        status=m.CoverageStatus.NO_FLAGGED_NATIVE_GAPS,
        pages=(),
        candidate_page_ids=("page-1",),
        packed_page_ids=(),
        flagged_page_ids=(),
        excluded_page_ids=(),
    )


def group_child(value: m.ResolvedEvidenceRange, position: int) -> groups.EvidenceGroupChild:
    return groups.EvidenceGroupChild(
        input_position=position,
        identity=groups.authoritative_child_identity(value),
        range=value,
    )


def resolved_context(value: m.ResolvedEvidenceRange, text: str) -> m.ResolvedContext:
    return m.ResolvedContext(
        parent=value.source.parent,
        physical_page_index=value.source.physical_page_index,
        start_offset=0,
        end_offset=len(text),
        text=text,
        text_sha256=sha256(text.encode()).hexdigest(),
        anchors=value.anchors,
    )


def group_unit(
    first: m.ResolvedEvidenceRange | None = None,
    second: m.ResolvedEvidenceRange | None = None,
) -> groups.ResolvedEvidenceGroup:
    first = first or resolved_range()
    second = second or resolved_range(start=10, end=20, text="gamma delta", chunk="chunk-2")
    children = tuple(
        sorted(
            (group_child(first, 0), group_child(second, 1)),
            key=lambda c: groups.canonical_child_sort_key(c.identity),
        )
    )
    return groups.ResolvedEvidenceGroup(
        group_id=groups.canonical_group_id(
            "explicit-pair-v1", (children[0].identity, children[1].identity)
        ),
        grouping_policy_id="explicit-pair-v1",
        children=children,
    )


def test_request_requires_exact_two_children_and_rejects_repeated_positions():
    first = resolved_range()
    second = resolved_range(start=10, end=20, text="gamma delta", chunk="chunk-2")
    base = m.EvidenceBundleRequest(
        **request_data()
    )
    valid = groups.EvidenceGroupRequest(
        base_request=base,
        units=(
            groups.EvidenceGroupUnitRequest(
                grouping_policy_id="explicit-pair-v1",
                members=(
                    groups.EvidenceGroupMemberRequest(
                        input_position=0,
                        child_identity=groups.authoritative_child_identity(first),
                    ),
                    groups.EvidenceGroupMemberRequest(
                        input_position=1,
                        child_identity=groups.authoritative_child_identity(second),
                    ),
                ),
            ),
        ),
    )
    assert valid.policy_id == "evidence-group-v1"
    assert valid.units[0].requires_all is True
    with pytest.raises(ValidationError):
        groups.EvidenceGroupUnitRequest(
            grouping_policy_id="explicit-pair-v1",
            members=(
                groups.EvidenceGroupMemberRequest(
                    input_position=0,
                    child_identity=groups.authoritative_child_identity(first),
                ),
                groups.EvidenceGroupMemberRequest(
                    input_position=0,
                    child_identity=groups.authoritative_child_identity(second),
                ),
            ),
        )


def test_identity_order_overlap_and_group_id_are_deterministic():
    first = resolved_range()
    second = resolved_range(start=10, end=20, text="gamma delta", chunk="chunk-2")
    first_id = groups.authoritative_child_identity(first)
    second_id = groups.authoritative_child_identity(second)
    assert groups.same_group_lineage(first_id, second_id)
    assert groups.canonical_child_sort_key(first_id) < groups.canonical_child_sort_key(second_id)
    assert not groups.canonical_range_overlap(first_id, second_id)
    group_id = groups.canonical_group_id("explicit-pair-v1", (first_id, second_id))
    assert group_id.startswith("sha256-")
    with pytest.raises(ValueError):
        groups.canonical_group_id("explicit-pair-v1", (second_id, first_id))


def test_overlap_is_strict_and_lineage_is_exact():
    first = groups.authoritative_child_identity(resolved_range(start=0, end=10))
    adjacent = groups.authoritative_child_identity(
        resolved_range(start=10, end=20, text="gamma delta", chunk="chunk-2")
    )
    overlap = groups.authoritative_child_identity(
        resolved_range(start=9, end=20, text="gamma delta", chunk="chunk-2")
    )
    assert not groups.canonical_range_overlap(first, adjacent)
    assert groups.canonical_range_overlap(first, overlap)
    foreign_scope = first.model_copy(
        update={"scope": first.scope.model_copy(update={"file_version_id": "file-foreign"})}
    )
    assert not groups.same_group_lineage(first, foreign_scope)


def test_group_error_messages_are_fixed_and_source_independent():
    assert {code.value for code in groups.EvidenceGroupErrorCode} == {
        "INVALID_REQUEST",
        "RESOURCE_LIMIT",
        "INVALID_GROUP_IDENTITY",
        "DUPLICATE_GROUP_MEMBER",
        "DUPLICATE_GROUP",
        "DUPLICATE_LOGICAL_UNIT",
        "GROUP_LINEAGE_MISMATCH",
        "GROUP_MEMBER_RESOLUTION_FAILED",
        "GROUP_RANGE_OVERLAP",
        "GROUP_PACKING_FAILED",
    }
    error = groups.EvidenceGroupError(code=groups.EvidenceGroupErrorCode.INVALID_REQUEST)
    assert error.message == "Evidence group request is invalid"
    assert "raw" not in str(groups.EvidenceGroupPreparationError(error))


def test_group_packer_accepts_two_children_with_two_labels_and_logical_unit():
    unit = group_unit()
    result = pack_evidence_group_units((unit,), m.PreviewBudget())
    assert tuple(entry.citation_label for entry in result.entries) == ("E1", "E2")
    assert len(result.entries) == 2
    assert len(result.logical_units) == 1
    assert result.logical_units[0].kind == "evidence_group"
    assert result.dispositions[0].citation_labels == ("E1", "E2")
    assert result.used_content_bytes == measure_proposed_preview(result.entries, result.context)


def test_group_packer_keeps_each_child_context_associated_with_its_child_label():
    first = resolved_range()
    second = resolved_range(start=10, end=20, text="gamma delta", chunk="chunk-2")
    first_context = resolved_context(first, "first context")
    second_context = resolved_context(second, "second context")
    children = tuple(
        sorted(
            (
                groups.EvidenceGroupChild(
                    input_position=0,
                    identity=groups.authoritative_child_identity(first),
                    range=first,
                    context=first_context,
                ),
                groups.EvidenceGroupChild(
                    input_position=1,
                    identity=groups.authoritative_child_identity(second),
                    range=second,
                    context=second_context,
                ),
            ),
            key=lambda child: groups.canonical_child_sort_key(child.identity),
        )
    )
    unit = groups.ResolvedEvidenceGroup(
        group_id=groups.canonical_group_id(
            "explicit-pair-v1", (children[0].identity, children[1].identity)
        ),
        grouping_policy_id="explicit-pair-v1",
        children=children,
    )

    result = pack_evidence_group_units((unit,), m.PreviewBudget())

    assert tuple(context.related_citation_labels for context in result.context) == (
        ("E1",),
        ("E2",),
    )


def test_group_packer_omits_both_children_when_one_entry_slot_remains():
    unit = group_unit()
    result = pack_evidence_group_units(
        (unit,), m.PreviewBudget(max_entries=1)
    )
    assert result.entries == ()
    assert result.context == ()
    assert result.logical_units == ()
    assert result.dispositions[0].omission.reason == "entry_limit_exceeded"
    assert result.used_content_bytes == measure_proposed_preview((), ())


def test_group_packer_omits_both_children_when_utf8_budget_is_too_small():
    unit = group_unit()
    result = pack_evidence_group_units(
        (unit,), m.PreviewBudget(max_content_bytes=77)
    )
    assert result.entries == ()
    assert result.dispositions[0].omission.reason == "preview_bytes_exceeded"
    assert result.used_content_bytes == 77


def test_group_packer_keeps_mixed_unit_order_and_labels():
    first = resolved_range()
    second = resolved_range(start=10, end=20, text="gamma delta", chunk="chunk-2")
    standalone = groups.ResolvedStandaloneUnit(
        input_position=0,
        reference=candidate_request(),
        ranges=(first,),
    )
    result = pack_evidence_group_units((standalone, group_unit(first, second)), m.PreviewBudget())
    assert tuple(entry.citation_label for entry in result.entries) == ("E1", "E2", "E3")
    assert tuple(unit.kind for unit in result.logical_units) == ("standalone", "evidence_group")
