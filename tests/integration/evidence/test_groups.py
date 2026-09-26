"""Synthetic resolver-to-group-service coverage."""

from pathlib import Path

import pytest

from academic_chatbot.evidence import groups
from academic_chatbot.evidence.group_service import EvidenceGroupService
from academic_chatbot.evidence.resolver import EvidenceReadResolver
from tests.fixtures.evidence_bundle.database import database


class CountingResolver:
    def __init__(self, data_root: Path) -> None:
        self._delegate = EvidenceReadResolver(data_root=data_root)
        self.calls = 0

    def resolve(self, request):
        self.calls += 1
        return self._delegate.resolve(request)


def group_request(base_request, resolved, *, positions=(0, 1), units=None):
    members = tuple(
        groups.EvidenceGroupMemberRequest(
            input_position=position,
            child_identity=groups.authoritative_child_identity(
                resolved.groups[position].ranges[0]
            ),
        )
        for position in positions
    )
    return groups.EvidenceGroupRequest(
        base_request=base_request,
        units=units
        or (
            groups.EvidenceGroupUnitRequest(
                grouping_policy_id="explicit-pair-v1", members=members
            ),
        ),
    )


def test_service_resolves_once_and_preserves_two_independent_children(tmp_path: Path):
    db = database(tmp_path)
    base = db.semantic(maximum_words=12)
    resolved = EvidenceReadResolver(data_root=db.paths.data_root).resolve(base)
    assert len(resolved.groups) >= 2
    request = group_request(base, resolved)
    resolver = CountingResolver(db.paths.data_root)

    bundle = EvidenceGroupService(resolver=resolver).build(request)

    assert resolver.calls == 1
    assert len(bundle.entries) == 2
    assert bundle.logical_units[0].kind == "evidence_group"
    assert bundle.dispositions[0].citation_labels == ("E1", "E2")
    assert bundle.entries[0].citation_label != bundle.entries[1].citation_label
    assert len({entry.source.text_sha256 for entry in bundle.entries}) == 2

    reversed_bundle = EvidenceGroupService(
        resolver=EvidenceReadResolver(data_root=db.paths.data_root)
    ).build(group_request(base, resolved, positions=(1, 0)))
    assert reversed_bundle.model_dump(mode="json") == bundle.model_dump(mode="json")


def test_service_rejects_a_child_identity_not_returned_by_authority(tmp_path: Path):
    db = database(tmp_path)
    base = db.semantic(maximum_words=12)
    resolved = EvidenceReadResolver(data_root=db.paths.data_root).resolve(base)
    request = group_request(base, resolved)
    member = request.units[0].members[1]
    forged = member.child_identity.model_copy(update={"text_sha256": "a" * 64})
    forged_request = request.model_copy(
        update={
            "units": (
                request.units[0].model_copy(
                    update={
                        "members": (
                            request.units[0].members[0],
                            member.model_copy(update={"child_identity": forged}),
                        )
                    }
                ),
            )
        }
    )
    with pytest.raises(groups.EvidenceGroupPreparationError) as error:
        EvidenceGroupService(resolver=CountingResolver(db.paths.data_root)).build(forged_request)
    assert error.value.error.code == "INVALID_GROUP_IDENTITY"


def test_service_rejects_duplicate_group_ids(tmp_path: Path):
    db = database(tmp_path)
    base = db.semantic(maximum_words=12)
    base = base.model_copy(
        update={
            "candidates": (
                base.candidates[0].model_copy(update={"reported_rank": 1}),
                base.candidates[1].model_copy(update={"reported_rank": 2}),
                base.candidates[0].model_copy(update={"reported_rank": 3}),
                base.candidates[1].model_copy(update={"reported_rank": 4}),
            )
        }
    )
    resolved = EvidenceReadResolver(data_root=db.paths.data_root).resolve(base)
    first_members = tuple(
        groups.EvidenceGroupMemberRequest(
            input_position=position,
            child_identity=groups.authoritative_child_identity(resolved.groups[position].ranges[0]),
        )
        for position in (0, 1)
    )
    second_members = tuple(
        groups.EvidenceGroupMemberRequest(
            input_position=position,
            child_identity=groups.authoritative_child_identity(resolved.groups[position].ranges[0]),
        )
        for position in (2, 3)
    )
    request = groups.EvidenceGroupRequest(
        base_request=base,
        units=(
            groups.EvidenceGroupUnitRequest(
                grouping_policy_id="explicit-pair-v1", members=first_members
            ),
            groups.EvidenceGroupUnitRequest(
                grouping_policy_id="explicit-pair-v1", members=second_members
            ),
        ),
    )
    with pytest.raises(groups.EvidenceGroupPreparationError) as error:
        EvidenceGroupService(resolver=CountingResolver(db.paths.data_root)).build(request)
    assert error.value.error.code == "DUPLICATE_GROUP"
