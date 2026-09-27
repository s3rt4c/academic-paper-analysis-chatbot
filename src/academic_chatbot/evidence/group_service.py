"""Post-resolution construction of the opt-in evidence-group envelope."""

from __future__ import annotations

from academic_chatbot.evidence import groups as g
from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.group_packing import pack_evidence_group_units
from academic_chatbot.evidence.packing import derive_bundle_state
from academic_chatbot.evidence.resolver import EvidenceReadResolver, EvidenceResolutionError


def _preparation_error(
    code: g.EvidenceGroupErrorCode,
    *,
    unit_index: int | None = None,
    input_position: int | None = None,
) -> g.EvidenceGroupPreparationError:
    return g.EvidenceGroupPreparationError(
        g.EvidenceGroupError(
            code=code,
            unit_index=unit_index,
            input_position=input_position,
        )
    )


class EvidenceGroupService:
    def __init__(self, *, resolver: EvidenceReadResolver) -> None:
        self._resolver = resolver

    def build(self, request: g.EvidenceGroupRequest) -> g.EvidenceGroupBundle:
        try:
            request = g.EvidenceGroupRequest.model_validate(request)
        except (TypeError, ValueError):
            raise _preparation_error(g.EvidenceGroupErrorCode.INVALID_REQUEST) from None

        try:
            resolved = self._resolver.resolve(request.base_request)
        except EvidenceResolutionError as error:
            raise _preparation_error(
                g.EvidenceGroupErrorCode.GROUP_MEMBER_RESOLUTION_FAILED,
                input_position=error.error.input_position,
            ) from None
        except Exception:
            raise _preparation_error(
                g.EvidenceGroupErrorCode.GROUP_MEMBER_RESOLUTION_FAILED
            ) from None

        by_position: dict[int, m.ResolvedCandidateGroup] = {}
        for resolved_group in resolved.groups:
            if resolved_group.input_position in by_position:
                raise _preparation_error(g.EvidenceGroupErrorCode.GROUP_MEMBER_RESOLUTION_FAILED)
            by_position[resolved_group.input_position] = resolved_group

        units: list[g.ResolvedEvidenceUnit] = []
        seen_children: set[str] = set()
        seen_groups: set[str] = set()

        for unit_index, unit in enumerate(request.units):
            if isinstance(unit, g.StandaloneUnitRequest):
                candidate = by_position.get(unit.input_position)
                if candidate is None:
                    raise _preparation_error(
                        g.EvidenceGroupErrorCode.GROUP_MEMBER_RESOLUTION_FAILED,
                        unit_index=unit_index,
                        input_position=unit.input_position,
                    )
                standalone = g.ResolvedStandaloneUnit(
                    input_position=unit.input_position,
                    reference=candidate.reference,
                    ranges=candidate.ranges,
                    context=candidate.context,
                )
                for value in standalone.ranges:
                    key = g.authoritative_child_identity(value).model_dump_json()
                    if key in seen_children:
                        raise _preparation_error(
                            g.EvidenceGroupErrorCode.DUPLICATE_LOGICAL_UNIT,
                            unit_index=unit_index,
                            input_position=unit.input_position,
                        )
                    seen_children.add(key)
                units.append(standalone)
                continue

            children: list[g.EvidenceGroupChild] = []
            for member in unit.members:
                candidate = by_position.get(member.input_position)
                if candidate is None:
                    raise _preparation_error(
                        g.EvidenceGroupErrorCode.GROUP_MEMBER_RESOLUTION_FAILED,
                        unit_index=unit_index,
                        input_position=member.input_position,
                    )
                matches = [
                    value
                    for value in candidate.ranges
                    if g.authoritative_child_identity(value) == member.child_identity
                ]
                if len(matches) != 1:
                    raise _preparation_error(
                        g.EvidenceGroupErrorCode.INVALID_GROUP_IDENTITY,
                        unit_index=unit_index,
                        input_position=member.input_position,
                    )
                value = matches[0]
                identity = g.authoritative_child_identity(value)
                children.append(
                    g.EvidenceGroupChild(
                        input_position=member.input_position,
                        identity=identity,
                        range=value,
                        context=candidate.context,
                    )
                )

            if children[0].identity == children[1].identity:
                raise _preparation_error(
                    g.EvidenceGroupErrorCode.DUPLICATE_GROUP_MEMBER, unit_index=unit_index
                )
            if not g.same_group_lineage(children[0].identity, children[1].identity):
                raise _preparation_error(
                    g.EvidenceGroupErrorCode.GROUP_LINEAGE_MISMATCH, unit_index=unit_index
                )
            ordered = tuple(
                sorted(children, key=lambda child: g.canonical_child_sort_key(child.identity))
            )
            if g.canonical_range_overlap(ordered[0].identity, ordered[1].identity):
                raise _preparation_error(
                    g.EvidenceGroupErrorCode.GROUP_RANGE_OVERLAP, unit_index=unit_index
                )
            identities = (ordered[0].identity, ordered[1].identity)
            group_id = g.canonical_group_id(unit.grouping_policy_id, identities)
            if group_id in seen_groups:
                raise _preparation_error(
                    g.EvidenceGroupErrorCode.DUPLICATE_GROUP, unit_index=unit_index
                )
            if any(child.identity.model_dump_json() in seen_children for child in ordered):
                raise _preparation_error(
                    g.EvidenceGroupErrorCode.DUPLICATE_LOGICAL_UNIT, unit_index=unit_index
                )
            seen_groups.add(group_id)
            seen_children.update(child.identity.model_dump_json() for child in ordered)
            units.append(
                g.ResolvedEvidenceGroup(
                    group_id=group_id,
                    grouping_policy_id=unit.grouping_policy_id,
                    children=(ordered[0], ordered[1]),
                )
            )

        try:
            packed = pack_evidence_group_units(tuple(units), resolved.request.preview_budget)
            coverage = resolved.coverage.model_copy(
                update={
                    "packed_page_ids": tuple(
                        dict.fromkeys(entry.source.parent.page_id for entry in packed.entries)
                    )
                }
            )
            omitted = sum(disposition.omission is not None for disposition in packed.dispositions)
            state, insufficient_reason = derive_bundle_state(
                len(packed.entries), omitted, coverage, candidate_count=len(resolved.groups)
            )
            payload = g.EvidenceGroupBundlePayload(
                scope=resolved.resolved_scope,
                origin=resolved.request.origin,
                logical_units=packed.logical_units,
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
                insufficient_reason=insufficient_reason,
                content_preview=packed.content_preview,
            )
            bundle = g.seal_evidence_group_bundle(payload)
            g.canonical_evidence_group_bytes(bundle)
            return bundle
        except g.EvidenceGroupPreparationError:
            raise
        except Exception:
            raise _preparation_error(g.EvidenceGroupErrorCode.GROUP_PACKING_FAILED) from None
