"""Deterministic downstream selection for additive positional acquisition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.semantic_position import (
    MAX_POSITIONAL_AUXILIARY_HITS,
    MAX_POSITIONAL_HITS_EXAMINED,
    SEMANTIC_POSITIONAL_ACQUISITION_POLICY_ID,
    SUPPORTED_POSITIONAL_CONCERN_ID,
    SemanticPositionalAcquisition,
)

GUARDED_EARLIEST_AUX_SELECTOR_POLICY_ID = "guarded-earliest-aux-selector-v1"
MAX_SELECTED_CANDIDATES = 8

type OccurrenceIdentity = tuple[str, str, str, str, str, str, str, str, int, int]
type CandidateProvenance = Literal[
    "BASELINE-ONLY", "POSITIONAL-ONLY", "BOTH"
]


class GuardedAuxSelectorError(ValueError):
    """Raised when guarded selection cannot prove its input contract."""


@dataclass(frozen=True, slots=True)
class CanonicalPosition:
    """Authoritative integer position for one acquired semantic occurrence."""

    occurrence_identity: OccurrenceIdentity
    document_generation_id: str
    page_id: str
    absolute_start: int
    source_policy_id: str


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
    """One validated acquired occurrence with selection provenance."""

    hit: SemanticRetrievalHit
    provenance: CandidateProvenance
    canonical_position: CanonicalPosition


def occurrence_identity(hit: SemanticRetrievalHit) -> OccurrenceIdentity:
    """Return the existing occurrence identity without changing its fields."""

    if not isinstance(hit, SemanticRetrievalHit):
        raise GuardedAuxSelectorError("semantic occurrence is invalid")
    values = (
        hit.project_id,
        hit.file_version_id,
        hit.document_generation_id,
        hit.page_id,
        hit.chunk_id,
        hit.embedding_span_id,
        hit.embedding_profile_id,
        hit.vector_generation_id,
        hit.start_offset,
        hit.end_offset,
    )
    if any(type(value) is not str or not value for value in values[:8]):
        raise GuardedAuxSelectorError("semantic occurrence identity is invalid")
    if type(values[8]) is not int or type(values[9]) is not int:
        raise GuardedAuxSelectorError("semantic occurrence range is invalid")
    if values[8] < 0 or values[9] <= values[8]:
        raise GuardedAuxSelectorError("semantic occurrence range is invalid")
    return values


def select_guarded_earliest_auxiliary(
    *,
    acquisition: SemanticPositionalAcquisition,
    current: tuple[SemanticRetrievalHit, ...],
    positions: Mapping[OccurrenceIdentity, CanonicalPosition],
    selector_policy_id: str,
) -> tuple[SemanticRetrievalHit, ...]:
    """Select the existing current prefix plus at most one positional auxiliary."""

    if selector_policy_id != GUARDED_EARLIEST_AUX_SELECTOR_POLICY_ID:
        raise GuardedAuxSelectorError("selector policy is unsupported")
    candidates = _validate_acquisition(acquisition, positions)
    if type(current) is not tuple or len(current) > MAX_SELECTED_CANDIDATES:
        raise GuardedAuxSelectorError("current selection exceeds the bound")

    current_ids: set[OccurrenceIdentity] = set()
    for hit in current:
        identity = occurrence_identity(hit)
        if identity in current_ids:
            raise GuardedAuxSelectorError("current selection contains a duplicate")
        candidate = candidates.get(identity)
        if candidate is None or candidate.hit != hit:
            raise GuardedAuxSelectorError("current selection is outside the acquisition")
        current_ids.add(identity)

    if len(current) < MAX_SELECTED_CANDIDATES:
        preserved = current
    else:
        preserved = current[: MAX_SELECTED_CANDIDATES - 1]

    eligible = [
        candidate
        for identity, candidate in candidates.items()
        if candidate.provenance == "POSITIONAL-ONLY" and identity not in current_ids
    ]
    if not eligible:
        return current

    auxiliary = min(
        eligible,
        key=lambda candidate: (
            candidate.canonical_position.absolute_start,
            candidate.hit.rank,
            occurrence_identity(candidate.hit),
        ),
    )
    return (*preserved, auxiliary.hit)


def _validate_acquisition(
    acquisition: SemanticPositionalAcquisition,
    positions: Mapping[OccurrenceIdentity, CanonicalPosition],
) -> dict[OccurrenceIdentity, SelectionCandidate]:
    if not isinstance(acquisition, SemanticPositionalAcquisition):
        raise GuardedAuxSelectorError("positional acquisition is invalid")
    if acquisition.policy_id != SEMANTIC_POSITIONAL_ACQUISITION_POLICY_ID:
        raise GuardedAuxSelectorError("positional acquisition policy is unsupported")
    if acquisition.concern_id != SUPPORTED_POSITIONAL_CONCERN_ID:
        raise GuardedAuxSelectorError("positional acquisition concern is unsupported")
    if type(positions) is not dict and not isinstance(positions, Mapping):
        raise GuardedAuxSelectorError("canonical positions are invalid")
    if len(acquisition.examined_hits) > MAX_POSITIONAL_HITS_EXAMINED:
        raise GuardedAuxSelectorError("examined acquisition exceeds the bound")
    if len(acquisition.positional_hits) > MAX_POSITIONAL_AUXILIARY_HITS:
        raise GuardedAuxSelectorError("positional acquisition exceeds the bound")
    if acquisition.ordered_hits != (
        acquisition.baseline_hits + acquisition.positional_hits
    ):
        raise GuardedAuxSelectorError("ordered acquisition is inconsistent")

    examined: dict[OccurrenceIdentity, SemanticRetrievalHit] = {}
    for hit in acquisition.examined_hits:
        identity = occurrence_identity(hit)
        previous = examined.get(identity)
        if previous is not None and previous != hit:
            raise GuardedAuxSelectorError("examined occurrence is contradictory")
        examined[identity] = hit

    partition_ids: dict[OccurrenceIdentity, set[str]] = {}
    partition_hits: dict[OccurrenceIdentity, SemanticRetrievalHit] = {}
    for partition_name, hits in (
        ("BASELINE-ONLY", acquisition.baseline_hits),
        ("POSITIONAL-ONLY", acquisition.positional_hits),
    ):
        for hit in hits:
            identity = occurrence_identity(hit)
            if identity not in examined:
                raise GuardedAuxSelectorError("partition occurrence is not examined")
            previous = partition_hits.get(identity)
            if previous is not None and previous != hit:
                raise GuardedAuxSelectorError("partition occurrence is contradictory")
            partition_hits[identity] = hit
            partition_ids.setdefault(identity, set()).add(partition_name)

    if set(positions) != set(partition_ids):
        raise GuardedAuxSelectorError("canonical positions do not match acquisition")

    candidates: dict[OccurrenceIdentity, SelectionCandidate] = {}
    for identity, hit in partition_hits.items():
        position = _validate_position(hit, identity, positions.get(identity))
        provenance_names = partition_ids[identity]
        if provenance_names == {"BASELINE-ONLY"}:
            provenance: CandidateProvenance = "BASELINE-ONLY"
        elif provenance_names == {"POSITIONAL-ONLY"}:
            provenance = "POSITIONAL-ONLY"
        elif provenance_names == {"BASELINE-ONLY", "POSITIONAL-ONLY"}:
            provenance = "BOTH"
        else:
            raise GuardedAuxSelectorError("occurrence provenance is invalid")
        if hit.project_id != acquisition.project_id:
            raise GuardedAuxSelectorError("candidate project lineage is stale")
        if hit.embedding_profile_id != acquisition.embedding_profile_id:
            raise GuardedAuxSelectorError("candidate profile lineage is stale")
        if hit.vector_generation_id != acquisition.vector_generation_id:
            raise GuardedAuxSelectorError("candidate vector lineage is stale")
        candidates[identity] = SelectionCandidate(
            hit=hit,
            provenance=provenance,
            canonical_position=position,
        )
    return candidates


def _validate_position(
    hit: SemanticRetrievalHit,
    identity: OccurrenceIdentity,
    position: CanonicalPosition | None,
) -> CanonicalPosition:
    if not isinstance(position, CanonicalPosition):
        raise GuardedAuxSelectorError("authoritative canonical position is missing")
    if position.occurrence_identity != identity:
        raise GuardedAuxSelectorError("canonical position identity is invalid")
    if position.source_policy_id != SEMANTIC_POSITIONAL_ACQUISITION_POLICY_ID:
        raise GuardedAuxSelectorError("canonical position source is unsupported")
    if (
        type(position.document_generation_id) is not str
        or not position.document_generation_id
        or position.document_generation_id != hit.document_generation_id
        or type(position.page_id) is not str
        or not position.page_id
        or position.page_id != hit.page_id
        or type(position.absolute_start) is not int
        or position.absolute_start < 0
    ):
        raise GuardedAuxSelectorError("authoritative canonical position is invalid")
    return position
