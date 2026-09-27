"""Production integration for the explicit rank-only parent reranker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

from academic_chatbot.domain.library import Project
from academic_chatbot.retrieval.rank_only_rerank import (
    MAX_RERANK_CANDIDATES,
    ParentOccurrenceKey,
    RankedParentOccurrence,
    apply_rank_only_parent_rerank,
)
from academic_chatbot.retrieval.selection import (
    GUARDED_EARLIEST_AUX_SELECTOR_POLICY_ID,
    MAX_SELECTED_CANDIDATES,
    CanonicalPosition,
    OccurrenceIdentity,
    occurrence_identity,
    select_guarded_earliest_auxiliary,
)
from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.semantic_position import (
    MAX_POSITIONAL_AUXILIARY_HITS,
    MAX_POSITIONAL_HITS_EXAMINED,
    SEMANTIC_POSITIONAL_ACQUISITION_POLICY_ID,
    SemanticPositionalAcquisition,
)
from academic_chatbot.retrieval.service import RetrievalHit, RetrievalResults

PRODUCTION_RANK_ONLY_PARENT_RERANK_POLICY_ID = "rank-only-parent-rerank-v1"
SUPPORTED_PRODUCTION_RANK_ONLY_CONCERN_ID = "stated-study-objective-v1"
MAX_PRODUCTION_LEXICAL_RANK_DEPTH = 100

_TupleItemT = TypeVar("_TupleItemT")


class ProductionRankOnlyError(ValueError):
    """Raised when production rank-only integration cannot prove its contract."""


class ProductionRankOnlyUnsupportedError(ProductionRankOnlyError):
    """Raised when the requested concern is outside this explicit integration."""


@dataclass(frozen=True, slots=True)
class ProductionRankOnlyResult:
    """The bounded rank-only result and unchanged downstream selection output."""

    policy_id: str | None
    activated: bool
    acquisition: SemanticPositionalAcquisition
    ranked_baseline: tuple[RankedParentOccurrence, ...]
    selected_hits: tuple[SemanticRetrievalHit, ...]
    lexical_retrieval_count: int
    lexical_result_count: int


class _LexicalSearch(Protocol):
    """The existing lexical retrieval boundary used by the explicit policy."""

    def search(self, project: Project, query: str, limit: int) -> RetrievalResults: ...


class ProductionRankOnlyParentRerankService:
    """Apply the opt-in policy around one caller-owned semantic acquisition."""

    def __init__(self, *, lexical_service: _LexicalSearch) -> None:
        self._lexical_service = lexical_service

    def rerank_and_select(
        self,
        *,
        project: Project,
        query: str,
        acquisition: SemanticPositionalAcquisition,
        positions: Mapping[OccurrenceIdentity, CanonicalPosition],
        policy_id: str | None = None,
    ) -> ProductionRankOnlyResult:
        """Optionally annotate and reorder the existing baseline parent lane."""

        baseline = _validate_request(
            project=project,
            query=query,
            acquisition=acquisition,
            positions=positions,
            policy_id=policy_id,
        )
        activated = policy_id == PRODUCTION_RANK_ONLY_PARENT_RERANK_POLICY_ID
        lexical_result_count = 0
        if activated:
            lexical_ranks, lexical_result_count = self._lexical_ranks(
                project=project,
                query=query,
                baseline=baseline,
            )
        else:
            lexical_ranks = {}

        ranked_baseline = apply_rank_only_parent_rerank(
            acquisition=acquisition,
            lexical_ranks=lexical_ranks,
            policy_id=policy_id,
        )
        current = tuple(
            item.hit for item in ranked_baseline[:MAX_SELECTED_CANDIDATES]
        )
        selected_hits = select_guarded_earliest_auxiliary(
            acquisition=acquisition,
            current=current,
            positions=positions,
            selector_policy_id=GUARDED_EARLIEST_AUX_SELECTOR_POLICY_ID,
        )
        return ProductionRankOnlyResult(
            policy_id=policy_id,
            activated=activated,
            acquisition=acquisition,
            ranked_baseline=ranked_baseline,
            selected_hits=selected_hits,
            lexical_retrieval_count=1 if activated else 0,
            lexical_result_count=lexical_result_count,
        )

    def _lexical_ranks(
        self,
        *,
        project: Project,
        query: str,
        baseline: tuple[SemanticRetrievalHit, ...],
    ) -> tuple[dict[ParentOccurrenceKey, int], int]:
        try:
            results = self._lexical_service.search(
                project, query, MAX_PRODUCTION_LEXICAL_RANK_DEPTH
            )
        except Exception as error:
            raise ProductionRankOnlyError("lexical retrieval failed") from error
        lexical_ranks = _validate_lexical_results(
            project=project,
            query=query,
            results=results,
            baseline=baseline,
        )
        return lexical_ranks, len(results.hits)


def _validate_request(
    *,
    project: Project,
    query: str,
    acquisition: SemanticPositionalAcquisition,
    positions: Mapping[OccurrenceIdentity, CanonicalPosition],
    policy_id: str | None,
) -> tuple[SemanticRetrievalHit, ...]:
    if not isinstance(project, Project):
        raise ProductionRankOnlyError("project is invalid")
    if type(query) is not str or not query.strip():
        raise ProductionRankOnlyError("query is invalid")
    if not isinstance(acquisition, SemanticPositionalAcquisition):
        raise ProductionRankOnlyError("positional acquisition is invalid")
    if policy_id is not None and type(policy_id) is not str:
        raise ProductionRankOnlyError("rank-only policy is invalid")
    if acquisition.project_id != project.project_id:
        raise ProductionRankOnlyError("acquisition project scope is inconsistent")
    if acquisition.query != query:
        raise ProductionRankOnlyError("acquisition query scope is inconsistent")
    if acquisition.policy_id != SEMANTIC_POSITIONAL_ACQUISITION_POLICY_ID:
        raise ProductionRankOnlyError("positional acquisition policy is unsupported")
    if acquisition.concern_id != SUPPORTED_PRODUCTION_RANK_ONLY_CONCERN_ID:
        raise ProductionRankOnlyUnsupportedError("rank-only concern is unsupported")
    if (
        type(acquisition.project_id) is not str
        or not acquisition.project_id
        or type(acquisition.query) is not str
        or not acquisition.query.strip()
        or type(acquisition.embedding_profile_id) is not str
        or not acquisition.embedding_profile_id
        or type(acquisition.vector_generation_id) is not str
        or not acquisition.vector_generation_id
    ):
        raise ProductionRankOnlyError("positional acquisition metadata is invalid")
    if (
        type(acquisition.requested_limit) is not int
        or not 1 <= acquisition.requested_limit <= MAX_POSITIONAL_HITS_EXAMINED
    ):
        raise ProductionRankOnlyError("acquisition candidate count exceeds the bound")

    examined: tuple[SemanticRetrievalHit, ...] = _require_tuple(
        acquisition.examined_hits, "examined acquisition"
    )
    baseline: tuple[SemanticRetrievalHit, ...] = _require_tuple(
        acquisition.baseline_hits, "baseline acquisition"
    )
    positional: tuple[SemanticRetrievalHit, ...] = _require_tuple(
        acquisition.positional_hits, "positional acquisition"
    )
    ordered: tuple[SemanticRetrievalHit, ...] = _require_tuple(
        acquisition.ordered_hits, "ordered acquisition"
    )
    if (
        len(examined) > MAX_POSITIONAL_HITS_EXAMINED
        or len(baseline) > MAX_RERANK_CANDIDATES
        or len(positional) > MAX_POSITIONAL_AUXILIARY_HITS
        or len(ordered) > MAX_POSITIONAL_HITS_EXAMINED
    ):
        raise ProductionRankOnlyError("acquisition candidate count exceeds the bound")
    if ordered != baseline + positional or len(ordered) != len(baseline) + len(positional):
        raise ProductionRankOnlyError("positional acquisition ordering is inconsistent")

    examined_by_id: dict[OccurrenceIdentity, SemanticRetrievalHit] = {}
    for hit in examined:
        identity = _validate_semantic_hit(hit, acquisition)
        previous = examined_by_id.get(identity)
        if previous is not None and previous != hit:
            raise ProductionRankOnlyError("examined occurrence is contradictory")
        examined_by_id[identity] = hit

    baseline_ids: set[OccurrenceIdentity] = set()
    previous_rank = 0
    for hit in baseline:
        identity = _validate_semantic_hit(hit, acquisition)
        if identity in baseline_ids:
            raise ProductionRankOnlyError("baseline occurrence identity is duplicated")
        if hit.rank <= previous_rank:
            raise ProductionRankOnlyError("baseline semantic ranks are not increasing")
        if identity not in examined_by_id:
            raise ProductionRankOnlyError("baseline occurrence is not examined")
        baseline_ids.add(identity)
        previous_rank = hit.rank

    positional_ids: set[OccurrenceIdentity] = set()
    for hit in positional:
        identity = _validate_semantic_hit(hit, acquisition)
        if identity in positional_ids:
            raise ProductionRankOnlyError("positional occurrence identity is duplicated")
        if identity in baseline_ids:
            raise ProductionRankOnlyError("baseline and positional lanes overlap")
        if identity not in examined_by_id:
            raise ProductionRankOnlyError("positional occurrence is not examined")
        positional_ids.add(identity)

    _validate_positions(
        positions=positions,
        expected_ids=baseline_ids | positional_ids,
        hits_by_id={
            occurrence_identity(hit): hit for hit in baseline + positional
        },
    )
    return baseline


def _validate_semantic_hit(
    hit: SemanticRetrievalHit,
    acquisition: SemanticPositionalAcquisition,
) -> OccurrenceIdentity:
    if not isinstance(hit, SemanticRetrievalHit):
        raise ProductionRankOnlyError("semantic occurrence is invalid")
    try:
        identity = occurrence_identity(hit)
    except ValueError as error:
        raise ProductionRankOnlyError("semantic occurrence identity is invalid") from error
    if (
        type(hit.paper_id) is not str
        or not hit.paper_id
        or hit.project_id != acquisition.project_id
        or hit.embedding_profile_id != acquisition.embedding_profile_id
        or hit.vector_generation_id != acquisition.vector_generation_id
        or type(hit.rank) is not int
        or hit.rank < 1
    ):
        raise ProductionRankOnlyError("semantic occurrence lineage is inconsistent")
    return identity


def _validate_positions(
    *,
    positions: Mapping[OccurrenceIdentity, CanonicalPosition],
    expected_ids: set[OccurrenceIdentity],
    hits_by_id: Mapping[OccurrenceIdentity, SemanticRetrievalHit],
) -> None:
    if not isinstance(positions, Mapping):
        raise ProductionRankOnlyError("canonical positions are invalid")
    try:
        actual_ids = set(positions)
    except TypeError as error:
        raise ProductionRankOnlyError("canonical position identity is invalid") from error
    if actual_ids != expected_ids:
        raise ProductionRankOnlyError("canonical positions do not match acquisition")
    for identity in expected_ids:
        position = positions.get(identity)
        hit = hits_by_id[identity]
        if not isinstance(position, CanonicalPosition):
            raise ProductionRankOnlyError("authoritative canonical position is missing")
        if (
            position.occurrence_identity != identity
            or position.source_policy_id != SEMANTIC_POSITIONAL_ACQUISITION_POLICY_ID
            or position.document_generation_id != hit.document_generation_id
            or position.page_id != hit.page_id
            or type(position.absolute_start) is not int
            or position.absolute_start < 0
        ):
            raise ProductionRankOnlyError("authoritative canonical position is invalid")


def _validate_lexical_results(
    *,
    project: Project,
    query: str,
    results: RetrievalResults,
    baseline: tuple[SemanticRetrievalHit, ...],
) -> dict[ParentOccurrenceKey, int]:
    if not isinstance(results, RetrievalResults):
        raise ProductionRankOnlyError("lexical retrieval result is invalid")
    if results.project_id != project.project_id:
        raise ProductionRankOnlyError("lexical result project scope is inconsistent")
    if results.query != query:
        raise ProductionRankOnlyError("lexical result query scope is inconsistent")
    hits: tuple[RetrievalHit, ...] = _require_tuple(results.hits, "lexical results")
    if len(hits) > MAX_PRODUCTION_LEXICAL_RANK_DEPTH:
        raise ProductionRankOnlyError("lexical result depth exceeds the bound")

    baseline_keys = {_parent_key(hit) for hit in baseline}
    baseline_chunks = {key[4] for key in baseline_keys}
    ranks: dict[ParentOccurrenceKey, int] = {}
    for expected_rank, hit in enumerate(hits, start=1):
        if not isinstance(hit, RetrievalHit):
            raise ProductionRankOnlyError("lexical result hit is invalid")
        if hit.project_id != project.project_id:
            raise ProductionRankOnlyError("lexical hit project scope is inconsistent")
        if type(hit.rank) is not int or hit.rank != expected_rank:
            raise ProductionRankOnlyError("lexical result ranks are invalid")
        key = _parent_key(hit)
        if key in ranks:
            raise ProductionRankOnlyError("lexical parent identity is duplicated")
        if key not in baseline_keys and key[4] in baseline_chunks:
            raise ProductionRankOnlyError("lexical scope mismatch")
        ranks[key] = hit.rank
    return ranks


def _parent_key(hit: SemanticRetrievalHit | RetrievalHit) -> ParentOccurrenceKey:
    values = (
        hit.project_id,
        hit.file_version_id,
        hit.document_generation_id,
        hit.page_id,
        hit.chunk_id,
    )
    if any(type(value) is not str or not value for value in values):
        raise ProductionRankOnlyError("lexical parent identity is invalid")
    return values


def _require_tuple(value: object, label: str) -> tuple[_TupleItemT, ...]:
    if type(value) is not tuple:
        raise ProductionRankOnlyError(f"{label} must be a tuple")
    return cast(tuple[_TupleItemT, ...], value)
