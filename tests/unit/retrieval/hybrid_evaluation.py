"""Rights-safe deterministic evaluation helpers for frozen hybrid retrieval.

This test-only module evaluates controlled channel inputs from repository-owned
synthetic fixture data.  It validates ranking and metric invariants, not real
corpus retrieval quality, BGE effectiveness, or production-scale performance.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import Any

from academic_chatbot.retrieval.hybrid_fusion import candidate_limit, fuse_candidates
from academic_chatbot.retrieval.semantic import SemanticRetrievalHit
from academic_chatbot.retrieval.service import RetrievalHit

Identity = tuple[str, str, str, str]
_SYNTHETIC_PARENT_TEXT = "synthetic evaluation parent chunk"


@dataclass(frozen=True, slots=True)
class ChannelCandidate:
    """One controlled channel ranking input for a parent occurrence."""

    identity: Identity
    rank: int
    raw_score: float
    span_id: str | None = None


@dataclass(frozen=True, slots=True)
class QueryFixture:
    """Independent ground truth and controlled inputs for one synthetic query."""

    fixture_id: str
    category: str
    project_id: str
    query: str
    relevant: frozenset[Identity]
    lexical: tuple[ChannelCandidate, ...]
    semantic: tuple[ChannelCandidate, ...]


@dataclass(frozen=True, slots=True)
class EvaluationFixture:
    """Auditable rights-safe fixture metadata and query cases."""

    fixture_id: str
    content_provenance: str
    evaluation_scope: str
    limitation: str
    fusion_profile_id: str
    final_limit: int
    channel_input_kind: str
    queries: tuple[QueryFixture, ...]


@dataclass(frozen=True, slots=True)
class MetricValues:
    """Binary parent-chunk metrics; ``None`` marks a zero-relevance query."""

    recall_at_10: float | None
    mrr_at_10: float | None
    ndcg_at_10: float | None


@dataclass(frozen=True, slots=True)
class EvaluationHit:
    """Stable projected evidence needed for ranking and provenance checks."""

    identity: Identity
    fusion_rank: int
    fusion_score: tuple[int, int]
    lexical_rank: int | None
    semantic_rank: int | None
    semantic_span_id: str | None


@dataclass(frozen=True, slots=True)
class ModeQueryEvaluation:
    """One mode's fixed top-k ranking and its query-level metrics."""

    ranking: tuple[Identity, ...]
    hits: tuple[EvaluationHit, ...]
    metrics: MetricValues


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    """All three mode evaluations for one labelled query."""

    fixture_id: str
    category: str
    project_id: str
    relevant: frozenset[Identity]
    candidate_depth: int
    excluded_cross_project_candidates: int
    lexical: ModeQueryEvaluation
    semantic: ModeQueryEvaluation
    hybrid: ModeQueryEvaluation


@dataclass(frozen=True, slots=True)
class ModeAggregate:
    """Macro-average metrics over only queries with labelled relevance."""

    metrics: MetricValues


@dataclass(frozen=True, slots=True)
class ComplementarityDiagnostics:
    """Fixed-fixture channel overlap and hybrid recovery diagnostics."""

    lexical_relevant: frozenset[Identity]
    semantic_relevant: frozenset[Identity]
    hybrid_relevant: frozenset[Identity]
    overlap_relevant: frozenset[Identity]
    lexical_only_relevant: frozenset[Identity]
    semantic_only_relevant: frozenset[Identity]
    lexical_only_recovered: frozenset[Identity]
    semantic_only_recovered: frozenset[Identity]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Deterministic result for the complete fixed fixture."""

    fusion_profile_id: str
    final_limit: int
    queries: tuple[QueryEvaluation, ...]
    lexical: ModeAggregate
    semantic: ModeAggregate
    hybrid: ModeAggregate
    diagnostics: ComplementarityDiagnostics

    def query(self, fixture_id: str) -> QueryEvaluation:
        """Return one audited query result by its stable fixture identifier."""

        for result in self.queries:
            if result.fixture_id == fixture_id:
                return result
        raise KeyError(fixture_id)


def load_fixture(path: Path) -> EvaluationFixture:
    """Load the repository-owned JSON fixture without external data access."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = tuple(_query_from_json(item) for item in _required_list(payload, "queries"))
    fixture = EvaluationFixture(
        fixture_id=_required_text(payload, "fixture_id"),
        content_provenance=_required_text(payload, "content_provenance"),
        evaluation_scope=_required_text(payload, "evaluation_scope"),
        limitation=_required_text(payload, "limitation"),
        fusion_profile_id=_required_text(payload, "fusion_profile_id"),
        final_limit=_required_positive_int(payload, "final_limit"),
        channel_input_kind=_required_text(payload, "channel_input_kind"),
        queries=queries,
    )
    if fixture.fusion_profile_id != "rrf-v1":
        raise ValueError("evaluation fixture must validate the frozen rrf-v1 profile")
    if len({query.fixture_id for query in queries}) != len(queries):
        raise ValueError("evaluation fixture query identifiers must be unique")
    return fixture


def evaluate_fixture(
    fixture: EvaluationFixture, *, shuffle_seed: int | None = None
) -> EvaluationReport:
    """Evaluate all modes with the accepted frozen Task 2 fusion implementation."""

    randomizer = random.Random(shuffle_seed) if shuffle_seed is not None else None
    evaluations = tuple(
        _evaluate_query(query, fixture.final_limit, randomizer) for query in fixture.queries
    )
    return EvaluationReport(
        fusion_profile_id=fixture.fusion_profile_id,
        final_limit=fixture.final_limit,
        queries=evaluations,
        lexical=ModeAggregate(_aggregate_metrics(result.lexical.metrics for result in evaluations)),
        semantic=ModeAggregate(
            _aggregate_metrics(result.semantic.metrics for result in evaluations)
        ),
        hybrid=ModeAggregate(_aggregate_metrics(result.hybrid.metrics for result in evaluations)),
        diagnostics=_diagnostics(evaluations),
    )


def metrics_for_ranking(
    ranking: tuple[Identity, ...], relevant: frozenset[Identity] | set[Identity], *, limit: int
) -> MetricValues:
    """Return binary Recall, MRR, and nDCG for one labelled top-k ranking.

    Zero-relevance queries are deliberately excluded from aggregate metrics and
    therefore return ``None`` for each metric instead of dividing by zero.
    """

    if not relevant:
        return MetricValues(None, None, None)
    top = ranking[:limit]
    recovered = sum(identity in relevant for identity in top)
    first_relevant = next(
        (rank for rank, identity in enumerate(top, start=1) if identity in relevant), None
    )
    dcg = sum(
        1.0 / log2(rank + 1) for rank, identity in enumerate(top, start=1) if identity in relevant
    )
    ideal = sum(1.0 / log2(rank + 1) for rank in range(1, min(limit, len(relevant)) + 1))
    return MetricValues(
        recall_at_10=recovered / len(relevant),
        mrr_at_10=0.0 if first_relevant is None else 1.0 / first_relevant,
        ndcg_at_10=dcg / ideal,
    )


def _evaluate_query(
    query: QueryFixture, final_limit: int, randomizer: random.Random | None
) -> QueryEvaluation:
    depth = candidate_limit(final_limit)
    lexical, lexical_excluded = _eligible_candidates(query.lexical, query.project_id, depth)
    semantic, semantic_excluded = _eligible_candidates(query.semantic, query.project_id, depth)
    lexical = _maybe_shuffle(lexical, randomizer)
    semantic = _maybe_shuffle(semantic, randomizer)
    return QueryEvaluation(
        fixture_id=query.fixture_id,
        category=query.category,
        project_id=query.project_id,
        relevant=query.relevant,
        candidate_depth=depth,
        excluded_cross_project_candidates=lexical_excluded + semantic_excluded,
        lexical=_evaluate_mode(lexical, (), query.relevant, final_limit),
        semantic=_evaluate_mode((), semantic, query.relevant, final_limit),
        hybrid=_evaluate_mode(lexical, semantic, query.relevant, final_limit),
    )


def _evaluate_mode(
    lexical: tuple[ChannelCandidate, ...],
    semantic: tuple[ChannelCandidate, ...],
    relevant: frozenset[Identity],
    final_limit: int,
) -> ModeQueryEvaluation:
    candidates = fuse_candidates(
        tuple(_lexical_hit(candidate) for candidate in lexical),
        tuple(_semantic_hit(candidate) for candidate in semantic),
        final_limit=final_limit,
    )
    hits = tuple(_evaluation_hit(candidate) for candidate in candidates)
    ranking = tuple(hit.identity for hit in hits)
    return ModeQueryEvaluation(
        ranking=ranking,
        hits=hits,
        metrics=metrics_for_ranking(ranking, relevant, limit=final_limit),
    )


def _lexical_hit(candidate: ChannelCandidate) -> RetrievalHit:
    project_id, generation_id, page_id, chunk_id = candidate.identity
    return RetrievalHit.model_construct(
        project_id=project_id,
        paper_id=f"paper-{chunk_id}",
        file_version_id=f"file-{chunk_id}",
        document_generation_id=generation_id,
        page_id=page_id,
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_id=chunk_id,
        chunk_ordinal=0,
        chunk_text=_SYNTHETIC_PARENT_TEXT,
        start_offset=0,
        end_offset=len(_SYNTHETIC_PARENT_TEXT),
        rank=candidate.rank,
        raw_bm25_score=candidate.raw_score,
        anchors=(),
    )


def _semantic_hit(candidate: ChannelCandidate) -> SemanticRetrievalHit:
    project_id, generation_id, page_id, chunk_id = candidate.identity
    return SemanticRetrievalHit.model_construct(
        project_id=project_id,
        paper_id=f"paper-{chunk_id}",
        file_version_id=f"file-{chunk_id}",
        document_generation_id=generation_id,
        page_id=page_id,
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_id=chunk_id,
        embedding_span_id=candidate.span_id or f"span-{chunk_id}",
        embedding_profile_id="deterministic-fixture-profile",
        vector_generation_id="deterministic-fixture-generation",
        start_offset=0,
        end_offset=9,
        embedding_span_text="synthetic",
        rank=candidate.rank,
        raw_semantic_score=candidate.raw_score,
        anchors=(),
    )


def _evaluation_hit(candidate: Any) -> EvaluationHit:
    identity = candidate.identity
    semantic = candidate.semantic_contribution
    return EvaluationHit(
        identity=(
            identity.project_id,
            identity.document_generation_id,
            identity.page_id,
            identity.chunk_id,
        ),
        fusion_rank=candidate.trace.fusion_rank,
        fusion_score=(
            candidate.trace.fusion_score.numerator,
            candidate.trace.fusion_score.denominator,
        ),
        lexical_rank=candidate.trace.lexical_rank,
        semantic_rank=candidate.trace.semantic_rank,
        semantic_span_id=None if semantic is None else semantic.semantic_hit.embedding_span_id,
    )


def _eligible_candidates(
    candidates: tuple[ChannelCandidate, ...], project_id: str, depth: int
) -> tuple[tuple[ChannelCandidate, ...], int]:
    in_project = tuple(candidate for candidate in candidates if candidate.identity[0] == project_id)
    return tuple(candidate for candidate in in_project if candidate.rank <= depth), len(
        candidates
    ) - len(in_project)


def _maybe_shuffle(
    candidates: tuple[ChannelCandidate, ...], randomizer: random.Random | None
) -> tuple[ChannelCandidate, ...]:
    if randomizer is None:
        return candidates
    shuffled = list(candidates)
    randomizer.shuffle(shuffled)
    return tuple(shuffled)


def _aggregate_metrics(metrics: Any) -> MetricValues:
    values = tuple(metric for metric in metrics if metric.recall_at_10 is not None)
    if not values:
        return MetricValues(None, None, None)
    return MetricValues(
        recall_at_10=sum(
            metric.recall_at_10 for metric in values if metric.recall_at_10 is not None
        )
        / len(values),
        mrr_at_10=sum(metric.mrr_at_10 for metric in values if metric.mrr_at_10 is not None)
        / len(values),
        ndcg_at_10=sum(metric.ndcg_at_10 for metric in values if metric.ndcg_at_10 is not None)
        / len(values),
    )


def _diagnostics(results: tuple[QueryEvaluation, ...]) -> ComplementarityDiagnostics:
    lexical = _relevant_hits(results, "lexical")
    semantic = _relevant_hits(results, "semantic")
    hybrid = _relevant_hits(results, "hybrid")
    lexical_only = lexical - semantic
    semantic_only = semantic - lexical
    return ComplementarityDiagnostics(
        lexical_relevant=frozenset(lexical),
        semantic_relevant=frozenset(semantic),
        hybrid_relevant=frozenset(hybrid),
        overlap_relevant=frozenset(lexical & semantic),
        lexical_only_relevant=frozenset(lexical_only),
        semantic_only_relevant=frozenset(semantic_only),
        lexical_only_recovered=frozenset(lexical_only & hybrid),
        semantic_only_recovered=frozenset(semantic_only & hybrid),
    )


def _relevant_hits(results: tuple[QueryEvaluation, ...], mode: str) -> set[Identity]:
    recovered: set[Identity] = set()
    for result in results:
        ranking = getattr(result, mode).ranking
        recovered.update(identity for identity in ranking if identity in result.relevant)
    return recovered


def _query_from_json(value: object) -> QueryFixture:
    payload = _mapping(value)
    fixture_id = _required_text(payload, "id")
    project_id = _required_text(payload, "project_id")
    lexical = _channel_from_json(
        _required_list(payload, "lexical"), project_id, fixture_id, "lexical"
    )
    semantic = _channel_from_json(
        _required_list(payload, "semantic"), project_id, fixture_id, "semantic"
    )
    generated = payload.get("generated_decoys", {})
    generated_mapping = _mapping(generated)
    lexical += _generated_decoys(
        project_id,
        fixture_id,
        "lexical",
        _optional_nonnegative_int(generated_mapping, "lexical_count"),
    )
    semantic += _generated_decoys(
        project_id,
        fixture_id,
        "semantic",
        _optional_nonnegative_int(generated_mapping, "semantic_count"),
    )
    return QueryFixture(
        fixture_id=fixture_id,
        category=_required_text(payload, "category"),
        project_id=project_id,
        query=_required_text(payload, "query"),
        relevant=frozenset(_identity(item) for item in _required_list(payload, "relevant")),
        lexical=lexical,
        semantic=semantic,
    )


def _channel_from_json(
    values: list[object], project_id: str, fixture_id: str, channel: str
) -> tuple[ChannelCandidate, ...]:
    candidates: list[ChannelCandidate] = []
    for value in values:
        payload = _mapping(value)
        candidates.append(
            ChannelCandidate(
                identity=_identity(payload.get("identity")),
                rank=_required_positive_int(payload, "rank"),
                raw_score=float(payload["raw_score"]),
                span_id=None if channel == "lexical" else _required_text(payload, "span_id"),
            )
        )
    if len({(candidate.identity, candidate.span_id) for candidate in candidates}) != len(
        candidates
    ):
        raise ValueError(f"{fixture_id} has duplicate {channel} candidate inputs")
    return tuple(candidates)


def _generated_decoys(
    project_id: str, fixture_id: str, channel: str, count: int
) -> tuple[ChannelCandidate, ...]:
    return tuple(
        ChannelCandidate(
            identity=(
                project_id,
                f"generation-{fixture_id}-{channel}-decoy-{rank}",
                f"page-{fixture_id}-{channel}-decoy-{rank}",
                f"chunk-{fixture_id}-{channel}-decoy-{rank}",
            ),
            rank=rank,
            raw_score=1.0 / (rank + 1),
            span_id=None if channel == "lexical" else f"span-{fixture_id}-{channel}-decoy-{rank}",
        )
        for rank in range(1, count + 1)
    )


def _identity(value: object) -> Identity:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError("fixture identity must contain four non-empty string fields")
    return value[0], value[1], value[2], value[3]


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("fixture value must be an object")
    return value


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"fixture {key} must be a non-empty string")
    return value


def _required_list(payload: dict[str, Any], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"fixture {key} must be a list")
    return value


def _required_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"fixture {key} must be a positive integer")
    return value


def _optional_nonnegative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    if type(value) is not int or value < 0:
        raise ValueError(f"fixture {key} must be a non-negative integer")
    return value
