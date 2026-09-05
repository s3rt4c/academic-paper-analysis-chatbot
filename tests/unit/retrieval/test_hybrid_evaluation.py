"""Rights-safe deterministic evaluation contracts for frozen hybrid retrieval."""

from __future__ import annotations

from math import isclose, log2
from pathlib import Path

from tests.unit.retrieval.hybrid_evaluation import (
    evaluate_fixture,
    load_fixture,
    metrics_for_ranking,
)

_FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "hybrid_retrieval" / "evaluation.json"


def test_metric_contract_excludes_zero_relevance_queries_without_dividing_by_zero() -> None:
    """Would fail if binary metrics fabricated a score for a no-relevance query."""
    relevant = {
        ("project", "generation", "page", "chunk-a"),
        ("project", "generation", "page", "chunk-b"),
    }
    ranking = (
        ("project", "generation", "page", "chunk-a"),
        ("project", "generation", "page", "chunk-decoy"),
        ("project", "generation", "page", "chunk-b"),
    )

    metrics = metrics_for_ranking(ranking, relevant, limit=10)
    no_relevance = metrics_for_ranking(ranking, frozenset(), limit=10)

    assert metrics.recall_at_10 == 1.0
    assert metrics.mrr_at_10 == 1.0
    assert isclose(metrics.ndcg_at_10 or 0.0, (1.0 + 1.0 / log2(4)) / (1.0 + 1.0 / log2(3)))
    assert no_relevance.recall_at_10 is None
    assert no_relevance.mrr_at_10 is None
    assert no_relevance.ndcg_at_10 is None


def test_fixture_evaluation_validates_complementarity_without_private_profile_selection() -> None:
    """Would fail if the synthetic fixture stopped preserving channel-unique parent evidence."""
    fixture = load_fixture(_FIXTURE_PATH)
    report = evaluate_fixture(fixture)

    assert fixture.content_provenance.startswith("Repository-authored synthetic")
    assert fixture.evaluation_scope == "RIGHTS-SAFE DETERMINISTIC SYNTHETIC FIXTURE EVALUATION"
    assert len(fixture.queries) == 9
    assert {query.category for query in fixture.queries} == {
        "lexical-favored",
        "semantic-favored",
        "dual-channel",
        "repeated-occurrence",
        "ambiguous",
        "no-hit",
        "cross-project",
        "multi-span",
        "depth-sensitive",
    }
    assert report.fusion_profile_id == "rrf-v1"
    assert report.hybrid.metrics.recall_at_10 >= report.lexical.metrics.recall_at_10
    assert report.hybrid.metrics.recall_at_10 >= report.semantic.metrics.recall_at_10
    assert report.diagnostics.lexical_only_recovered == report.diagnostics.lexical_only_relevant
    assert report.diagnostics.semantic_only_recovered == report.diagnostics.semantic_only_relevant
    assert (
        ("evaluation-project", "generation-lexical", "page-lexical", "chunk-lexical-target")
        in report.query("lexical-diagnostic-identifier").hybrid.ranking
    )
    assert (
        ("evaluation-project", "generation-semantic", "page-semantic", "chunk-semantic-target")
        in report.query("semantic-meaning-paraphrase").hybrid.ranking
    )


def test_fixture_evaluation_keeps_occurrences_and_semantic_votes_exact() -> None:
    """Would fail if fusion merged repeated occurrences or let two spans cast two parent votes."""
    report = evaluate_fixture(load_fixture(_FIXTURE_PATH))

    repeated = report.query("repeated-occurrence-identity")
    dual = report.query("dual-channel-method-identifier")
    multi_span = report.query("multi-span-parent-vote")
    cross_project = report.query("cross-project-eligibility")
    no_hit = report.query("no-labelled-relevance")
    depth_sensitive = report.query("depth-sensitive-complementarity")

    assert len(repeated.hybrid.ranking) == 2
    assert len(set(repeated.hybrid.ranking)) == 2
    assert len(dual.hybrid.ranking) == 4
    assert dual.hybrid.hits[0].lexical_rank == 3
    assert dual.hybrid.hits[0].semantic_rank == 2
    assert len(multi_span.hybrid.ranking) == 4
    assert multi_span.hybrid.hits[0].semantic_span_id == "span-multi-primary"
    assert cross_project.excluded_cross_project_candidates == 1
    assert all(identity[0] == cross_project.project_id for identity in cross_project.hybrid.ranking)
    assert no_hit.hybrid.metrics.recall_at_10 is None
    assert no_hit.hybrid.metrics.mrr_at_10 is None
    assert no_hit.hybrid.metrics.ndcg_at_10 is None
    assert depth_sensitive.candidate_depth == 50
    assert depth_sensitive.hybrid.ranking[0][3] == "chunk-depth-target"


def test_fixture_evaluation_is_identical_after_seeded_equivalent_channel_shuffle() -> None:
    """Would fail if input order changed frozen fusion ranks, scores, metrics, or diagnostics."""
    fixture = load_fixture(_FIXTURE_PATH)

    original = evaluate_fixture(fixture)
    shuffled = evaluate_fixture(fixture, shuffle_seed=20260904)

    assert shuffled == original
