from __future__ import annotations

import pytest

from academic_chatbot.retrieval.fts import RetrievalQueryError
from academic_chatbot.retrieval.multi_query import (
    LEXICAL_MULTI_QUERY_FUSION_POLICY_ID,
    LEXICAL_MULTI_QUERY_POLICY_ID,
    LexicalBranchResults,
    LexicalMultiQueryService,
    LexicalQueryBranch,
    branch_candidate_depth,
    merge_lexical_results,
    plan_lexical_multi_query,
)
from academic_chatbot.retrieval.service import RetrievalHit, RetrievalIntegrityError


def _branch(
    branch_id: str,
    *,
    kind: str = "base",
    terms: tuple[str, ...] = ("alpha",),
) -> LexicalQueryBranch:
    return LexicalQueryBranch(
        branch_id=branch_id,
        kind=kind,
        terms=terms,
        expression=" OR ".join(f'"{term}"' for term in terms),
    )


def _hit(
    chunk_id: str,
    *,
    rank: int = 1,
    raw_bm25_score: float = -1.0,
    project_id: str = "project-1",
    chunk_text: str | None = None,
) -> RetrievalHit:
    text = chunk_text or f"content for {chunk_id}"
    return RetrievalHit.model_construct(
        project_id=project_id,
        paper_id="paper-1",
        file_version_id="file-1",
        document_generation_id="generation-1",
        page_id="page-1",
        physical_page_index=0,
        display_page_number=1,
        printed_page_label=None,
        chunk_id=chunk_id,
        chunk_ordinal=0,
        chunk_text=text,
        start_offset=0,
        end_offset=len(text),
        rank=rank,
        raw_bm25_score=raw_bm25_score,
        anchors=(),
    )


def _branch_results(
    branch: LexicalQueryBranch, *hits: RetrievalHit
) -> LexicalBranchResults:
    return LexicalBranchResults(branch=branch, hits=tuple(hits))


def test_policy_identity_and_base_branch_are_stable() -> None:
    first = plan_lexical_multi_query("Alpha")
    repeated = plan_lexical_multi_query("Alpha")

    assert first == repeated
    assert first.policy_id == LEXICAL_MULTI_QUERY_POLICY_ID
    assert first.fusion_policy_id == LEXICAL_MULTI_QUERY_FUSION_POLICY_ID
    assert first.base_plan.selected_terms == ("alpha",)
    assert first.branches[0].branch_id == "base-000"
    assert first.branches[0].kind == "base"
    assert first.branches[0].terms == ("alpha",)
    assert first.branches[0].expression == '"alpha"'


@pytest.mark.parametrize(
    ("term", "expected"),
    (
        ("study", ("studies", "studied", "studying")),
        ("watch", ("watches", "watched", "watching")),
        ("box", ("boxes", "boxed", "boxing")),
        ("buzz", ("buzzes", "buzzed", "buzzing")),
        ("fix", ("fixes", "fixed", "fixing")),
        ("use", ("uses", "used", "using")),
        ("agree", ("agrees", "agreed", "agreeing")),
        ("play", ("plays", "played", "playing")),
    ),
)
def test_morphology_variants_follow_the_frozen_suffix_rules(
    term: str, expected: tuple[str, ...]
) -> None:
    plan = plan_lexical_multi_query(term)

    morphology = tuple(
        branch.terms[-1] for branch in plan.branches if branch.kind == "morphology"
    )

    assert morphology == expected


def test_ineligible_terms_do_not_receive_morphology() -> None:
    for term in ("café", "123"):
        plan = plan_lexical_multi_query(term)

        assert plan.active_family_ids == ()
        assert tuple(branch.kind for branch in plan.branches) == ("base",)


@pytest.mark.parametrize(
    ("trigger", "family_id", "generated"),
    (
        (
            "objective",
            "research-objective-v1",
            (
                "aim",
                "objective",
                "purpose",
                "goal",
                "investigate",
                "examine",
                "evaluate",
                "assess",
            ),
        ),
        (
            "method",
            "research-method-v1",
            (
                "method",
                "approach",
                "procedure",
                "protocol",
                "design",
                "analysis",
                "technique",
                "experiment",
            ),
        ),
        (
            "result",
            "research-result-v1",
            (
                "result",
                "finding",
                "outcome",
                "effect",
                "conclusion",
                "observation",
                "association",
                "change",
            ),
        ),
    ),
)
def test_intent_registry_is_closed_ordered_and_exact(
    trigger: str, family_id: str, generated: tuple[str, ...]
) -> None:
    plan = plan_lexical_multi_query(trigger)

    assert plan.active_family_ids == (family_id,)
    intent = tuple(branch for branch in plan.branches if branch.kind == "intent")
    assert len(intent) == 1
    assert intent[0].terms == tuple(dict.fromkeys((trigger, *generated)))


def test_no_intent_family_activation_for_unregistered_academic_terms() -> None:
    plan = plan_lexical_multi_query("thermal conductivity")

    assert plan.active_family_ids == ()
    assert all(branch.kind != "intent" for branch in plan.branches)


def test_three_active_families_fail_closed() -> None:
    with pytest.raises(RetrievalQueryError, match="resource limits"):
        plan_lexical_multi_query("aim method result")


def test_branch_order_and_deduplication_are_deterministic() -> None:
    plan = plan_lexical_multi_query("objective")
    kinds = tuple(branch.kind for branch in plan.branches)

    assert kinds[0] == "base"
    assert kinds[1:4] == ("morphology", "morphology", "morphology")
    assert kinds[-1] == "intent"
    assert len({branch.expression for branch in plan.branches}) == len(plan.branches)

    duplicate_plan = plan_lexical_multi_query("study studies")
    assert duplicate_plan.active_family_ids == ()
    assert len(duplicate_plan.branches) < 1 + 2 * 3
    assert any(branch.aliases for branch in duplicate_plan.branches)


def test_user_fts_syntax_is_always_literalized() -> None:
    plan = plan_lexical_multi_query('study OR NOT NEAR * "quoted" (term)')

    assert plan.base_plan.selected_terms == ("study", "not", "near", "quoted", "term")
    for branch in plan.branches:
        assert "*" not in branch.expression
        assert "NOT" not in branch.expression
        assert '"not"' in branch.expression
        assert '"near"' in branch.expression


@pytest.mark.parametrize(
    "query",
    (
        "x " * 2049,
        "x " * 129,
        ",".join("term" for _ in range(257)),
        "a" * 65,
        " ".join(f"term{index}" for index in range(33)),
        " ".join(("alpha", "beta", "gamma", "delta", "epsilon", "zeta")),
    ),
)
def test_planner_rejects_each_reachable_resource_overflow(query: str) -> None:
    with pytest.raises(RetrievalQueryError, match="resource limits"):
        plan_lexical_multi_query(query)


def test_branch_candidate_depth_is_frozen_and_bounded() -> None:
    assert branch_candidate_depth(1) == 20
    assert branch_candidate_depth(20) == 100
    assert branch_candidate_depth(100) == 100

    with pytest.raises(RetrievalQueryError, match="limit"):
        branch_candidate_depth(0)
    with pytest.raises(RetrievalQueryError, match="limit"):
        branch_candidate_depth(101)


def test_merge_uses_rank_only_scores_and_merges_duplicate_parent_identity() -> None:
    base = _branch("base-000")
    morphology = _branch("morphology-000-000", kind="morphology", terms=("alpha", "alphas"))

    candidate_a = _hit("chunk-a", rank=1, raw_bm25_score=100.0)
    candidate_b = _hit("chunk-b", rank=2, raw_bm25_score=-100.0)
    candidate_a_repeat = _hit("chunk-a", rank=1, raw_bm25_score=-500.0)
    candidate_c = _hit("chunk-c", rank=2, raw_bm25_score=-900.0)

    result = merge_lexical_results(
        project_id="project-1",
        query="alpha",
        branch_results=(
            _branch_results(base, candidate_a, candidate_b),
            _branch_results(morphology, candidate_a_repeat, candidate_c),
        ),
        final_limit=10,
    )

    assert [hit.chunk_id for hit in result.hits] == ["chunk-a", "chunk-b", "chunk-c"]
    assert result.hits[0].raw_bm25_score == 100.0
    assert [hit.rank for hit in result.hits] == [1, 2, 3]


def test_merge_tie_break_is_deterministic_after_exact_rank_score() -> None:
    first = _branch("base-000")
    second = _branch("morphology-000-000", kind="morphology", terms=("alpha", "alphas"))

    result = merge_lexical_results(
        project_id="project-1",
        query="alpha",
        branch_results=(
            _branch_results(first, _hit("chunk-a", raw_bm25_score=999.0)),
            _branch_results(second, _hit("chunk-b", raw_bm25_score=-999.0)),
        ),
        final_limit=10,
    )

    assert [hit.chunk_id for hit in result.hits] == ["chunk-a", "chunk-b"]


def test_merge_rejects_cross_project_candidates() -> None:
    with pytest.raises(RetrievalIntegrityError, match="project"):
        merge_lexical_results(
            project_id="project-1",
            query="alpha",
            branch_results=(
                _branch_results(_branch("base-000"), _hit("chunk-a", project_id="project-2")),
            ),
            final_limit=10,
        )


def test_merge_rejects_contradictory_duplicate_evidence() -> None:
    first = _hit("chunk-a", chunk_text="first")
    contradictory = _hit("chunk-a", chunk_text="second")

    with pytest.raises(RetrievalIntegrityError, match="contradictory"):
        merge_lexical_results(
            project_id="project-1",
            query="alpha",
            branch_results=(
                _branch_results(_branch("base-000"), first),
                _branch_results(
                    _branch("morphology-000-000", kind="morphology"), contradictory
                ),
            ),
            final_limit=10,
        )


def test_merge_rejects_over_limit_final_results_and_candidates() -> None:
    with pytest.raises(RetrievalQueryError, match="limit"):
        merge_lexical_results(
            project_id="project-1",
            query="alpha",
            branch_results=(_branch_results(_branch("base-000")),),
            final_limit=101,
        )

    hits = tuple(_hit(f"chunk-{index}") for index in range(1601))
    with pytest.raises(RetrievalQueryError, match="resource limits"):
        merge_lexical_results(
            project_id="project-1",
            query="alpha",
            branch_results=(_branch_results(_branch("base-000"), *hits),),
            final_limit=100,
        )


def test_service_is_an_independent_multi_query_boundary(tmp_path) -> None:
    service = LexicalMultiQueryService(data_root=tmp_path)

    assert service.__class__.__name__ == "LexicalMultiQueryService"

    with pytest.raises(RetrievalQueryError, match="limit"):
        service.search(
            project=type("ProjectLike", (), {"project_id": "project-1"})(),
            query="alpha",
            limit=101,
        )


def test_branch_result_dataclass_remains_immutable() -> None:
    branch = _branch("base-000")
    result = _branch_results(branch, _hit("chunk-a"))

    with pytest.raises((AttributeError, TypeError, ValueError)):
        result.hits = ()  # type: ignore[misc]
