"""Bounded deterministic multi-query lexical retrieval."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Literal

from academic_chatbot.db.connection import DatabasePathError, open_read_only_connection
from academic_chatbot.domain.library import Project
from academic_chatbot.retrieval.fts import (
    RetrievalQueryError,
    search_active_chunks,
)
from academic_chatbot.retrieval.query_plan import LexicalQueryPlan, plan_natural_language_query
from academic_chatbot.retrieval.service import (
    RetrievalHit,
    RetrievalIntegrityError,
    RetrievalResults,
    RetrievalStorageError,
    _hit_from_row,
)
from academic_chatbot.storage.paths import ProjectPaths

LEXICAL_MULTI_QUERY_POLICY_ID = "lexical-multi-query-v1"
LEXICAL_MULTI_QUERY_FUSION_POLICY_ID = "lexical-multi-query-fusion-v1"
LEXICAL_MULTI_QUERY_FUSION_K = 60

_MAX_QUERY_BYTES = 4096
_MAX_ACTIVE_FAMILIES = 2
_MAX_EXPANDED_TERMS = 64
_MAX_BRANCHES = 16
_MAX_EXPRESSION_BYTES = 8192
_MAX_FINAL_LIMIT = 100
_MAX_BRANCH_DEPTH = 100
_MIN_BRANCH_DEPTH = 20
_MAX_TOTAL_CANDIDATES = 1600
_ASCII_TERM = re.compile(r"^[a-z][a-z0-9]*$")
_VOWELS = frozenset("aeiou")


@dataclass(frozen=True, slots=True)
class LexicalQueryBranch:
    """One independently executed lexical expression."""

    branch_id: str
    kind: Literal["base", "morphology", "intent"]
    terms: tuple[str, ...]
    expression: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LexicalMultiQueryPlan:
    """Immutable branch plan for ``lexical-multi-query-v1``."""

    policy_id: str
    fusion_policy_id: str
    base_plan: LexicalQueryPlan
    active_family_ids: tuple[str, ...]
    branches: tuple[LexicalQueryBranch, ...]


@dataclass(frozen=True, slots=True)
class LexicalBranchResults:
    """Ordered hits from one independent lexical branch."""

    branch: LexicalQueryBranch
    hits: tuple[RetrievalHit, ...]


@dataclass(frozen=True, slots=True)
class _IntentFamily:
    family_id: str
    triggers: tuple[str, ...]
    generated_terms: tuple[str, ...]


_INTENT_FAMILIES = (
    _IntentFamily(
        family_id="research-objective-v1",
        triggers=(
            "aim",
            "objective",
            "purpose",
            "goal",
            "investigate",
            "examine",
            "evaluate",
            "assess",
            "determine",
        ),
        generated_terms=(
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
    _IntentFamily(
        family_id="research-method-v1",
        triggers=(
            "method",
            "methods",
            "approach",
            "procedure",
            "protocol",
            "design",
            "analysis",
        ),
        generated_terms=(
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
    _IntentFamily(
        family_id="research-result-v1",
        triggers=(
            "result",
            "results",
            "finding",
            "findings",
            "outcome",
            "effect",
            "conclusion",
        ),
        generated_terms=(
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
)


def plan_lexical_multi_query(query: str) -> LexicalMultiQueryPlan:
    """Build a bounded, deterministic multi-branch lexical query plan."""

    base_plan = plan_natural_language_query(query)
    if len(query.encode("utf-8")) > _MAX_QUERY_BYTES:
        raise RetrievalQueryError("lexical query exceeds resource limits")

    active_families = tuple(
        family
        for family in _INTENT_FAMILIES
        if any(term in family.triggers for term in base_plan.selected_terms)
    )
    if len(active_families) > _MAX_ACTIVE_FAMILIES:
        raise RetrievalQueryError("lexical multi-query exceeds resource limits")

    morphology_sources = tuple(
        (index, term, _morphology_variants(term))
        for index, term in enumerate(base_plan.selected_terms)
        if _ASCII_TERM.fullmatch(term)
    )
    expanded_term_count = (
        len(base_plan.selected_terms)
        + sum(len(variants) for _, _, variants in morphology_sources)
        + sum(len(family.generated_terms) for family in active_families)
    )
    if expanded_term_count > _MAX_EXPANDED_TERMS:
        raise RetrievalQueryError("lexical multi-query exceeds resource limits")

    pending: list[LexicalQueryBranch] = [
        LexicalQueryBranch(
            branch_id="base-000",
            kind="base",
            terms=base_plan.selected_terms,
            expression=base_plan.expression,
        )
    ]
    for source_index, _, variants in morphology_sources:
        for variant_index, variant in enumerate(variants):
            terms = _deduplicate((*base_plan.selected_terms, variant))
            pending.append(
                _branch(
                    branch_id=f"morphology-{source_index:03d}-{variant_index:03d}",
                    kind="morphology",
                    terms=terms,
                )
            )
    for family_index, family in enumerate(active_families):
        terms = _deduplicate((*base_plan.selected_terms, *family.generated_terms))
        pending.append(
            _branch(
                branch_id=f"intent-{family_index:03d}-{family.family_id}",
                kind="intent",
                terms=terms,
            )
        )

    branches = _deduplicate_branches(pending)
    if len(branches) > _MAX_BRANCHES:
        raise RetrievalQueryError("lexical multi-query exceeds resource limits")
    return LexicalMultiQueryPlan(
        policy_id=LEXICAL_MULTI_QUERY_POLICY_ID,
        fusion_policy_id=LEXICAL_MULTI_QUERY_FUSION_POLICY_ID,
        base_plan=base_plan,
        active_family_ids=tuple(family.family_id for family in active_families),
        branches=tuple(branches),
    )


def branch_candidate_depth(final_limit: int) -> int:
    """Return the frozen per-branch retrieval depth."""

    _validate_final_limit(final_limit)
    return min(_MAX_BRANCH_DEPTH, max(_MIN_BRANCH_DEPTH, final_limit * 5))


class LexicalMultiQueryService:
    """Search active project chunks through independent lexical branches."""

    def __init__(self, *, data_root: Path) -> None:
        self._data_root = data_root.resolve(strict=False)

    def search(self, project: Project, query: str, limit: int = 10) -> RetrievalResults:
        """Return one deterministic merged lexical result channel."""

        branch_depth = branch_candidate_depth(limit)
        plan = plan_lexical_multi_query(query)
        paths = ProjectPaths.create(self._data_root, project_id=project.project_id)
        try:
            connection = open_read_only_connection(
                paths.database_path, data_root=self._data_root
            )
        except DatabasePathError as error:
            raise RetrievalStorageError(str(error)) from error

        branch_results: list[LexicalBranchResults] = []
        total_candidates = 0
        try:
            for branch in plan.branches:
                rows = search_active_chunks(
                    connection,
                    project_id=project.project_id,
                    match_expression=branch.expression,
                    limit=branch_depth,
                )
                total_candidates += len(rows)
                if total_candidates > _MAX_TOTAL_CANDIDATES:
                    raise RetrievalQueryError("lexical multi-query exceeds resource limits")
                hits = tuple(
                    _hit_from_row(connection, row=row, rank=index)
                    for index, row in enumerate(rows, start=1)
                )
                branch_results.append(LexicalBranchResults(branch=branch, hits=hits))
        except sqlite3.DatabaseError as error:
            raise RetrievalStorageError("project database could not be searched") from error
        finally:
            connection.close()

        return merge_lexical_results(
            project_id=project.project_id,
            query=query,
            branch_results=tuple(branch_results),
            final_limit=limit,
        )


def merge_lexical_results(
    *,
    project_id: str,
    query: str,
    branch_results: Sequence[LexicalBranchResults],
    final_limit: int,
) -> RetrievalResults:
    """Merge independent branch results by exact rank-only reciprocal rank."""

    _validate_final_limit(final_limit)
    if len(branch_results) > _MAX_BRANCHES:
        raise RetrievalQueryError("lexical multi-query exceeds resource limits")

    grouped: dict[tuple[str, str, str, str], list[tuple[int, int, RetrievalHit]]] = {}
    branch_ids: set[str] = set()
    total_candidates = 0
    for branch_order, result in enumerate(branch_results):
        if result.branch.branch_id in branch_ids:
            raise RetrievalIntegrityError("duplicate lexical branch identity")
        branch_ids.add(result.branch.branch_id)
        branch_seen: set[tuple[str, str, str, str]] = set()
        total_candidates += len(result.hits)
        if total_candidates > _MAX_TOTAL_CANDIDATES:
            raise RetrievalQueryError("lexical multi-query exceeds resource limits")
        for within_rank, hit in enumerate(result.hits, start=1):
            if not isinstance(hit, RetrievalHit):
                raise RetrievalIntegrityError("lexical branch result is malformed")
            if hit.project_id != project_id:
                raise RetrievalIntegrityError("lexical branch result project does not match")
            identity = _candidate_identity(hit)
            if identity in branch_seen:
                raise RetrievalIntegrityError("duplicate lexical candidate in one branch")
            branch_seen.add(identity)
            group = grouped.setdefault(identity, [])
            if group and _evidence_facts(group[0][2]) != _evidence_facts(hit):
                raise RetrievalIntegrityError("contradictory lexical evidence for one candidate")
            group.append((branch_order, within_rank, hit))

    if len(grouped) > _MAX_TOTAL_CANDIDATES:
        raise RetrievalQueryError("lexical multi-query exceeds resource limits")

    ordered = sorted(grouped.items(), key=_candidate_order_key)
    hits = tuple(
        _representative_hit(identity, contributions).model_copy(update={"rank": rank})
        for rank, (identity, contributions) in enumerate(ordered[:final_limit], start=1)
    )
    return RetrievalResults(project_id=project_id, query=query, hits=hits)


def _branch(
    *, branch_id: str, kind: Literal["base", "morphology", "intent"], terms: tuple[str, ...]
) -> LexicalQueryBranch:
    branch_plan = plan_natural_language_query(" ".join(terms))
    expression = branch_plan.expression
    if len(expression.encode("utf-8")) > _MAX_EXPRESSION_BYTES:
        raise RetrievalQueryError("lexical multi-query exceeds resource limits")
    return LexicalQueryBranch(
        branch_id=branch_id,
        kind=kind,
        terms=branch_plan.selected_terms,
        expression=expression,
    )


def _deduplicate(terms: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(terms))


def _deduplicate_branches(branches: Sequence[LexicalQueryBranch]) -> list[LexicalQueryBranch]:
    by_expression: dict[str, int] = {}
    unique: list[LexicalQueryBranch] = []
    for branch in branches:
        existing_index = by_expression.get(branch.expression)
        if existing_index is None:
            by_expression[branch.expression] = len(unique)
            unique.append(branch)
            continue
        existing = unique[existing_index]
        unique[existing_index] = replace(
            existing, aliases=(*existing.aliases, branch.branch_id)
        )
    return unique


def _morphology_variants(term: str) -> tuple[str, ...]:
    if _ASCII_TERM.fullmatch(term) is None:
        return ()
    plural = _plural_variant(term)
    past = _past_variant(term)
    gerund = _gerund_variant(term)
    variants = _deduplicate((plural, past, gerund))
    for variant in variants:
        if len(variant) > 64 or len(variant.encode("utf-8")) > 128:
            raise RetrievalQueryError("lexical multi-query exceeds resource limits")
    return variants


def _plural_variant(term: str) -> str:
    if _ends_with_consonant_y(term):
        return f"{term[:-1]}ies"
    if term.endswith(("s", "x", "z", "ch", "sh")):
        return f"{term}es"
    return f"{term}s"


def _past_variant(term: str) -> str:
    if term.endswith("e"):
        return f"{term}d"
    if _ends_with_consonant_y(term):
        return f"{term[:-1]}ied"
    return f"{term}ed"


def _gerund_variant(term: str) -> str:
    stem = term[:-1] if term.endswith("e") and not term.endswith("ee") else term
    return f"{stem}ing"


def _ends_with_consonant_y(term: str) -> bool:
    return (
        len(term) >= 2
        and term[-1] == "y"
        and term[-2] in "abcdefghijklmnopqrstuvwxyz"
        and term[-2] not in _VOWELS
    )


def _validate_final_limit(final_limit: int) -> None:
    if type(final_limit) is not int or not 1 <= final_limit <= _MAX_FINAL_LIMIT:
        raise RetrievalQueryError("limit must be a positive integer no greater than 100")


def _candidate_identity(hit: RetrievalHit) -> tuple[str, str, str, str]:
    return (
        hit.project_id,
        hit.document_generation_id,
        hit.page_id,
        hit.chunk_id,
    )


def _evidence_facts(hit: RetrievalHit) -> tuple[object, ...]:
    return (
        hit.paper_id,
        hit.file_version_id,
        hit.document_generation_id,
        hit.page_id,
        hit.physical_page_index,
        hit.display_page_number,
        hit.printed_page_label,
        hit.chunk_id,
        hit.chunk_ordinal,
        hit.chunk_text,
        hit.start_offset,
        hit.end_offset,
        hit.anchors,
    )


def _candidate_order_key(
    item: tuple[
        tuple[str, str, str, str],
        list[tuple[int, int, RetrievalHit]],
    ],
) -> tuple[Fraction, int, int, int, tuple[str, str, str, str]]:
    identity, contributions = item
    score = sum(
        (
            Fraction(1, LEXICAL_MULTI_QUERY_FUSION_K + within_rank)
            for _, within_rank, _ in contributions
        ),
        Fraction(0, 1),
    )
    representative_branch, _, _ = min(
        contributions,
        key=lambda contribution: (contribution[1], contribution[0], identity),
    )
    return (
        -score,
        min(within_rank for _, within_rank, _ in contributions),
        -len(contributions),
        representative_branch,
        identity,
    )


def _representative_hit(
    identity: tuple[str, str, str, str], contributions: list[tuple[int, int, RetrievalHit]]
) -> RetrievalHit:
    _, _, hit = min(
        contributions,
        key=lambda contribution: (contribution[1], contribution[0], identity),
    )
    return hit
