"""Deterministic additive semantic positional acquisition policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from academic_chatbot.retrieval.semantic import SemanticRetrievalHit


SEMANTIC_POSITIONAL_ACQUISITION_POLICY_ID = "semantic-positional-acquisition-v1"
SUPPORTED_POSITIONAL_CONCERN_ID = "stated-study-objective-v1"
MAX_POSITIONAL_HITS_EXAMINED = 100
MAX_POSITIONAL_AUXILIARY_HITS = 32
MAX_POSITIONED_DOCUMENT_GENERATIONS = 100
MAX_POSITIONED_PAGES_PER_DOCUMENT = 4096
MAX_POSITIONED_PAGE_ROWS = 16384


class SemanticPositionalAcquisitionError(ValueError):
    """Raised when positional acquisition input cannot be proven safe."""


class SemanticPositionalPolicyError(SemanticPositionalAcquisitionError):
    """Raised when an explicit positional concern is unsupported."""


@dataclass(frozen=True, slots=True)
class PositionalAcquisitionSelection:
    """Explicitly activate one supported positional concern."""

    concern_id: str


@dataclass(frozen=True, slots=True)
class PositionedPage:
    """Read-only canonical page-length metadata for one document generation."""

    document_generation_id: str
    page_id: str
    physical_page_index: int
    canonical_text_length: int


@dataclass(frozen=True, slots=True)
class SemanticPositionalAcquisition:
    """Baseline-preserving semantic candidates plus an auxiliary position lane."""

    project_id: str
    query: str
    embedding_profile_id: str
    vector_generation_id: str
    policy_id: str
    concern_id: str
    requested_limit: int
    examined_hits: tuple[SemanticRetrievalHit, ...]
    baseline_hits: tuple[SemanticRetrievalHit, ...]
    positional_hits: tuple[SemanticRetrievalHit, ...]
    ordered_hits: tuple[SemanticRetrievalHit, ...]


def build_positional_acquisition(
    *,
    project_id: str,
    query: str,
    embedding_profile_id: str,
    vector_generation_id: str,
    examined_hits: tuple[SemanticRetrievalHit, ...],
    pages: tuple[PositionedPage, ...],
    requested_limit: int,
    selection: PositionalAcquisitionSelection,
) -> SemanticPositionalAcquisition:
    """Partition a bounded semantic universe without changing semantic ranks."""

    validate_positional_selection(selection)
    if type(requested_limit) is not int or not 1 <= requested_limit <= MAX_POSITIONAL_HITS_EXAMINED:
        raise SemanticPositionalAcquisitionError("requested limit is outside positional bounds")

    examined = tuple(examined_hits)
    if len(examined) > MAX_POSITIONAL_HITS_EXAMINED:
        raise SemanticPositionalAcquisitionError("examined semantic hits exceed the bound")

    page_index = _index_pages(tuple(pages))
    seen: dict[tuple[object, ...], SemanticRetrievalHit] = {}
    for expected_rank, hit in enumerate(examined, start=1):
        _validate_hit_rank(hit, expected_rank)
        identity = _hit_identity(hit)
        previous = seen.get(identity)
        if previous is not None:
            if _duplicate_facts(previous) != _duplicate_facts(hit):
                raise SemanticPositionalAcquisitionError(
                    "duplicate semantic occurrence has contradictory facts"
                )
            continue
        seen[identity] = hit
        _validate_hit_range(hit, page_index)

    baseline: list[SemanticRetrievalHit] = []
    baseline_identity: set[tuple[object, ...]] = set()
    for hit in examined[:requested_limit]:
        identity = _hit_identity(hit)
        if identity not in baseline_identity:
            baseline.append(hit)
            baseline_identity.add(identity)
    positional: list[SemanticRetrievalHit] = []
    for hit in examined[requested_limit:]:
        identity = _hit_identity(hit)
        if identity in baseline_identity or any(
            _hit_identity(existing) == identity for existing in positional
        ):
            continue
        if _is_eligible(hit, page_index):
            positional.append(hit)
            if len(positional) == MAX_POSITIONAL_AUXILIARY_HITS:
                break

    return SemanticPositionalAcquisition(
        project_id=project_id,
        query=query,
        embedding_profile_id=embedding_profile_id,
        vector_generation_id=vector_generation_id,
        policy_id=SEMANTIC_POSITIONAL_ACQUISITION_POLICY_ID,
        concern_id=selection.concern_id,
        requested_limit=requested_limit,
        examined_hits=examined,
        baseline_hits=tuple(baseline),
        positional_hits=tuple(positional),
        ordered_hits=tuple(baseline) + tuple(positional),
    )


def validate_positional_selection(selection: PositionalAcquisitionSelection) -> None:
    """Validate the one explicit positional concern before retrieval work."""

    _validate_selection(selection)


def _validate_selection(selection: PositionalAcquisitionSelection) -> None:
    if not isinstance(selection, PositionalAcquisitionSelection):
        raise SemanticPositionalPolicyError("positional selection is invalid")
    if selection.concern_id != SUPPORTED_POSITIONAL_CONCERN_ID:
        raise SemanticPositionalPolicyError("positional concern is unsupported")


def _index_pages(
    pages: tuple[PositionedPage, ...],
) -> dict[tuple[str, str], tuple[int, int, int, int]]:
    if len(pages) > MAX_POSITIONED_PAGE_ROWS:
        raise SemanticPositionalAcquisitionError("positioned page metadata exceeds the bound")

    grouped: dict[str, list[PositionedPage]] = {}
    seen_page_ids: set[tuple[str, str]] = set()
    for page in pages:
        if not isinstance(page, PositionedPage):
            raise SemanticPositionalAcquisitionError("positioned page metadata is invalid")
        if (
            not page.document_generation_id
            or not page.page_id
            or type(page.physical_page_index) is not int
            or page.physical_page_index < 0
            or type(page.canonical_text_length) is not int
            or page.canonical_text_length <= 0
        ):
            raise SemanticPositionalAcquisitionError("positioned page metadata is invalid")
        page_key = (page.document_generation_id, page.page_id)
        if page_key in seen_page_ids:
            raise SemanticPositionalAcquisitionError("positioned page metadata is duplicated")
        seen_page_ids.add(page_key)
        grouped.setdefault(page.document_generation_id, []).append(page)

    if len(grouped) > MAX_POSITIONED_DOCUMENT_GENERATIONS:
        raise SemanticPositionalAcquisitionError(
            "positioned document generation metadata exceeds the bound"
        )

    index: dict[tuple[str, str], tuple[int, int, int, int]] = {}
    for document_generation_id, document_pages in grouped.items():
        if len(document_pages) > MAX_POSITIONED_PAGES_PER_DOCUMENT:
            raise SemanticPositionalAcquisitionError(
                "positioned document page metadata exceeds the bound"
            )
        ordered = sorted(document_pages, key=lambda page: (page.physical_page_index, page.page_id))
        if any(
            page.physical_page_index != expected_index
            for expected_index, page in enumerate(ordered)
        ):
            raise SemanticPositionalAcquisitionError(
                "positioned page order is not contiguous"
            )
        document_length = sum(page.canonical_text_length for page in ordered)
        document_length += max(len(ordered) - 1, 0)
        page_base = 0
        for page in ordered:
            index[(document_generation_id, page.page_id)] = (
                page_base,
                page.canonical_text_length,
                document_length,
                page.physical_page_index,
            )
            page_base += page.canonical_text_length + 1
    return index


def _validate_hit_rank(hit: Any, expected_rank: int) -> None:
    rank = getattr(hit, "rank", None)
    if (
        type(rank) is not int
        or rank != expected_rank
        or not 1 <= rank <= MAX_POSITIONAL_HITS_EXAMINED
    ):
        raise SemanticPositionalAcquisitionError("semantic hit ranks are invalid")


def _validate_hit_range(
    hit: SemanticRetrievalHit,
    page_index: dict[tuple[str, str], tuple[int, int, int, int]],
) -> None:
    key = (hit.document_generation_id, hit.page_id)
    metadata = page_index.get(key)
    if metadata is None:
        raise SemanticPositionalAcquisitionError("semantic hit page metadata is missing")
    _, page_length, _, physical_page_index = metadata
    if (
        hit.physical_page_index != physical_page_index
        or
        type(hit.start_offset) is not int
        or type(hit.end_offset) is not int
        or not 0 <= hit.start_offset < hit.end_offset <= page_length
    ):
        raise SemanticPositionalAcquisitionError("semantic hit offsets are invalid")


def _is_eligible(
    hit: SemanticRetrievalHit,
    page_index: dict[tuple[str, str], tuple[int, int, int, int]],
) -> bool:
    page_base, _, document_length, _ = page_index[(hit.document_generation_id, hit.page_id)]
    absolute_start = page_base + hit.start_offset
    return document_length > 0 and 4 * absolute_start < document_length


def _hit_identity(hit: SemanticRetrievalHit) -> tuple[object, ...]:
    return (
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


def _duplicate_facts(hit: SemanticRetrievalHit) -> tuple[object, ...]:
    return (
        _hit_identity(hit),
        hit.paper_id,
        hit.physical_page_index,
        hit.display_page_number,
        hit.printed_page_label,
        hit.embedding_span_text,
        hit.raw_semantic_score,
        hit.anchors,
    )
