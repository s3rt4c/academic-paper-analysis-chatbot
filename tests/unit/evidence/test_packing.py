"""Pure packing oracles with synthetic, detached occurrence records."""

from hashlib import sha256

import pytest

from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.packing import normalize_candidate_groups, pack_candidate_groups
from academic_chatbot.evidence.serialization import EvidencePreparationError
from tests.fixtures.evidence_bundle.contracts import PROFILE, VECTOR, resolved_range_data


def group(text="alpha beta", *, page=0, rank=1, semantic=False, equal=False, only=False):
    from academic_chatbot.documents.native_pdf import build_native_pdf_anchor
    from academic_chatbot.documents.normalization import canonicalize_extracted_words
    from academic_chatbot.embeddings.models import EmbeddingSpanIdentity
    from academic_chatbot.library.repository import _page_anchor_id_for

    canonical = canonicalize_extracted_words(
        tuple(
            dict(text=word, x0=1.0, top=1.0 + i * 8, x1=90.0, bottom=6.0 + i * 8)
            for i, word in enumerate(text.split(" "))
        ),
        page_width_points=100.0,
        page_height_points=1000.0,
    )
    native = tuple(
        build_native_pdf_anchor(
            file_version_id="file-1",
            source_pdf_sha256="d" * 64,
            physical_page_index=page,
            page_width_points=100.0,
            page_height_points=1000.0,
            source_page_rotation_degrees=0,
            canonical_page=canonical,
            word=word,
        )
        for word in canonical.words
    )
    base = resolved_range_data()
    parent = base["source"]["parent"]
    parent = {**parent, "page_id": f"page-{page}", "chunk_id": f"chunk-{page}"}
    anchors = tuple(
        m.BundleAnchor(
            page_anchor_id=_page_anchor_id_for(
                page_id=parent["page_id"], evidence_id=a.evidence_id
            ),
            evidence_id=a.evidence_id,
            char_start=a.char_start,
            char_end=a.char_end,
            anchor_text=a.anchor_text,
            anchor_text_sha256=a.anchor_text_sha256,
            boxes_sha256=a.boxes_sha256,
            boxes=a.boxes,
        )
        for a in native
    )
    source = m.EvidenceSourceRef.model_validate(
        {
            **base["source"],
            "parent": parent,
            "physical_page_index": page,
            "display_page_number": page + 1,
            "end_offset": len(text),
            "canonical_page_text_sha256": sha256(text.encode()).hexdigest(),
            "text_sha256": sha256(text.encode()).hexdigest(),
            "page_height_points": 1000.0,
            "parser_profile_sha256": native[0].parser_profile_sha256,
            "anchor_ids": tuple(a.page_anchor_id for a in anchors),
        }
    )
    lexical = m.ResolvedEvidenceRange(
        source=source,
        text=text,
        anchors=anchors,
        contributions=(m.ContributionProvenance(channel="lexical"),),
    )
    lexical_ref = m.LexicalRangeRef(
        start_offset=0,
        end_offset=len(text),
        expected_text_sha256=source.text_sha256,
        expected_anchor_ids=source.anchor_ids,
    )
    ranges = [lexical]
    context = None
    semantic_ref = None
    if semantic:
        start = 0 if equal else anchors[-1].char_start
        span = EmbeddingSpanIdentity(
            **{k: v for k, v in parent.items() if k != "project_id"},
            embedding_profile_id=PROFILE,
            start_offset=start,
            end_offset=len(text),
        )
        provenance = m.ContributionProvenance(
            channel="semantic",
            embedding_profile_id=PROFILE,
            vector_generation_id=VECTOR,
            embedding_span_id=span.embedding_span_id,
        )
        selected = anchors if equal else (anchors[-1],)
        semantic_ref = m.SemanticRangeRef(
            embedding_span_id=span.embedding_span_id,
            start_offset=start,
            end_offset=len(text),
            expected_text_sha256=sha256(text[start:].encode()).hexdigest(),
            expected_anchor_ids=tuple(a.page_anchor_id for a in selected),
        )
        semantic_range = m.ResolvedEvidenceRange(
            source=source.model_copy(
                update={
                    "start_offset": start,
                    "text_sha256": semantic_ref.expected_text_sha256,
                    "anchor_ids": semantic_ref.expected_anchor_ids,
                    "provenance": (provenance,),
                }
            ),
            text=text[start:],
            anchors=selected,
            contributions=(provenance,),
        )
        ranges = [semantic_range] if only else [lexical, semantic_range]
        if only and not equal:
            context = m.ResolvedContext(
                parent=source.parent,
                physical_page_index=page,
                start_offset=0,
                end_offset=len(text),
                text=text,
                text_sha256=source.text_sha256,
                anchors=anchors,
            )
    reference = m.EvidenceCandidateRef(
        parent=source.parent,
        reported_rank=rank,
        lexical=None if only else lexical_ref,
        semantic=semantic_ref,
    )
    return m.ResolvedCandidateGroup(
        input_position=0, reference=reference, ranges=tuple(ranges), context=context
    )


def resolved(*groups, budget=None, status=m.CoverageStatus.NO_FLAGGED_NATIVE_GAPS):
    groups = tuple(g.model_copy(update={"input_position": i}) for i, g in enumerate(groups))
    scope = m.ResolvedEvidenceRange.model_validate(resolved_range_data()).source.scope
    origin = (
        m.CandidateOrigin(
            mode="hybrid",
            fusion_profile_id="rrf-v1",
            embedding_profile_id=PROFILE,
            vector_generation_id=VECTOR,
        )
        if any(g.reference.semantic for g in groups)
        else m.CandidateOrigin(mode="lexical")
    )
    request = m.EvidenceBundleRequest(
        scope=m.BundleSourceScope(**scope.model_dump(exclude={"source_pdf_sha256"})),
        origin=origin,
        candidates=tuple(g.reference for g in groups),
        preview_budget=budget or m.PreviewBudget(),
    )
    coverage = m.EvidenceCoverage(
        status=status,
        pages=(),
        candidate_page_ids=tuple(dict.fromkeys(g.reference.parent.page_id for g in groups)),
        packed_page_ids=(),
        flagged_page_ids=(),
        excluded_page_ids=(),
    )
    return m.ResolvedEvidenceInput(
        request=request, resolved_scope=scope, coverage=coverage, groups=groups
    )


def pack(*groups, budget=None):
    data = resolved(*groups, budget=budget)
    return pack_candidate_groups(normalize_candidate_groups(data), data.request.preview_budget)


def test_exact_duplicate_preserves_first_position():
    first = group()
    result = pack(first, first)
    assert len(result.entries) == 1
    assert result.dispositions[1].omission.reason == "duplicate_candidate"
    assert result.dispositions[1].omission.duplicate_of_input_position == 0


def test_duplicate_does_not_resurrect_omitted_first_candidate():
    first = group("x" * 1000)
    result = pack(first, first, budget=m.PreviewBudget(max_content_bytes=200))
    assert result.entries == ()
    assert result.dispositions[0].omission.reason == "preview_bytes_exceeded"
    assert result.dispositions[1].omission.duplicate_of_input_position == 0


def test_different_pages_with_identical_text_remain_distinct():
    assert len(pack(group(), group(page=1, rank=3)).entries) == 2


def test_dual_channel_same_range_has_one_label_and_two_provenances():
    result = pack(group(semantic=True, equal=True))
    assert len(result.entries) == 1
    assert result.entries[0].citation_label == "E1"
    assert tuple(p.channel for p in result.entries[0].contributions) == ("lexical", "semantic")
    assert result.entries[0].source.provenance == result.entries[0].contributions


def test_unequal_overlapping_ranges_remain_distinct():
    result = pack(group(semantic=True))
    assert [e.text for e in result.entries] == ["alpha beta", "beta"]
    assert result.context == ()


def test_nonfitting_group_does_not_consume_labels():
    result = pack(
        group("alpha"),
        group("x" * 1000, page=1, rank=2),
        group("beta", page=2, rank=3),
        budget=m.PreviewBudget(max_content_bytes=350),
    )
    assert [e.text for e in result.entries] == ["alpha", "beta"]
    assert [e.citation_label for e in result.entries] == ["E1", "E2"]


def test_packing_continues_to_later_fitting_group():
    result = pack(
        group("x" * 1000),
        group("beta", page=1, rank=2),
        budget=m.PreviewBudget(max_content_bytes=200),
    )
    assert [e.text for e in result.entries] == ["beta"]


def test_atomic_group_never_loses_one_contribution():
    result = pack(group(semantic=True), budget=m.PreviewBudget(max_entries=1))
    assert result.entries == ()


def test_entry_limit_only():
    result = pack(group(), group(page=1, rank=2), budget=m.PreviewBudget(max_entries=1))
    assert result.dispositions[1].omission.violated_constraints == ("entry_limit_exceeded",)


def test_both_limits_report_entry_limit_primary():
    result = pack(
        group("alpha"),
        group("x" * 1000, page=1, rank=2),
        budget=m.PreviewBudget(max_entries=1, max_content_bytes=200),
    )
    omission = result.dispositions[1].omission
    assert omission.reason == "entry_limit_exceeded"
    assert set(omission.violated_constraints) == {"entry_limit_exceeded", "preview_bytes_exceeded"}


@pytest.mark.parametrize("rank", [1, 0])
def test_tied_or_decreasing_distinct_rank_fails(rank):
    with pytest.raises(EvidencePreparationError) as error:
        pack(group(rank=2), group(page=1, rank=rank + 1))
    assert error.value.error.code == "INVALID_ORDER"


def test_contradictory_duplicate_rank_fails():
    first = group()
    second = first.model_copy(
        update={"reference": first.reference.model_copy(update={"reported_rank": 4})}
    )
    with pytest.raises(EvidencePreparationError) as error:
        pack(first, second)
    assert error.value.error.code == "CONTRADICTORY_DUPLICATE"


def test_semantic_context_is_atomic_and_non_citable():
    selected = group("alpha beta", semantic=True, only=True)
    result = pack(selected)
    assert len(result.context) == 1 and result.context[0].citable is False
    assert result.context[0].related_citation_labels == ("E1",)
    assert result.context[0].text == "alpha beta"
    assert pack(selected, budget=m.PreviewBudget(max_content_bytes=200)).entries == ()


@pytest.mark.parametrize("text", ["ASCII", "Türkçe", "漢字", "😀", '"quote"', "back\\slash", "é"])
def test_complete_preview_exact_fit_and_one_byte_over(text):
    from academic_chatbot.evidence.budget import measure_preview_bytes

    selected = group(text)
    full = pack(selected)
    exact = measure_preview_bytes(full.content_preview)
    assert len(pack(selected, budget=m.PreviewBudget(max_content_bytes=exact)).entries) == 1
    rejected = pack(selected, budget=m.PreviewBudget(max_content_bytes=exact - 1))
    assert rejected.entries == ()
    assert rejected.used_content_bytes == 77


def test_different_spans_in_same_chunk_are_not_duplicates():
    first = group(semantic=True, only=True, rank=1)
    second = group(semantic=True, only=True, equal=True, rank=3)
    result = pack(first, second)
    assert [e.text for e in result.entries] == ["beta", "alpha beta"]
    assert all(d.omission is None for d in result.dispositions)


@pytest.mark.parametrize("field", ["expected_text_sha256", "expected_anchor_ids", "end_offset"])
def test_conflicting_repeated_reference_assertions_fail(field):
    first = group()
    values = {"expected_text_sha256": "a" * 64, "expected_anchor_ids": ("wrong",), "end_offset": 9}
    ref = first.reference.model_copy(
        update={"lexical": first.reference.lexical.model_copy(update={field: values[field]})}
    )
    with pytest.raises(EvidencePreparationError) as error:
        pack(first, first.model_copy(update={"reference": ref}))
    assert error.value.error.code == "CONTRADICTORY_DUPLICATE"


def test_all_duplicate_and_order_validation_precedes_any_packing():
    first = group("x" * 1000)
    bad = group(page=1, rank=1)
    with pytest.raises(EvidencePreparationError) as error:
        pack(first, bad, budget=m.PreviewBudget(max_content_bytes=77))
    assert error.value.error.code == "INVALID_ORDER"


def test_two_hundred_entries_fit_and_labels_are_contiguous():
    result = pack(
        *(group("a b", page=i, rank=i + 1, semantic=True) for i in range(100)),
        budget=m.PreviewBudget(max_entries=200, max_content_bytes=65536),
    )
    assert len(result.entries) == 200
    assert [e.citation_label for e in result.entries] == [f"E{i}" for i in range(1, 201)]


def test_entry_cap_counts_coalesced_ranges():
    result = pack(group(semantic=True, equal=True), budget=m.PreviewBudget(max_entries=1))
    assert len(result.entries) == 1 and result.dispositions[0].omission is None
