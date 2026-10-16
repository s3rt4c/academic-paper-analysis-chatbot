"""Crowded ask uses the same scoped evidence authority as model-free inspection."""

import json

import numpy as np
import pytest

from academic_chatbot.embeddings.repository import EmbeddingRepository
from academic_chatbot.evidence.inspection_models import EvidenceInspectionRequest
from academic_chatbot.evidence.inspection_service import EvidenceInspectionService
from academic_chatbot.evidence.models import EvidenceBundlePayload, PreviewBudget
from academic_chatbot.evidence.resolver import EvidenceReadResolver
from academic_chatbot.evidence.serialization import seal_bundle
from academic_chatbot.evidence.service import EvidenceBundleService
from academic_chatbot.generation.orchestrator import SinglePaperAskService
from academic_chatbot.generation.service import LocalGenerationService
from academic_chatbot.library.repository import ProjectRepository
from academic_chatbot.retrieval.hybrid_service import HybridRetrievalService
from academic_chatbot.retrieval.scoped_models import ScopedChannel, ScopedChannels
from academic_chatbot.retrieval.scoped_service import SourceScopedRetrievalService
from academic_chatbot.retrieval.service import RetrievalService
from tests.e2e.test_local_ask import _disable_external_io, _FakeModel
from tests.fixtures.retrieval.scoped import SparseEmbedder, crowded, semantic_oracle
from tests.integration.embeddings.test_vector_publication import _builder


class _EvidenceModel(_FakeModel):
    def generate(self, request, *, cancel):
        result = super().generate(request, cancel=cancel)
        evidence = json.loads(request.messages[1].content)["evidence"]
        assert evidence and all("foreignmarker" not in item["text"] for item in evidence)
        return result.model_copy(
            update={
                "content": json.dumps(
                    {
                        "answer": "The selected synthetic evidence is cited.",
                        "citation_labels": [item["citation_label"] for item in evidence],
                    }
                )
            }
        )


def components(tmp_path):
    fixture = crowded(tmp_path, vectors=True)
    root = fixture.db.paths.data_root
    scoped = SourceScopedRetrievalService(
        data_root=root,
        repository=ProjectRepository(fixture.db.paths),
        lexical_service=RetrievalService(data_root=root),
        semantic_service=fixture.semantic,
    )
    resolver = EvidenceReadResolver(data_root=root)
    bundles = EvidenceBundleService(resolver=resolver)
    return fixture, scoped, resolver, bundles


def ask_service(fixture, scoped, resolver, bundles, model):
    return SinglePaperAskService(
        repository=ProjectRepository(fixture.db.paths),
        retrieval=scoped,
        evidence=bundles,
        generation=LocalGenerationService(resolver=resolver, model=model),
        embedding_profile_id=fixture.semantic.profile.embedding_profile_id,
    )


def ask(service):
    return service.ask(project_id="project-one", paper_id="selected", question="Alpha")


def test_crowded_ask_and_inspection_share_selected_authority(tmp_path, monkeypatch):
    fixture, scoped, resolver, bundles = components(tmp_path)
    # Independent exhaustive BM25 ordering over this corpus, with no production top-k helper.
    rows = fixture.db.rows(
        "SELECT p.paper_id,bm25(chunk_fts) FROM chunk_fts "
        "JOIN chunks c ON c.chunk_id=chunk_fts.chunk_id "
        "JOIN document_generations dg ON dg.document_generation_id=c.document_generation_id "
        "JOIN file_versions fv ON fv.file_version_id=dg.file_version_id "
        "JOIN papers p ON p.paper_id=fv.paper_id WHERE chunk_fts MATCH 'Alpha' "
        "ORDER BY bm25(chunk_fts)"
    )
    assert len(rows) == 66 and all(row[0] == "distractor" for row in rows[:64])
    assert rows[64][0] == "selected"
    # Separate persisted subset map + exact unit-basis fixture vectors prove selected scores=0,
    # all 64 foreign scores=1; selected cannot enter the old semantic cap50 either.
    assert len(semantic_oracle(fixture)) == 2
    selected_rows = {r[0] for r in semantic_oracle(fixture)}
    matrix_file = next(fixture.db.paths.data_root.rglob("vectors.npy"))
    matrix = np.load(matrix_file, allow_pickle=False)
    assert matrix.shape[0] == 66
    assert all(float(matrix[i, 0]) == (0.0 if i in selected_rows else 1.0) for i in range(66))
    old = HybridRetrievalService(
        data_root=fixture.db.paths.data_root,
        semantic_service=fixture.semantic,
    ).search(fixture.project, "Alpha", limit=10)
    assert len(old.hits) == 10 and not any(h.parent_chunk.paper_id == "selected" for h in old.hits)
    _disable_external_io(monkeypatch)
    for mode in ("lexical", "semantic", "hybrid"):
        hits = scoped.search_scoped(fixture.request(mode)).hits
        assert hits and all(
            (h.parent_chunk.paper_id if mode == "hybrid" else h.paper_id) == "selected"
            for h in hits
        )
    inspection = EvidenceInspectionService(
        retrieval_service=scoped, bundle_service=bundles
    ).inspect(EvidenceInspectionRequest(retrieval_request=fixture.request("hybrid")))
    assert inspection.outcome == "ready"
    model = _EvidenceModel()
    result = ask(ask_service(fixture, scoped, resolver, bundles, model))
    assert (
        result.outcome == "answered" and result.bundle_fingerprint == inspection.bundle.fingerprint
    )
    assert all(
        c.paper_id == "selected"
        and c.file_version_id == fixture.scope.file_version_id
        and c.document_generation_id == fixture.scope.document_generation_id
        for c in result.citations
    )
    evidence = json.loads(model.calls[0].messages[1].content)["evidence"]
    assert evidence and all(
        "selectedmarker" in e["text"] and "foreignmarker" not in e["text"] for e in evidence
    )


@pytest.mark.parametrize(
    "stage", ["before_acquisition", "after_channel", "after_bundle", "during_generation"]
)
@pytest.mark.parametrize("authority", ["source", "vector"])
def test_authority_races_never_release_answer(tmp_path, monkeypatch, stage, authority):
    fixture, scoped, resolver, bundles = components(tmp_path)

    def mutate():
        fixture.db.execute(
            "DELETE FROM "
            + (
                "generation_publications"
                if authority == "source"
                else "vector_generation_publications"
            )
        )

    model = _EvidenceModel(during=mutate if stage == "during_generation" else None)
    if stage == "before_acquisition":
        mutate()
    elif stage == "after_channel":
        original = scoped._lexical.search_scoped

        def search(request):
            value = original(request)
            mutate()
            return value

        monkeypatch.setattr(scoped._lexical, "search_scoped", search)
    elif stage == "after_bundle":
        original = bundles.build

        def build(request):
            value = original(request)
            mutate()
            return value

        monkeypatch.setattr(bundles, "build", build)
    _disable_external_io(monkeypatch)
    result = ask(ask_service(fixture, scoped, resolver, bundles, model))
    assert result.outcome in ("rejected", "failed") and "answer" not in result.model_dump()
    assert len(model.calls) == (1 if stage == "during_generation" else 0)


@pytest.mark.parametrize("damage", ["partial", "foreign_label", "invalid_json"])
def test_crowded_nonready_and_invalid_model_output_are_safe(tmp_path, monkeypatch, damage):
    fixture, scoped, resolver, bundles = components(tmp_path)
    model = _EvidenceModel()
    if damage == "partial":
        fixture.db.execute(
            "UPDATE pages SET needs_ocr=1 WHERE document_generation_id=?",
            (fixture.scope.document_generation_id,),
        )
        _builder(
            EmbeddingRepository(fixture.db.paths),
            fixture.db.paths,
            SparseEmbedder(fixture.semantic.profile),
        ).build(project_id="project-one")
    else:
        original = model.generate

        def generate(request, *, cancel):
            value = original(request, cancel=cancel)
            return value.model_copy(
                update={
                    "content": '{"answer":"Answer","citation_labels":["FOREIGN"]}'
                    if damage == "foreign_label"
                    else "{invalid"
                }
            )

        monkeypatch.setattr(model, "generate", generate)
    _disable_external_io(monkeypatch)
    result = ask(ask_service(fixture, scoped, resolver, bundles, model))
    assert result.outcome == "abstained" if damage == "partial" else result.outcome == "rejected"
    assert len(model.calls) == (0 if damage == "partial" else 1)
    assert "answer" not in result.model_dump()


@pytest.mark.parametrize("substitution", ["empty", "subset", "order", "budget"])
def test_same_source_bundle_cannot_substitute_acquired_candidates_or_budget(
    tmp_path, monkeypatch, substitution
):
    fixture, scoped, resolver, bundles = components(tmp_path)
    original_request = fixture.request("hybrid")
    original_build = bundles.build
    captured = []
    if substitution == "empty":
        original_search = scoped.search_scoped

        def empty(request):
            value = original_search(request)
            captured.extend(value.hits)
            channel = ScopedChannel(state="empty", requested_limit=50, returned_count=0)
            return value.model_copy(
                update={"hits": (), "channels": ScopedChannels(lexical=channel, semantic=channel)}
            )

        monkeypatch.setattr(scoped, "search_scoped", empty)

    def substitute(request):
        if substitution == "empty":
            from academic_chatbot.evidence.references import candidate_ref_from_hybrid

            assert request.candidates == () and len(captured) == 2
            altered = request.model_copy(
                update={"candidates": tuple(candidate_ref_from_hybrid(hit) for hit in captured)}
            )
        elif substitution == "subset":
            assert len(request.candidates) == 2
            altered = request.model_copy(update={"candidates": request.candidates[:1]})
        elif substitution == "order":
            value = original_build(request)
            payload = EvidenceBundlePayload.model_validate(
                value.model_dump(mode="python", exclude={"fingerprint"})
            )
            return seal_bundle(
                payload.model_copy(
                    update={
                        "dispositions": tuple(
                            disposition.model_copy(update={"input_position": index})
                            for index, disposition in enumerate(reversed(value.dispositions))
                        )
                    }
                )
            )
        else:
            altered = request.model_copy(update={"preview_budget": PreviewBudget(max_entries=1)})
        assert request.scope == original_request.scope
        return original_build(altered)

    monkeypatch.setattr(bundles, "build", substitute)
    model = _EvidenceModel()
    _disable_external_io(monkeypatch)
    result = ask(ask_service(fixture, scoped, resolver, bundles, model))
    assert result.outcome == "failed" and result.code == "INTERNAL_INVARIANT_FAILURE"
    assert model.calls == []


def test_historical_vector_switch_at_evidence_open_is_authority_failure(tmp_path, monkeypatch):
    fixture, scoped, resolver, bundles = components(tmp_path)
    old_vector = fixture.request("hybrid").semantic_identity.vector_generation_id
    foreign = tuple(
        fixture.db.rows(
            "SELECT file_version_id,document_generation_id FROM generation_publications "
            "WHERE file_version_id!=?",
            (fixture.scope.file_version_id,),
        )[0]
    )
    fixture.db.execute("DELETE FROM generation_publications WHERE file_version_id=?", (foreign[0],))
    historical = (
        _builder(
            EmbeddingRepository(fixture.db.paths),
            fixture.db.paths,
            SparseEmbedder(fixture.semantic.profile),
        )
        .build(project_id="project-one")
        .generation.vector_generation_id
    )
    fixture.db.execute("INSERT INTO generation_publications VALUES (?,?)", foreign)
    fixture.db.execute(
        "UPDATE vector_generation_publications SET vector_generation_id=?", (old_vector,)
    )
    original = bundles.build

    def changed_before_resolver_open(request):
        fixture.db.execute(
            "UPDATE vector_generation_publications SET vector_generation_id=?", (historical,)
        )
        return original(request)

    monkeypatch.setattr(bundles, "build", changed_before_resolver_open)
    model = _EvidenceModel()
    _disable_external_io(monkeypatch)
    result = ask(ask_service(fixture, scoped, resolver, bundles, model))
    assert result.outcome == "failed" and result.code == "AUTHORITY_UNAVAILABLE"
    assert model.calls == []
