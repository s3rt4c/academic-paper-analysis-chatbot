from __future__ import annotations

from dataclasses import replace

import pytest

from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.resolver import EvidenceResolutionError
from academic_chatbot.evidence.serialization import EvidencePreparationError, seal_bundle
from academic_chatbot.evidence.service import EvidenceBundleService
from academic_chatbot.generation.models import (
    LocalGenerationAbstained,
    LocalGenerationAbstentionReason,
    LocalGenerationFailed,
    LocalGenerationFailureCode,
    LocalGenerationRejected,
    LocalGenerationRejectionCode,
)
from academic_chatbot.generation.orchestrator import SinglePaperAskService
from academic_chatbot.library.repository import (
    CurrentPaperSource,
    PaperSourceAmbiguousError,
    PaperSourceNotCurrentError,
    UnknownPaperError,
)
from academic_chatbot.retrieval.hybrid_service import HybridRetrievalIntegrityError
from academic_chatbot.retrieval.scoped_models import (
    ScopedChannel,
    ScopedChannels,
    ScopedHybridResult,
    ScopedRetrievalError,
    ScopedRetrievalErrorCode,
    ScopedRetrievalRequest,
    SemanticIdentity,
    policy_for,
)
from tests.fixtures.evidence_bundle.contracts import PROFILE, VECTOR, hybrid_hit
from tests.unit.evidence.test_packing import group, resolved


class _Cancel:
    def is_set(self) -> bool:
        return False


class _Repository:
    def __init__(self, source: CurrentPaperSource | Exception) -> None:
        self.source = source
        self.calls: list[tuple[str, str]] = []

    def sole_current_paper_source_read_only(
        self, *, project_id: str, paper_id: str
    ) -> CurrentPaperSource:
        self.calls.append((project_id, paper_id))
        if isinstance(self.source, Exception):
            raise self.source
        return self.source


class _Retrieval:
    def __init__(self, hits=()) -> None:
        self.hits = hits
        self.calls = []
        self.rechecks = []

    def prepare_request(self, **kwargs):
        self.calls.append(kwargs)
        return ScopedRetrievalRequest(
            scope=m.BundleSourceScope(**_source().__dict__),
            query=kwargs["query"],
            mode=kwargs["mode"],
            limit=kwargs["limit"],
            policy=policy_for("hybrid"),
            semantic_identity=SemanticIdentity(
                embedding_profile_id=PROFILE,
                vector_generation_id=VECTOR,
                embedding_profile_sha256="a" * 64,
                source_snapshot_sha256="b" * 64,
                vector_store_manifest_sha256="c" * 64,
            ),
        )

    def search_scoped(self, request):
        self.calls.append(request)
        channel = ScopedChannel(state="empty", requested_limit=50, returned_count=0)
        result = ScopedHybridResult(
            request_fingerprint=request.fingerprint,
            source=request.scope,
            query=request.query,
            limit=request.limit,
            policy=request.policy,
            semantic_identity=request.semantic_identity,
            channels=ScopedChannels(lexical=channel, semantic=channel),
            hits=(),
        )
        if not self.hits:
            return result
        channel = ScopedChannel(state="nonempty", requested_limit=50, returned_count=3)
        return result.model_copy(
            update={
                "hits": self.hits,
                "channels": ScopedChannels(lexical=channel, semantic=channel),
            }
        )

    def recheck_current(self, request):
        self.rechecks.append(request)


def _hit():
    hit = hybrid_hit()
    score = hit.trace.fusion_score.model_copy(update={"numerator": 85, "denominator": 1806})
    return hit.model_copy(update={"trace": hit.trace.model_copy(update={"fusion_score": score})})


class _Evidence:
    def __init__(self, bundle) -> None:
        self.bundle = bundle
        self.requests = []

    def build(self, request):
        self.requests.append(request)
        payload = m.EvidenceBundlePayload.model_validate(
            self.bundle.model_dump(mode="python", exclude={"fingerprint"})
        )
        dispositions = tuple(
            self.bundle.dispositions[0].model_copy(
                update={"reference": reference, "input_position": index}
            )
            for index, reference in enumerate(request.candidates)
        )
        return seal_bundle(
            payload.model_copy(
                update={
                    "origin": request.origin,
                    "dispositions": dispositions,
                }
            )
        )


class _Generation:
    def __init__(self, result, *, during=None) -> None:
        self.result = result
        self.during = during
        self.calls = []

    def generate(self, *, question, bundle, cancel):
        self.calls.append((question, bundle, cancel))
        if self.during is not None:
            self.during()
        return self.result


def _source() -> CurrentPaperSource:
    return CurrentPaperSource(
        project_id="project-1",
        paper_id="paper-1",
        file_version_id="file-1",
        document_generation_id="generation-1",
    )


def _ready_bundle():
    data = resolved(group())
    return EvidenceBundleService(resolver=_Resolved(data)).build(data.request)


class _Resolved:
    def __init__(self, value) -> None:
        self.value = value

    def resolve(self, request):
        return self.value


def _service(*, repository=None, retrieval=None, evidence=None, generation=None):
    bundle = _ready_bundle()
    generation = generation or _Generation(
        LocalGenerationAbstained(reason=LocalGenerationAbstentionReason.CANCELLED)
    )
    return (
        SinglePaperAskService(
            repository=repository or _Repository(_source()),
            retrieval=retrieval or _Retrieval((_hit(),)),
            evidence=evidence or _Evidence(bundle),
            generation=generation,
            embedding_profile_id=PROFILE,
        ),
        generation,
    )


@pytest.mark.parametrize(
    ("source_error", "expected_code"),
    [
        (UnknownPaperError("private"), LocalGenerationRejectionCode.INVALID_REQUEST),
        (
            PaperSourceNotCurrentError("private"),
            LocalGenerationRejectionCode.SOURCE_NOT_CURRENT,
        ),
        (
            PaperSourceAmbiguousError("private"),
            LocalGenerationRejectionCode.SOURCE_SCOPE_AMBIGUOUS,
        ),
    ],
)
def test_source_resolution_is_fail_closed(source_error, expected_code) -> None:
    service, generation = _service(repository=_Repository(source_error))

    result = service.ask(
        project_id="project-1", paper_id="paper-1", question="Question?", cancel=_Cancel()
    )

    assert result == LocalGenerationRejected(code=expected_code)
    assert generation.calls == []
    assert "private" not in result.model_dump_json()


@pytest.mark.parametrize("question", ["", "   ", "x" * 4097])
def test_invalid_question_rejects_before_retrieval(question: str) -> None:
    retrieval = _Retrieval()
    service, _ = _service(retrieval=retrieval)

    result = service.ask(
        project_id="project-1", paper_id="paper-1", question=question, cancel=_Cancel()
    )

    assert result == LocalGenerationRejected(code=LocalGenerationRejectionCode.INVALID_REQUEST)
    assert retrieval.calls == []


def test_foreign_hits_fail_closed_before_candidate_conversion() -> None:
    local = _hit()
    foreign_parent = local.parent_chunk.model_copy(update={"paper_id": "paper-foreign"})
    foreign = local.model_copy(update={"parent_chunk": foreign_parent})
    evidence = _Evidence(_ready_bundle())
    service, _ = _service(retrieval=_Retrieval((foreign, local)), evidence=evidence)

    result = service.ask(
        project_id="project-1", paper_id="paper-1", question="Question?", cancel=_Cancel()
    )

    assert result == LocalGenerationFailed(
        code=LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE
    )
    assert evidence.requests == []


def test_scoped_request_owns_acquisition_and_vector_identity() -> None:
    local = _hit()
    retrieval = _Retrieval((local,))
    evidence = _Evidence(_ready_bundle())
    service, _ = _service(retrieval=retrieval, evidence=evidence)
    service.ask(project_id="project-1", paper_id="paper-1", question="Question?")
    assert retrieval.calls[0] == dict(
        project_id="project-1",
        paper_id="paper-1",
        query="Question?",
        mode="hybrid",
        limit=10,
        embedding_profile_id=PROFILE,
    )
    assert isinstance(retrieval.calls[1], ScopedRetrievalRequest)
    assert retrieval.rechecks
    request = evidence.requests[0]
    assert request.scope == m.BundleSourceScope(**_source().__dict__)
    assert len(request.candidates) == 1
    assert request.candidates[0].parent == local.identity
    assert request.origin.model_dump(mode="json") == {
        "mode": "hybrid",
        "ordering": "caller_asserted_retrieval_order",
        "fusion_profile_id": "rrf-v1",
        "embedding_profile_id": PROFILE,
        "vector_generation_id": VECTOR,
    }


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (m.PreviewState.PREVIEW_PARTIAL, LocalGenerationAbstentionReason.EVIDENCE_PARTIAL),
        (
            m.PreviewState.INSUFFICIENT_EVIDENCE,
            LocalGenerationAbstentionReason.EVIDENCE_INSUFFICIENT,
        ),
    ],
)
def test_nonready_preview_never_calls_generation(state, reason) -> None:
    bundle = _ready_bundle().model_copy(
        update={
            "state": state,
            "insufficient_reason": (
                m.InsufficientReason.NO_CANDIDATES
                if state is m.PreviewState.INSUFFICIENT_EVIDENCE
                else None
            ),
        }
    )
    generation = _Generation(
        LocalGenerationFailed(code=LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE)
    )
    service, _ = _service(evidence=_Evidence(bundle), generation=generation)

    result = service.ask(
        project_id="project-1", paper_id="paper-1", question="Question?", cancel=_Cancel()
    )

    assert result == LocalGenerationAbstained(reason=reason)
    assert generation.calls == []


def test_ready_preview_calls_generation_and_passes_result_through() -> None:
    expected = LocalGenerationFailed(code=LocalGenerationFailureCode.GENERATION_TIMEOUT)
    generation = _Generation(expected)
    service, _ = _service(generation=generation)

    result = service.ask(
        project_id="project-1", paper_id="paper-1", question="Question?", cancel=_Cancel()
    )

    assert result == expected
    assert len(generation.calls) == 1


def test_retrieval_failure_is_sanitized_without_private_detail() -> None:
    class BrokenRetrieval(_Retrieval):
        def search_scoped(self, request):
            raise HybridRetrievalIntegrityError("private source text and path")

    service, generation = _service(retrieval=BrokenRetrieval())

    result = service.ask(
        project_id="project-1", paper_id="paper-1", question="Question?", cancel=_Cancel()
    )

    assert result == LocalGenerationFailed(code=LocalGenerationFailureCode.AUTHORITY_UNAVAILABLE)
    assert "private" not in result.model_dump_json()
    assert generation.calls == []


def test_source_scope_is_rechecked_after_an_answer() -> None:
    from academic_chatbot.generation.models import AnswerCitation, LocalGenerationAnswered

    repository = _Repository(_source())
    answered = LocalGenerationAnswered(
        answer="Answer.",
        citation_labels=("E1",),
        citations=(
            AnswerCitation(
                citation_label="E1",
                paper_id="paper-1",
                file_version_id="file-1",
                document_generation_id="generation-1",
                physical_page_index=0,
                display_page_number=1,
                printed_page_label=None,
                page_id="page-1",
                start_offset=0,
                end_offset=5,
                text_sha256="a" * 64,
                anchor_ids=("anchor-1",),
            ),
        ),
        bundle_fingerprint="b" * 64,
        rendered_prompt_tokens=10,
        completion_tokens=2,
    )
    service, _ = _service(
        repository=repository,
        generation=_Generation(
            answered,
            during=lambda: setattr(
                repository,
                "source",
                replace(_source(), document_generation_id="generation-2"),
            ),
        ),
    )

    result = service.ask(
        project_id="project-1", paper_id="paper-1", question="Question?", cancel=_Cancel()
    )

    assert result == LocalGenerationRejected(code=LocalGenerationRejectionCode.SOURCE_NOT_CURRENT)
    assert "Answer." not in result.model_dump_json()


@pytest.mark.parametrize("code", list(ScopedRetrievalErrorCode))
@pytest.mark.parametrize("stage", ["prepare", "search", "recheck"])
def test_scoped_errors_preserve_safe_taxonomy(code, stage):
    class Broken(_Retrieval):
        def prepare_request(self, **kwargs):
            if stage == "prepare":
                raise ScopedRetrievalError(code)
            return super().prepare_request(**kwargs)

        def search_scoped(self, request):
            if stage == "search":
                raise ScopedRetrievalError(code)
            return super().search_scoped(request)

        def recheck_current(self, request):
            raise ScopedRetrievalError(code)

    service, generation = _service(retrieval=Broken((_hit(),)))
    result = service.ask(project_id="project-1", paper_id="paper-1", question="Question?")
    expected = (
        "INVALID_REQUEST"
        if code in {"INVALID_REQUEST", "SOURCE_NOT_FOUND", "UNSAFE_PATH", "PROFILE_MISMATCH"}
        else code.value
        if code
        in {
            "SOURCE_NOT_CURRENT",
            "SOURCE_SCOPE_AMBIGUOUS",
            "RESOURCE_LIMIT",
            "INTERNAL_INVARIANT_FAILURE",
        }
        else "AUTHORITY_UNAVAILABLE"
    )
    assert result.code == expected
    assert generation.calls == []


@pytest.mark.parametrize("field", ["scope", "query", "limit", "semantic_identity"])
def test_substituted_admitted_request_fails_before_search(field):
    class Substituted(_Retrieval):
        def prepare_request(self, **kwargs):
            request = super().prepare_request(**kwargs)
            value = {
                "scope": request.scope.model_copy(update={"paper_id": "foreign"}),
                "query": "different",
                "limit": 9,
                "semantic_identity": request.semantic_identity.model_copy(
                    update={"embedding_profile_id": "foreign"}
                ),
            }[field]
            return request.model_copy(update={field: value})

    retrieval = Substituted((_hit(),))
    service, generation = _service(retrieval=retrieval)
    result = service.ask(project_id="project-1", paper_id="paper-1", question="Question?")
    assert result.code == "INTERNAL_INVARIANT_FAILURE"
    assert generation.calls == [] and len(retrieval.calls) == 1


@pytest.mark.parametrize(
    "field", ["paper_id", "file_version_id", "document_generation_id", "anchor_ids"]
)
def test_foreign_or_inconsistent_generation_citation_never_releases(field):
    from academic_chatbot.generation.models import LocalGenerationAnswered
    from academic_chatbot.generation.service import _citation

    class ForeignGeneration:
        def generate(self, *, question, bundle, cancel):
            citation = _citation(bundle.entries[0])
            citation = citation.model_copy(
                update={field: ("foreign",) if field == "anchor_ids" else "foreign"}
            )
            return LocalGenerationAnswered(
                answer="Answer.",
                citation_labels=(citation.citation_label,),
                citations=(citation,),
                bundle_fingerprint=bundle.fingerprint,
                rendered_prompt_tokens=10,
                completion_tokens=2,
            )

    service, _ = _service(generation=ForeignGeneration())
    result = service.ask(project_id="project-1", paper_id="paper-1", question="Question?")
    assert result.outcome == "failed" and result.code == "INTERNAL_INVARIANT_FAILURE"
    assert "answer" not in result.model_dump()


@pytest.mark.parametrize("state", list(m.PreviewState))
def test_invalid_bundle_seal_rejects_before_abstention_or_generation(state):
    class InvalidSeal(_Evidence):
        def build(self, request):
            return super().build(request).model_copy(update={"fingerprint": "0" * 64})

    service, generation = _service(
        evidence=InvalidSeal(_ready_bundle().model_copy(update={"state": state}))
    )
    result = service.ask(project_id="project-1", paper_id="paper-1", question="Question?")
    assert result.outcome == "rejected" and result.code == "BUNDLE_FINGERPRINT_INVALID"
    assert generation.calls == []


def test_current_publication_change_between_source_admission_reads_is_rejected():
    repository = _Repository(_source())
    new_source = replace(_source(), document_generation_id="generation-2")

    class PublishedBetweenReads(_Retrieval):
        def prepare_request(self, **kwargs):
            request = super().prepare_request(**kwargs)
            repository.source = new_source
            return request.model_copy(update={"scope": m.BundleSourceScope(**new_source.__dict__)})

    service, generation = _service(repository=repository, retrieval=PublishedBetweenReads())
    result = service.ask(project_id="project-1", paper_id="paper-1", question="Question?")
    assert result.outcome == "rejected" and result.code == "SOURCE_NOT_CURRENT"
    assert generation.calls == [] and len(repository.calls) == 2


@pytest.mark.parametrize("error_type", [EvidenceResolutionError, EvidencePreparationError])
def test_stale_vector_detected_in_evidence_build_is_authority_failure(error_type):
    class StaleVector:
        def build(self, request):
            raise error_type(m.EvidenceErrorCode.VECTOR_NOT_CURRENT)

    service, generation = _service(evidence=StaleVector())
    result = service.ask(project_id="project-1", paper_id="paper-1", question="Question?")
    assert result.outcome == "failed" and result.code == "AUTHORITY_UNAVAILABLE"
    assert generation.calls == []
