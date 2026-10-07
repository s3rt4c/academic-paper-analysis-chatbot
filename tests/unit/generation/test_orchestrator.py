from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from academic_chatbot.evidence import models as m
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
from tests.fixtures.evidence_bundle.contracts import PROFILE, VECTOR, hybrid_hit
from tests.unit.evidence.test_packing import group, resolved


class _Cancel:
    def is_set(self) -> bool:
        return False


class _Repository:
    def __init__(self, source: CurrentPaperSource | Exception) -> None:
        self.source = source
        self.calls: list[tuple[str, str]] = []

    def sole_current_paper_source(
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

    def search(self, project, query: str, limit: int = 10):
        self.calls.append((project.project_id, query, limit))
        return SimpleNamespace(
            project_id=project.project_id,
            query=query,
            fusion_profile_id="rrf-v1",
            hits=self.hits,
        )


class _Evidence:
    def __init__(self, bundle) -> None:
        self.bundle = bundle
        self.requests = []

    def build(self, request):
        self.requests.append(request)
        return self.bundle


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
            retrieval=retrieval or _Retrieval((hybrid_hit(),)),
            evidence=evidence or _Evidence(bundle),
            generation=generation,
            embedding_profile_id=PROFILE,
            vector_generation_id=VECTOR,
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

    assert result == LocalGenerationRejected(
        code=LocalGenerationRejectionCode.INVALID_REQUEST
    )
    assert retrieval.calls == []


def test_foreign_hits_are_discarded_before_candidate_conversion() -> None:
    local = hybrid_hit()
    foreign_parent = local.parent_chunk.model_copy(
        update={"paper_id": "paper-foreign"}
    )
    foreign = local.model_copy(update={"parent_chunk": foreign_parent})
    evidence = _Evidence(_ready_bundle())
    service, _ = _service(
        retrieval=_Retrieval((foreign, local)), evidence=evidence
    )

    service.ask(
        project_id="project-1", paper_id="paper-1", question="Question?", cancel=_Cancel()
    )

    assert len(evidence.requests) == 1
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

    assert result is expected
    assert len(generation.calls) == 1


def test_retrieval_failure_is_sanitized_without_private_detail() -> None:
    class BrokenRetrieval:
        def search(self, project, query: str, limit: int = 10):
            raise HybridRetrievalIntegrityError("private source text and path")

    service, generation = _service(retrieval=BrokenRetrieval())

    result = service.ask(
        project_id="project-1", paper_id="paper-1", question="Question?", cancel=_Cancel()
    )

    assert result == LocalGenerationFailed(
        code=LocalGenerationFailureCode.AUTHORITY_UNAVAILABLE
    )
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

    assert result == LocalGenerationRejected(
        code=LocalGenerationRejectionCode.SOURCE_NOT_CURRENT
    )
    assert "Answer." not in result.model_dump_json()
