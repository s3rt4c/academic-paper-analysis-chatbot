from __future__ import annotations

import pytest

from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.resolver import EvidenceResolutionError
from academic_chatbot.evidence.service import EvidenceBundleService
from academic_chatbot.generation.service import LocalGenerationService
from academic_chatbot.ports.model import ModelTimings, StructuredGenerationResult
from tests.unit.evidence.test_packing import group, resolved


class Cancel:
    def is_set(self) -> bool:
        return False


class Resolver:
    def __init__(self, value: m.ResolvedEvidenceInput) -> None:
        self.value = value
        self.error: EvidenceResolutionError | None = None

    def resolve(self, request: m.EvidenceBundleRequest) -> m.ResolvedEvidenceInput:
        if self.error is not None:
            raise self.error
        return self.value


class FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def generate(self, request, *, cancel):
        self.calls += 1
        return StructuredGenerationResult(
            content=self.content,
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            timings=ModelTimings(first_token_ms=1.0, total_ms=2.0, tokens_per_second=2.0),
        )


def _case(*, status=m.CoverageStatus.NO_FLAGGED_NATIVE_GAPS, groups=None):
    data = resolved(*(groups if groups is not None else (group(),)), status=status)
    resolver = Resolver(data)
    bundle = EvidenceBundleService(resolver=resolver).build(data.request)
    return resolver, bundle


def _service(resolver: Resolver, model: FakeModel) -> LocalGenerationService:
    return LocalGenerationService(resolver=resolver, model=model)


def test_invalid_fingerprint_rejects_before_model_access() -> None:
    resolver, bundle = _case()
    model = FakeModel('{"answer":"A","citation_labels":["E1"]}')
    result = _service(resolver, model).generate(
        question="Question?",
        bundle=bundle.model_copy(update={"fingerprint": "0" * 64}),
        cancel=Cancel(),
    )
    assert result.outcome == "rejected"
    assert result.code == "BUNDLE_FINGERPRINT_INVALID"
    assert model.calls == 0


@pytest.mark.parametrize(
    ("status", "groups", "reason"),
    [
        (m.CoverageStatus.UNKNOWN_METADATA, None, "evidence_partial"),
        (m.CoverageStatus.NO_FLAGGED_NATIVE_GAPS, (), "evidence_insufficient"),
    ],
)
def test_nonready_bundle_abstains_without_model(status, groups, reason) -> None:
    resolver, bundle = _case(status=status, groups=groups)
    model = FakeModel('{"answer":"A","citation_labels":["E1"]}')
    result = _service(resolver, model).generate(
        question="Question?", bundle=bundle, cancel=Cancel()
    )
    assert result.outcome == "abstained"
    assert result.reason == reason
    assert model.calls == 0


def test_ready_answer_projects_authoritative_citation() -> None:
    resolver, bundle = _case()
    model = FakeModel('{"answer":" A ","citation_labels":["E1"]}')
    result = _service(resolver, model).generate(
        question="Question?", bundle=bundle, cancel=Cancel()
    )
    assert result.outcome == "answered"
    assert result.answer == " A "
    assert result.citation_labels == ("E1",)
    assert result.citations[0].text_sha256 == bundle.entries[0].source.text_sha256
    assert result.bundle_fingerprint == bundle.fingerprint
    assert result.rendered_prompt_tokens == 10
    assert result.completion_tokens == 4


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"answer":"A","citation_labels":["E1"]} trailing',
        '{"answer":"A","citation_labels":["E1"],"extra":1}',
        '{"answer":" ","citation_labels":["E1"]}',
        '{"answer":"A","citation_labels":[]}',
        '{"answer":"A","citation_labels":["E1","E1"]}',
        '{"answer":"A","answer":"B","citation_labels":["E1"]}',
        '{"answer":"A","citation_labels":["E1"],"citation_labels":["E2"]}',
    ],
)
def test_invalid_model_output_is_rejected_without_leaking_answer(content: str) -> None:
    resolver, bundle = _case()
    result = _service(resolver, FakeModel(content)).generate(
        question="Question?", bundle=bundle, cancel=Cancel()
    )
    assert result.outcome == "rejected"
    assert result.code == "MODEL_OUTPUT_INVALID"
    assert '"answer"' not in result.model_dump_json()
    assert content not in result.model_dump_json()


def test_unknown_label_rejects_without_returning_answer() -> None:
    resolver, bundle = _case()
    result = _service(
        resolver, FakeModel('{"answer":"SECRET","citation_labels":["E999"]}')
    ).generate(question="Question?", bundle=bundle, cancel=Cancel())
    assert result.outcome == "rejected"
    assert result.code == "CITATION_REFERENCE_INVALID"
    assert "SECRET" not in result.model_dump_json()


def test_authority_staleness_discards_answer() -> None:
    resolver, bundle = _case()
    resolver.error = EvidenceResolutionError(m.EvidenceErrorCode.GENERATION_NOT_CURRENT)
    result = _service(
        resolver, FakeModel('{"answer":"SECRET","citation_labels":["E1"]}')
    ).generate(question="Question?", bundle=bundle, cancel=Cancel())
    assert result.outcome == "rejected"
    assert result.code == "SOURCE_NOT_CURRENT"
    assert "SECRET" not in result.model_dump_json()


def test_unavailable_authority_returns_safe_failure() -> None:
    resolver, bundle = _case()
    resolver.error = EvidenceResolutionError(m.EvidenceErrorCode.STORAGE_UNAVAILABLE)
    result = _service(
        resolver, FakeModel('{"answer":"SECRET","citation_labels":["E1"]}')
    ).generate(question="Question?", bundle=bundle, cancel=Cancel())
    assert result.outcome == "failed"
    assert result.code == "AUTHORITY_UNAVAILABLE"
    assert result.message == "Evidence authority is unavailable"
    assert "SECRET" not in result.model_dump_json()


def test_broken_cancellation_signal_fails_safely_before_model() -> None:
    resolver, bundle = _case()
    model = FakeModel('{"answer":"SECRET","citation_labels":["E1"]}')

    class BrokenCancel:
        def is_set(self) -> bool:
            raise RuntimeError("synthetic nested cancellation detail")

    result = _service(resolver, model).generate(
        question="Question?", bundle=bundle, cancel=BrokenCancel()
    )

    assert result.outcome == "failed"
    assert result.code == "INTERNAL_INVARIANT_FAILURE"
    assert "synthetic" not in result.model_dump_json()
    assert model.calls == 0
