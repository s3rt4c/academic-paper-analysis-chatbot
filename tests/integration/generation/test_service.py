from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.resolver import EvidenceReadResolver, EvidenceResolutionError
from academic_chatbot.evidence.service import EvidenceBundleService
from academic_chatbot.generation.service import LocalGenerationService
from academic_chatbot.ports.model import ModelTimings, StructuredGenerationResult
from tests.fixtures.evidence_bundle.database import database


class _Cancel:
    def is_set(self) -> bool:
        return False


class _Model:
    def __init__(self, content: str, during: Callable[[], None] | None = None) -> None:
        self._content = content
        self._during = during

    def generate(self, request, *, cancel):
        if self._during is not None:
            self._during()
        return StructuredGenerationResult(
            content=self._content,
            prompt_tokens=17,
            completion_tokens=5,
            total_tokens=22,
            timings=ModelTimings(first_token_ms=1.0, total_ms=2.0, tokens_per_second=2.5),
        )


def _bundle(db, mode: str):
    resolver = EvidenceReadResolver(data_root=db.paths.data_root)
    return resolver, EvidenceBundleService(resolver=resolver).build(db.retrieved(mode=mode))


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_current_bundle_answers_through_real_authority(tmp_path: Path, mode: str) -> None:
    db = database(tmp_path)
    resolver, bundle = _bundle(db, mode)
    labels = [entry.citation_label for entry in bundle.entries]
    content = '{"answer":"Current","citation_labels":' + str(labels).replace("'", '"') + "}"

    result = LocalGenerationService(resolver=resolver, model=_Model(content)).generate(
        question="What is the objective?", bundle=bundle, cancel=_Cancel()
    )

    assert result.outcome == "answered"
    assert result.citation_labels == tuple(labels)
    assert tuple(c.text_sha256 for c in result.citations) == tuple(
        entry.source.text_sha256 for entry in bundle.entries
    )


def test_publication_change_during_generation_discards_output(tmp_path: Path) -> None:
    db = database(tmp_path)
    resolver, bundle = _bundle(db, "lexical")
    model = _Model(
        '{"answer":"SECRET","citation_labels":["E1"]}',
        during=lambda: db.execute("DELETE FROM generation_publications"),
    )

    result = LocalGenerationService(resolver=resolver, model=model).generate(
        question="Question?", bundle=bundle, cancel=_Cancel()
    )

    assert result.outcome == "rejected"
    assert result.code == "SOURCE_NOT_CURRENT"
    assert "SECRET" not in result.model_dump_json()


def test_vector_publication_change_during_generation_discards_output(
    tmp_path: Path,
) -> None:
    db = database(tmp_path)
    resolver, bundle = _bundle(db, "semantic")
    model = _Model(
        '{"answer":"SECRET","citation_labels":["E1"]}',
        during=lambda: db.corrupt(
            "UPDATE vector_generation_publications "
            "SET vector_generation_id = 'vector-generation-foreign'"
        ),
    )

    result = LocalGenerationService(resolver=resolver, model=model).generate(
        question="Question?", bundle=bundle, cancel=_Cancel()
    )

    assert result.outcome == "rejected"
    assert result.code == "SOURCE_NOT_CURRENT"
    assert "SECRET" not in result.model_dump_json()


def test_unavailable_real_authority_is_safely_mapped(tmp_path: Path) -> None:
    db = database(tmp_path)
    _, bundle = _bundle(db, "lexical")

    class _Unavailable:
        def resolve(self, request: m.EvidenceBundleRequest) -> m.ResolvedEvidenceInput:
            raise EvidenceResolutionError(m.EvidenceErrorCode.STORAGE_UNAVAILABLE)

    result = LocalGenerationService(
        resolver=_Unavailable(),
        model=_Model('{"answer":"SECRET","citation_labels":["E1"]}'),
    ).generate(question="Question?", bundle=bundle, cancel=_Cancel())

    assert result.outcome == "failed"
    assert result.code == "AUTHORITY_UNAVAILABLE"
    assert "SECRET" not in result.model_dump_json()


def test_cancellation_during_authority_rebuild_discards_answer(tmp_path: Path) -> None:
    db = database(tmp_path)
    delegate, bundle = _bundle(db, "lexical")

    class Cancel:
        value = False

        def is_set(self) -> bool:
            return self.value

    cancel = Cancel()

    class CancellingResolver:
        def resolve(self, request: m.EvidenceBundleRequest) -> m.ResolvedEvidenceInput:
            value = delegate.resolve(request)
            cancel.value = True
            return value

    result = LocalGenerationService(
        resolver=CancellingResolver(),
        model=_Model('{"answer":"SECRET","citation_labels":["E1"]}'),
    ).generate(question="Question?", bundle=bundle, cancel=cancel)

    assert result.outcome == "abstained"
    assert result.reason == "cancelled"
    assert "SECRET" not in result.model_dump_json()
