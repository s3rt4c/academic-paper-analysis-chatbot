"""Single-paper retrieval-to-generation application flow."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from pydantic import ValidationError

from academic_chatbot.domain.library import Project
from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.references import candidate_ref_from_hybrid
from academic_chatbot.evidence.resolver import EvidenceResolutionError
from academic_chatbot.evidence.serialization import EvidencePreparationError
from academic_chatbot.generation.models import (
    LocalGenerationAbstained,
    LocalGenerationAbstentionReason,
    LocalGenerationFailed,
    LocalGenerationFailureCode,
    LocalGenerationRejected,
    LocalGenerationRejectionCode,
    LocalGenerationResult,
)
from academic_chatbot.library.repository import (
    CurrentPaperSource,
    PaperSourceAmbiguousError,
    PaperSourceNotCurrentError,
    UnknownPaperError,
)
from academic_chatbot.ports.model import CancellationSignal
from academic_chatbot.retrieval.fts import RetrievalQueryError
from academic_chatbot.retrieval.hybrid_models import HybridRetrievalResults
from academic_chatbot.retrieval.hybrid_service import HybridRetrievalIntegrityError
from academic_chatbot.retrieval.semantic import (
    SemanticArtifactIntegrityError,
    SemanticIndexStaleError,
    SemanticIndexUnavailableError,
    SemanticProfileError,
    SemanticQueryError,
    SemanticRetrievalIntegrityError,
)
from academic_chatbot.retrieval.service import RetrievalIntegrityError, RetrievalStorageError

_RETRIEVAL_LIMIT = 10


class _SourceRepository(Protocol):
    def sole_current_paper_source(
        self, *, project_id: str, paper_id: str
    ) -> CurrentPaperSource: ...


class _HybridRetrieval(Protocol):
    def search(
        self, project: Project, query: str, limit: int = 10
    ) -> HybridRetrievalResults: ...


class _EvidenceService(Protocol):
    def build(self, request: m.EvidenceBundleRequest) -> m.EvidenceBundle: ...


class _GenerationService(Protocol):
    def generate(
        self,
        *,
        question: str,
        bundle: m.EvidenceBundle,
        cancel: CancellationSignal,
    ) -> LocalGenerationResult: ...


class _NeverCancelled:
    def is_set(self) -> bool:
        return False


_NEVER_CANCELLED = _NeverCancelled()


def _rejected(code: LocalGenerationRejectionCode) -> LocalGenerationRejected:
    return LocalGenerationRejected(code=code)


def _failed(code: LocalGenerationFailureCode) -> LocalGenerationFailed:
    return LocalGenerationFailed(code=code)


def _valid_identifier(value: object) -> bool:
    return type(value) is str and bool(value) and len(value) <= m.MAX_IDENTIFIER_CHARS


def _valid_question(value: object) -> bool:
    if type(value) is not str or not value.strip():
        return False
    try:
        return len(value.encode("utf-8")) <= 4096
    except UnicodeEncodeError:
        return False


def _authority_error(
    error: EvidenceResolutionError | EvidencePreparationError,
) -> LocalGenerationResult:
    if error.error.code in {
        m.EvidenceErrorCode.STORAGE_UNAVAILABLE,
        m.EvidenceErrorCode.VECTOR_UNAVAILABLE,
    }:
        return _failed(LocalGenerationFailureCode.AUTHORITY_UNAVAILABLE)
    if error.error.code == m.EvidenceErrorCode.RESOURCE_LIMIT:
        return _rejected(LocalGenerationRejectionCode.RESOURCE_LIMIT)
    return _rejected(LocalGenerationRejectionCode.SOURCE_NOT_CURRENT)


class SinglePaperAskService:
    """Coordinate one explicit paper and one question through existing authorities."""

    def __init__(
        self,
        *,
        repository: _SourceRepository,
        retrieval: _HybridRetrieval,
        evidence: _EvidenceService,
        generation: _GenerationService,
        embedding_profile_id: str,
        vector_generation_id: str | Callable[[], str],
    ) -> None:
        self._repository = repository
        self._retrieval = retrieval
        self._evidence = evidence
        self._generation = generation
        self._embedding_profile_id = embedding_profile_id
        self._vector_generation_id = vector_generation_id

    def ask(
        self,
        *,
        project_id: str,
        paper_id: str,
        question: str,
        cancel: CancellationSignal = _NEVER_CANCELLED,
    ) -> LocalGenerationResult:
        if not (
            _valid_identifier(project_id)
            and _valid_identifier(paper_id)
            and _valid_question(question)
        ):
            return _rejected(LocalGenerationRejectionCode.INVALID_REQUEST)

        source = self._resolve_initial_source(project_id=project_id, paper_id=paper_id)
        if not isinstance(source, CurrentPaperSource):
            return source
        if source.project_id != project_id or source.paper_id != paper_id:
            return _failed(LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE)

        try:
            retrieved = self._retrieval.search(
                Project(project_id=project_id, display_name="single-paper-ask"),
                question,
                limit=_RETRIEVAL_LIMIT,
            )
            hits = tuple(
                hit
                for hit in retrieved.hits
                if hit.identity.project_id == source.project_id
                and hit.parent_chunk.paper_id == source.paper_id
                and hit.parent_chunk.file_version_id == source.file_version_id
                and hit.identity.document_generation_id
                == source.document_generation_id
            )
            vector_generation_id = (
                self._vector_generation_id()
                if callable(self._vector_generation_id)
                else self._vector_generation_id
            )
            request = m.EvidenceBundleRequest(
                scope=m.BundleSourceScope(
                    project_id=source.project_id,
                    paper_id=source.paper_id,
                    file_version_id=source.file_version_id,
                    document_generation_id=source.document_generation_id,
                ),
                origin=m.CandidateOrigin(
                    mode="hybrid",
                    fusion_profile_id="rrf-v1",
                    embedding_profile_id=self._embedding_profile_id,
                    vector_generation_id=vector_generation_id,
                ),
                candidates=tuple(candidate_ref_from_hybrid(hit) for hit in hits),
                preview_budget=m.PreviewBudget(),
            )
            bundle = self._evidence.build(request)
        except (
            RetrievalQueryError,
            RetrievalStorageError,
            RetrievalIntegrityError,
            HybridRetrievalIntegrityError,
            SemanticArtifactIntegrityError,
            SemanticIndexStaleError,
            SemanticIndexUnavailableError,
            SemanticProfileError,
            SemanticQueryError,
            SemanticRetrievalIntegrityError,
        ):
            return _failed(LocalGenerationFailureCode.AUTHORITY_UNAVAILABLE)
        except (EvidenceResolutionError, EvidencePreparationError) as error:
            return _authority_error(error)
        except (ValidationError, TypeError, ValueError):
            return _rejected(LocalGenerationRejectionCode.INVALID_REQUEST)
        except Exception:
            return _failed(LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE)

        if bundle.state == m.PreviewState.PREVIEW_PARTIAL:
            return LocalGenerationAbstained(
                reason=LocalGenerationAbstentionReason.EVIDENCE_PARTIAL
            )
        if bundle.state == m.PreviewState.INSUFFICIENT_EVIDENCE:
            return LocalGenerationAbstained(
                reason=LocalGenerationAbstentionReason.EVIDENCE_INSUFFICIENT
            )

        try:
            result = self._generation.generate(
                question=question, bundle=bundle, cancel=cancel
            )
        except Exception:
            return _failed(LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE)
        if result.outcome != "answered":
            return result

        current = self._resolve_release_source(project_id=project_id, paper_id=paper_id)
        if not isinstance(current, CurrentPaperSource):
            return current
        if current != source:
            return _rejected(LocalGenerationRejectionCode.SOURCE_NOT_CURRENT)
        return result

    def _resolve_initial_source(
        self, *, project_id: str, paper_id: str
    ) -> CurrentPaperSource | LocalGenerationRejected | LocalGenerationFailed:
        try:
            return self._repository.sole_current_paper_source(
                project_id=project_id, paper_id=paper_id
            )
        except UnknownPaperError:
            return _rejected(LocalGenerationRejectionCode.INVALID_REQUEST)
        except PaperSourceNotCurrentError:
            return _rejected(LocalGenerationRejectionCode.SOURCE_NOT_CURRENT)
        except PaperSourceAmbiguousError:
            return _rejected(LocalGenerationRejectionCode.SOURCE_SCOPE_AMBIGUOUS)
        except Exception:
            return _failed(LocalGenerationFailureCode.AUTHORITY_UNAVAILABLE)

    def _resolve_release_source(
        self, *, project_id: str, paper_id: str
    ) -> CurrentPaperSource | LocalGenerationRejected | LocalGenerationFailed:
        try:
            return self._repository.sole_current_paper_source(
                project_id=project_id, paper_id=paper_id
            )
        except PaperSourceAmbiguousError:
            return _rejected(LocalGenerationRejectionCode.SOURCE_SCOPE_AMBIGUOUS)
        except (UnknownPaperError, PaperSourceNotCurrentError):
            return _rejected(LocalGenerationRejectionCode.SOURCE_NOT_CURRENT)
        except Exception:
            return _failed(LocalGenerationFailureCode.AUTHORITY_UNAVAILABLE)
