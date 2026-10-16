"""Single-paper retrieval-to-generation application flow."""

from __future__ import annotations

from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.inspection_service import _validate_bundle
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
from academic_chatbot.generation.service import _citation
from academic_chatbot.library.repository import (
    CurrentPaperSource,
    PaperSourceAmbiguousError,
    PaperSourceNotCurrentError,
    UnknownPaperError,
)
from academic_chatbot.ports.model import CancellationSignal
from academic_chatbot.retrieval.fts import RetrievalQueryError
from academic_chatbot.retrieval.hybrid_service import HybridRetrievalIntegrityError
from academic_chatbot.retrieval.scoped_models import (
    Mode,
    ScopedHybridResult,
    ScopedRetrievalError,
    ScopedRetrievalRequest,
    ScopedRetrievalResult,
)
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
    def sole_current_paper_source_read_only(
        self, *, project_id: str, paper_id: str
    ) -> CurrentPaperSource: ...


class _ScopedRetrieval(Protocol):
    def prepare_request(
        self,
        *,
        project_id: str,
        paper_id: str,
        query: str,
        mode: Mode,
        limit: int = 10,
        embedding_profile_id: str | None = None,
    ) -> ScopedRetrievalRequest: ...
    def search_scoped(self, request: ScopedRetrievalRequest) -> ScopedRetrievalResult: ...
    def recheck_current(self, request: ScopedRetrievalRequest) -> None: ...


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
        m.EvidenceErrorCode.VECTOR_NOT_CURRENT,
    }:
        return _failed(LocalGenerationFailureCode.AUTHORITY_UNAVAILABLE)
    if error.error.code == m.EvidenceErrorCode.RESOURCE_LIMIT:
        return _rejected(LocalGenerationRejectionCode.RESOURCE_LIMIT)
    return _rejected(LocalGenerationRejectionCode.SOURCE_NOT_CURRENT)


def _scoped_error(error: ScopedRetrievalError) -> LocalGenerationRejected | LocalGenerationFailed:
    if error.code in {"INVALID_REQUEST", "SOURCE_NOT_FOUND", "UNSAFE_PATH", "PROFILE_MISMATCH"}:
        return _rejected(LocalGenerationRejectionCode.INVALID_REQUEST)
    if error.code in {"SOURCE_NOT_CURRENT", "SOURCE_SCOPE_AMBIGUOUS", "RESOURCE_LIMIT"}:
        return _rejected(LocalGenerationRejectionCode(error.code.value))
    if error.code == "INTERNAL_INVARIANT_FAILURE":
        return _failed(LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE)
    return _failed(LocalGenerationFailureCode.AUTHORITY_UNAVAILABLE)


class SinglePaperAskService:
    """Coordinate one explicit paper and one question through existing authorities."""

    def __init__(
        self,
        *,
        repository: _SourceRepository,
        retrieval: _ScopedRetrieval,
        evidence: _EvidenceService,
        generation: _GenerationService,
        embedding_profile_id: str,
    ) -> None:
        self._repository = repository
        self._retrieval = retrieval
        self._evidence = evidence
        self._generation = generation
        self._embedding_profile_id = embedding_profile_id

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
            scoped = ScopedRetrievalRequest.model_validate(
                self._retrieval.prepare_request(
                    project_id=project_id,
                    paper_id=paper_id,
                    query=question,
                    mode="hybrid",
                    limit=_RETRIEVAL_LIMIT,
                    embedding_profile_id=self._embedding_profile_id,
                )
            )
            expected_scope = m.BundleSourceScope(**source.__dict__)
            if (
                scoped.scope.project_id != project_id
                or scoped.scope.paper_id != paper_id
                or scoped.query != question
                or scoped.mode != "hybrid"
                or scoped.limit != _RETRIEVAL_LIMIT
                or scoped.semantic_identity is None
                or scoped.semantic_identity.embedding_profile_id != self._embedding_profile_id
            ):
                raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")
            if scoped.scope != expected_scope:
                # A new publication can legitimately land between admission reads.
                current = self._resolve_release_source(project_id=project_id, paper_id=paper_id)
                if not isinstance(current, CurrentPaperSource):
                    return current
                if current.project_id != project_id or current.paper_id != paper_id:
                    raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")
                if current != source:
                    return _rejected(LocalGenerationRejectionCode.SOURCE_NOT_CURRENT)
                raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")
            retrieved: ScopedRetrievalResult = TypeAdapter(ScopedRetrievalResult).validate_python(
                self._retrieval.search_scoped(scoped)
            )
            if (
                not isinstance(retrieved, ScopedHybridResult)
                or retrieved.request_fingerprint != scoped.fingerprint
                or retrieved.source != scoped.scope
                or retrieved.query != scoped.query
                or retrieved.limit != scoped.limit
                or retrieved.policy != scoped.policy
                or retrieved.semantic_identity != scoped.semantic_identity
            ):
                raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")
            identity = retrieved.semantic_identity
            if identity is None:
                raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")
            self._retrieval.recheck_current(scoped)
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
                    embedding_profile_id=identity.embedding_profile_id,
                    vector_generation_id=identity.vector_generation_id,
                ),
                candidates=tuple(candidate_ref_from_hybrid(hit) for hit in retrieved.hits),
                preview_budget=m.PreviewBudget(),
            )
            bundle = self._evidence.build(request)
            try:
                bundle, _ = _validate_bundle(bundle, request)
            except EvidencePreparationError as error:
                return _rejected(
                    LocalGenerationRejectionCode.RESOURCE_LIMIT
                    if error.error.code == m.EvidenceErrorCode.RESOURCE_LIMIT
                    else LocalGenerationRejectionCode.BUNDLE_FINGERPRINT_INVALID
                )
            if any(entry.source.scope != bundle.scope for entry in bundle.entries):
                raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")
            self._retrieval.recheck_current(scoped)
        except ScopedRetrievalError as error:
            return _scoped_error(error)
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
            return _failed(LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE)
        except Exception:
            return _failed(LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE)

        if bundle.state == m.PreviewState.PREVIEW_PARTIAL:
            return LocalGenerationAbstained(reason=LocalGenerationAbstentionReason.EVIDENCE_PARTIAL)
        if bundle.state == m.PreviewState.INSUFFICIENT_EVIDENCE:
            return LocalGenerationAbstained(
                reason=LocalGenerationAbstentionReason.EVIDENCE_INSUFFICIENT
            )

        try:
            result: LocalGenerationResult = TypeAdapter(LocalGenerationResult).validate_python(
                self._generation.generate(question=question, bundle=bundle, cancel=cancel)
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
        try:
            self._retrieval.recheck_current(scoped)
        except ScopedRetrievalError as error:
            return _scoped_error(error)
        except Exception:
            return _failed(LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE)
        entries = {entry.citation_label: entry for entry in bundle.entries}
        if (
            result.bundle_fingerprint != bundle.fingerprint
            or tuple(c.citation_label for c in result.citations) != result.citation_labels
            or any(
                citation.citation_label not in entries
                or citation != _citation(entries[citation.citation_label])
                for citation in result.citations
            )
        ):
            return _failed(LocalGenerationFailureCode.INTERNAL_INVARIANT_FAILURE)
        return result

    def _resolve_initial_source(
        self, *, project_id: str, paper_id: str
    ) -> CurrentPaperSource | LocalGenerationRejected | LocalGenerationFailed:
        try:
            return self._repository.sole_current_paper_source_read_only(
                project_id=project_id, paper_id=paper_id
            )
        except UnknownPaperError:
            return _rejected(LocalGenerationRejectionCode.INVALID_REQUEST)
        except PaperSourceNotCurrentError:
            return _rejected(LocalGenerationRejectionCode.SOURCE_NOT_CURRENT)
        except PaperSourceAmbiguousError:
            return _rejected(LocalGenerationRejectionCode.SOURCE_SCOPE_AMBIGUOUS)
        except ScopedRetrievalError as error:
            return _scoped_error(error)
        except Exception:
            return _failed(LocalGenerationFailureCode.AUTHORITY_UNAVAILABLE)

    def _resolve_release_source(
        self, *, project_id: str, paper_id: str
    ) -> CurrentPaperSource | LocalGenerationRejected | LocalGenerationFailed:
        try:
            return self._repository.sole_current_paper_source_read_only(
                project_id=project_id, paper_id=paper_id
            )
        except PaperSourceAmbiguousError:
            return _rejected(LocalGenerationRejectionCode.SOURCE_SCOPE_AMBIGUOUS)
        except (UnknownPaperError, PaperSourceNotCurrentError):
            return _rejected(LocalGenerationRejectionCode.SOURCE_NOT_CURRENT)
        except ScopedRetrievalError as error:
            if error.code == "SOURCE_NOT_FOUND":
                return _rejected(LocalGenerationRejectionCode.SOURCE_NOT_CURRENT)
            return _scoped_error(error)
        except Exception:
            return _failed(LocalGenerationFailureCode.AUTHORITY_UNAVAILABLE)
