"""Fresh authoritative source/vector admission and generation-free acquisition."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from academic_chatbot.db.connection import DatabasePathError
from academic_chatbot.evidence.models import BundleSourceScope, Identifier
from academic_chatbot.library.repository import ProjectRepository, guard_scoped_database
from academic_chatbot.retrieval.hybrid_service import (
    HybridRetrievalIntegrityError,
    HybridRetrievalService,
)
from academic_chatbot.retrieval.scoped_models import (
    Mode,
    ScopedChannel,
    ScopedChannels,
    ScopedHybridResult,
    ScopedLexicalResult,
    ScopedRetrievalError,
    ScopedRetrievalRequest,
    ScopedRetrievalResult,
    ScopedSemanticResult,
    policy_for,
)
from academic_chatbot.retrieval.semantic import (
    SemanticArtifactIntegrityError,
    SemanticIndexStaleError,
    SemanticIndexUnavailableError,
    SemanticProfileError,
    SemanticQueryError,
    SemanticRetrievalIntegrityError,
    SemanticRetrievalService,
)
from academic_chatbot.retrieval.service import (
    RetrievalIntegrityError,
    RetrievalResults,
    RetrievalStorageError,
)


class _Lexical(Protocol):
    def search_scoped(self, request: ScopedRetrievalRequest) -> RetrievalResults: ...


def safe_error(error: Exception) -> ScopedRetrievalError:
    if isinstance(error, ScopedRetrievalError):
        return error
    if isinstance(error, (SemanticIndexUnavailableError,)):
        return ScopedRetrievalError("VECTOR_UNAVAILABLE")
    if isinstance(error, SemanticIndexStaleError):
        return ScopedRetrievalError("VECTOR_NOT_CURRENT")
    if isinstance(error, (SemanticArtifactIntegrityError, SemanticRetrievalIntegrityError)):
        return ScopedRetrievalError("VECTOR_INTEGRITY")
    if isinstance(error, SemanticProfileError):
        return ScopedRetrievalError("PROFILE_MISMATCH")
    if isinstance(error, (SemanticQueryError, ValidationError)):
        return ScopedRetrievalError("INVALID_REQUEST")
    if isinstance(error, (DatabasePathError, RetrievalStorageError)):
        return ScopedRetrievalError("STORAGE_UNAVAILABLE")
    if isinstance(error, (sqlite3.DatabaseError, RetrievalIntegrityError)):
        return ScopedRetrievalError("STORAGE_INTEGRITY")
    if isinstance(error, MemoryError):
        return ScopedRetrievalError("RESOURCE_LIMIT")
    if isinstance(error, HybridRetrievalIntegrityError):
        return ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")
    return ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")


def _channel(count: int | None, limit: int) -> ScopedChannel:
    return ScopedChannel(
        state="not_run" if count is None else ("nonempty" if count else "empty"),
        requested_limit=None if count is None else limit,
        returned_count=count or 0,
    )


class SourceScopedRetrievalService:
    def __init__(
        self,
        *,
        data_root: Path,
        repository: ProjectRepository,
        lexical_service: _Lexical,
        semantic_service: SemanticRetrievalService
        | Callable[[], SemanticRetrievalService]
        | None = None,
        hybrid_service: HybridRetrievalService | Callable[[], HybridRetrievalService] | None = None,
    ) -> None:
        self._root = Path(data_root)
        self._repository = repository
        self._lexical = lexical_service
        self._semantic = semantic_service
        self._hybrid = hybrid_service

    def _semantic_service(self) -> SemanticRetrievalService:
        if self._semantic is None:
            raise ScopedRetrievalError("VECTOR_UNAVAILABLE")
        if callable(self._semantic):
            self._semantic = self._semantic()
        return self._semantic

    def _hybrid_service(self) -> HybridRetrievalService:
        if self._hybrid is None:
            from typing import cast

            from academic_chatbot.retrieval.service import RetrievalService

            self._hybrid = HybridRetrievalService(
                data_root=self._root,
                semantic_service=self._semantic_service(),
                lexical_service=cast(RetrievalService, self._lexical),
            )
        if callable(self._hybrid):
            self._hybrid = self._hybrid()
        return self._hybrid

    def _source(self, project_id: str, paper_id: str) -> BundleSourceScope:
        guard_scoped_database(self._root, project_id)
        source = self._repository.sole_current_paper_source_read_only(
            project_id=project_id, paper_id=paper_id
        )
        return BundleSourceScope(
            project_id=source.project_id,
            paper_id=source.paper_id,
            file_version_id=source.file_version_id,
            document_generation_id=source.document_generation_id,
        )

    def prepare_request(
        self,
        *,
        project_id: str,
        paper_id: str,
        query: str,
        mode: Mode,
        limit: int = 10,
        embedding_profile_id: str | None = None,
    ) -> ScopedRetrievalRequest:
        try:
            TypeAdapter(Mode).validate_python(mode)
            # Pure request validation occurs before storage or lazy model access.
            provisional = ScopedRetrievalRequest(
                scope=BundleSourceScope(
                    project_id=project_id,
                    paper_id=paper_id,
                    file_version_id="admission-pending",
                    document_generation_id="admission-pending",
                ),
                query=query,
                mode="lexical",
                limit=limit,
                policy=policy_for("lexical"),
                semantic_identity=None,
            )
            if mode == "lexical" and embedding_profile_id is not None:
                raise ScopedRetrievalError("INVALID_REQUEST")
            if mode != "lexical":
                TypeAdapter(Identifier).validate_python(embedding_profile_id)
            scope = self._source(project_id, paper_id)
            identity = None
            if mode != "lexical":
                identity = self._semantic_service().current_identity(project_id=project_id)
                if identity.embedding_profile_id != embedding_profile_id:
                    raise ScopedRetrievalError("PROFILE_MISMATCH")
            request = ScopedRetrievalRequest(
                scope=scope,
                query=provisional.query,
                mode=mode,
                limit=limit,
                policy=policy_for(mode),
                semantic_identity=identity,
            )
            self.recheck_current(request)
            return request
        except Exception as error:
            raise safe_error(error) from None

    def recheck_current(self, request: ScopedRetrievalRequest) -> None:
        try:
            request = ScopedRetrievalRequest.model_validate(request)
            if self._source(request.scope.project_id, request.scope.paper_id) != request.scope:
                raise ScopedRetrievalError("SOURCE_NOT_CURRENT")
            if request.semantic_identity is not None:
                current = self._semantic_service().current_identity(
                    project_id=request.scope.project_id
                )
                if current.embedding_profile_id != request.semantic_identity.embedding_profile_id:
                    raise ScopedRetrievalError("PROFILE_MISMATCH")
                if current != request.semantic_identity:
                    raise ScopedRetrievalError("VECTOR_NOT_CURRENT")
        except Exception as error:
            raise safe_error(error) from None

    def search_scoped(self, request: ScopedRetrievalRequest) -> ScopedRetrievalResult:
        from academic_chatbot.retrieval.scoped_serialization import canonical_scoped_result_bytes

        try:
            request = ScopedRetrievalRequest.model_validate(request)
        except ValidationError:
            raise ScopedRetrievalError("INVALID_REQUEST") from None
        try:
            self.recheck_current(request)
            common = dict(
                request_fingerprint=request.fingerprint,
                source=request.scope,
                query=request.query,
                limit=request.limit,
                policy=request.policy,
                semantic_identity=request.semantic_identity,
            )
            result: ScopedRetrievalResult
            if request.mode == "lexical":
                lexical = self._lexical.search_scoped(request)
                if lexical.project_id != request.scope.project_id or lexical.query != request.query:
                    raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")
                result = ScopedLexicalResult.model_validate(
                    common
                    | dict(
                        hits=lexical.hits,
                        channels=ScopedChannels(
                            lexical=_channel(len(lexical.hits), request.limit),
                            semantic=_channel(None, request.limit),
                        ),
                    )
                )
            elif request.mode == "semantic":
                semantic = self._semantic_service().search_scoped(request)
                if (
                    semantic.project_id != request.scope.project_id
                    or semantic.query != request.query
                    or request.semantic_identity is None
                    or semantic.embedding_profile_id
                    != request.semantic_identity.embedding_profile_id
                    or semantic.vector_generation_id
                    != request.semantic_identity.vector_generation_id
                ):
                    raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE")
                result = ScopedSemanticResult.model_validate(
                    common
                    | dict(
                        hits=semantic.hits,
                        channels=ScopedChannels(
                            lexical=_channel(None, request.limit),
                            semantic=_channel(len(semantic.hits), request.limit),
                        ),
                    )
                )
            else:
                hybrid, lexical_count, semantic_count = self._hybrid_service()._acquire_scoped(
                    request
                )
                depth = min(100, max(50, request.limit * 5))
                result = ScopedHybridResult.model_validate(
                    common
                    | dict(
                        hits=hybrid.hits,
                        channels=ScopedChannels(
                            lexical=_channel(lexical_count, depth),
                            semantic=_channel(semantic_count, depth),
                        ),
                    )
                )
            canonical_scoped_result_bytes(result)
            self.recheck_current(request)
            return result
        except Exception as error:
            # A malformed acquired packet is an invariant failure, never a request rejection.
            if isinstance(error, ValidationError):
                raise ScopedRetrievalError("INTERNAL_INVARIANT_FAILURE") from None
            raise safe_error(error) from None
