"""Bounded synchronous reconciliation for project-local vector artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from academic_chatbot.embeddings.models import EmbeddingProfile
from academic_chatbot.embeddings.repository import EmbeddingRepository
from academic_chatbot.embeddings.vector_build import (
    VectorBuildError,
    _verify_finalized_artifact,
)
from academic_chatbot.storage.paths import PathEscapeError, ProjectPaths


class VectorReconciliationError(ValueError):
    """Raised when an authoritative active vector generation is not healthy."""


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    active_generation_id: str | None
    recovered_generation_ids: tuple[str, ...]
    stale_generation_ids: tuple[str, ...]
    discarded_candidate_ids: tuple[str, ...]


def reconcile_vector_generations(
    *,
    paths: ProjectPaths,
    repository: EmbeddingRepository,
    profile: EmbeddingProfile,
    project_id: str,
) -> ReconciliationResult:
    """Verify active state and discard only incomplete DB candidates.

    Filesystem artifacts never activate themselves.  Finalized artifacts that
    lack authoritative metadata remain inert forensic orphans; this bounded
    function deliberately does not recursively clean unknown directories.
    """

    if project_id != paths.project_id:
        raise VectorReconciliationError("reconciliation cannot access another project")
    if repository.get_profile(profile.embedding_profile_id) != profile:
        raise VectorReconciliationError("requested embedding profile is not registered")

    active = repository.active_generation(
        project_id=project_id, embedding_profile_id=profile.embedding_profile_id
    )
    if active is not None:
        try:
            _verify_finalized_artifact(paths, repository, active, profile)
        except (OSError, PathEscapeError, ValueError, VectorBuildError) as error:
            raise VectorReconciliationError(
                "authoritative active vector artifact is missing or corrupt"
            ) from error

    recovered: list[str] = []
    stale: list[str] = []
    for finalized in repository.finalized_unpublished_generations(
        project_id=project_id, embedding_profile_id=profile.embedding_profile_id
    ):
        if not repository.is_generation_current(finalized.vector_generation_id):
            repository.mark_stale(finalized.vector_generation_id)
            stale.append(finalized.vector_generation_id)
            continue
        try:
            _verify_finalized_artifact(paths, repository, finalized, profile)
            repository.publish(finalized.vector_generation_id)
        except (OSError, PathEscapeError, ValueError, VectorBuildError) as error:
            raise VectorReconciliationError(
                "finalized vector candidate is missing or corrupt"
            ) from error
        recovered.append(finalized.vector_generation_id)

    discarded: list[str] = []
    for candidate in repository.candidate_generations(
        project_id=project_id, embedding_profile_id=profile.embedding_profile_id
    ):
        # DB_CANDIDATE cannot be visible.  Deleting only its rows unlocks a
        # deterministic retry while retaining any filesystem artifact as inert.
        repository.discard_candidate(candidate.vector_generation_id)
        discarded.append(candidate.vector_generation_id)
    return ReconciliationResult(
        active_generation_id=(
            recovered[-1] if recovered else None if active is None else active.vector_generation_id
        ),
        recovered_generation_ids=tuple(recovered),
        stale_generation_ids=tuple(stale),
        discarded_candidate_ids=tuple(discarded),
    )
