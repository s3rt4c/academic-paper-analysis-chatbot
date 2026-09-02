"""Offline embedding profile and external-artifact contracts."""

from academic_chatbot.embeddings.artifacts import (
    EmbeddingArtifactPaths,
    VerifiedEmbeddingArtifacts,
    load_verified_artifacts,
    verify_artifact_inventory,
)
from academic_chatbot.embeddings.models import (
    ArtifactFile,
    EmbeddingArtifactManifest,
    EmbeddingProfile,
    EmbeddingRole,
    EmbeddingSpanIdentity,
)
from academic_chatbot.embeddings.profile import (
    canonical_profile_bytes,
    embedding_profile_id_for,
    embedding_span_id_for,
)

__all__ = [
    "ArtifactFile",
    "EmbeddingArtifactManifest",
    "EmbeddingArtifactPaths",
    "EmbeddingProfile",
    "EmbeddingRole",
    "EmbeddingSpanIdentity",
    "VerifiedEmbeddingArtifacts",
    "canonical_profile_bytes",
    "embedding_profile_id_for",
    "embedding_span_id_for",
    "load_verified_artifacts",
    "verify_artifact_inventory",
]
