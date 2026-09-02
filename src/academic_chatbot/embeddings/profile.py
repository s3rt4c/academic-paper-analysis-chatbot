"""Canonical semantic identities for immutable embedding contracts."""

from __future__ import annotations

import hashlib

from academic_chatbot.embeddings.models import (
    EmbeddingProfile,
    EmbeddingSpanIdentity,
    canonical_json_bytes,
)


def canonical_profile_bytes(profile: EmbeddingProfile) -> bytes:
    """Serialize exactly the host-independent Task 0 semantic payload."""

    return canonical_json_bytes(profile.semantic_identity_payload())


def embedding_profile_id_for(profile: EmbeddingProfile) -> str:
    """Return the deterministic identity for one vector-semantic profile."""

    return "ep-sha256-" + hashlib.sha256(canonical_profile_bytes(profile)).hexdigest()


def embedding_span_id_for(identity: EmbeddingSpanIdentity) -> str:
    """Return a deterministic occurrence identity for a future embedding span."""

    return identity.embedding_span_id
