"""Exact preview-byte accounting and an unimplemented future token boundary."""

from __future__ import annotations

from hashlib import sha256
from typing import Protocol, Self

from pydantic import Field, model_validator

from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.serialization import canonical_json_bytes


class ExactPromptTokenCount(m.EvidenceValue):
    """A future counter result bound to the complete prompt and exact profiles."""

    complete_rendered_prompt: m.SourceText
    complete_prompt_sha256: m.Sha256
    generation_profile: m.Identifier
    tokenizer_profile: m.Identifier
    template_hash: m.Sha256
    token_count: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def _bind_complete_prompt(self) -> Self:
        if (
            sha256(self.complete_rendered_prompt.encode("utf-8")).hexdigest()
            != self.complete_prompt_sha256
        ):
            raise ValueError("complete prompt hash does not match")
        return self


class ExactPromptTokenCounter(Protocol):
    def count(
        self,
        complete_rendered_prompt: str,
        expected_generation_profile: str,
        expected_tokenizer_profile: str,
        expected_template_hash: str,
    ) -> ExactPromptTokenCount: ...


def _preview_payload(
    entries: tuple[m.EvidenceBundleEntry, ...], contexts: tuple[m.EvidenceContext, ...]
) -> dict[str, object]:
    # Tentative proposals may exceed the final model's 200-entry admission.
    # Measure the whole projection before constructing an admitted ContentPreview.
    return {
        "concern_profile_id": "stated-study-objective-v1",
        "evidence": [
            dict(
                citation_label=e.citation_label,
                physical_page_index=e.source.physical_page_index,
                text=e.text,
                trust=e.trust,
            )
            for e in entries
        ],
        "context": [
            dict(
                related_citation_labels=list(c.related_citation_labels),
                physical_page_index=c.physical_page_index,
                text=c.text,
                trust=c.trust,
                citable=c.citable,
            )
            for c in contexts
        ],
    }


def render_content_preview(
    entries: tuple[m.EvidenceBundleEntry, ...], contexts: tuple[m.EvidenceContext, ...]
) -> m.ContentPreview:
    return m.ContentPreview(
        evidence=tuple(
            m.ContentEvidence(
                citation_label=e.citation_label,
                physical_page_index=e.source.physical_page_index,
                text=e.text,
                trust=e.trust,
            )
            for e in entries
        ),
        context=tuple(
            m.ContentContext(
                related_citation_labels=c.related_citation_labels,
                physical_page_index=c.physical_page_index,
                text=c.text,
                trust=c.trust,
                citable=c.citable,
            )
            for c in contexts
        ),
    )


def measure_preview_bytes(preview: m.ContentPreview) -> int:
    return len(canonical_json_bytes(preview.model_dump(mode="json")))


def measure_proposed_preview(
    entries: tuple[m.EvidenceBundleEntry, ...], contexts: tuple[m.EvidenceContext, ...]
) -> int:
    """Measure the full proposal, including one exceeding the final entry cap."""
    return len(canonical_json_bytes(_preview_payload(entries, contexts)))
