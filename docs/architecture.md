# Public architecture summary

## Scope boundary

This document summarizes the intended architecture of the Local Academic Paper Analysis Chatbot. It is a design direction, not a claim that every component below is implemented. The repository's completed Phase 0 work is limited to the feasibility evidence, safety boundaries, provenance records, and tests described in the README.

## Intended local analysis flow

1. **PDF ingestion and extraction.** A local pipeline accepts user-selected PDFs, preserves immutable source identity, extracts native text, and uses OCR only when extraction quality requires it.
2. **Structure-aware preparation.** Extracted content is normalized into section-aware chunks while retaining document, page, and span provenance.
3. **Hybrid retrieval.** The planned retrieval layer combines full-text search (FTS) with dense retrieval. Candidates are merged, reranked, and returned with evidence spans rather than detached text alone.
4. **Evidence-grounded generation.** A local LLM receives a bounded evidence package. Citation verification checks that each material claim can be tied back to source spans; unsupported claims are rejected or labelled as unavailable.
5. **Staged deep analysis.** Longer analyses are designed as resumable, resource-bounded stages with explicit inputs, outputs, and checkpoints instead of one unbounded generation.

## Planned local components

| Area | Intended responsibility |
| --- | --- |
| Document pipeline | PDF intake, extraction/OCR, page provenance, and quality checks |
| Retrieval | FTS plus dense indexes, candidate fusion, reranking, and evidence packaging |
| Evidence layer | Span identity, citation verification, and answer-to-source traceability |
| Local model adapter | Pinned local runtime/model selection and constrained generation |
| Storage | SQLite-backed project state, document metadata, checkpoints, and index generations |
| Resource governor | RAM/disk/GPU admission checks, one-heavy-task policy, cancellation, and cleanup |
| User workflow | Local project library, document views, analysis jobs, and evidence-linked answers |

## Privacy and network boundaries

The intended default is local analysis. Private documents, derived text, indexes, prompts, and generated outputs remain local. Any future academic-discovery integration is a distinct, explicit operation limited to approved bibliographic queries and identifiers; document text is not a default network payload.

Local runtime/model files, credentials, working data, logs, and exports are excluded from version control. Provenance manifests may record public upstream URLs, versions, and hashes without bundling the referenced artifacts.

## Phase 0 evidence relationship

Phase 0 validates individual foundations for this direction:

- deterministic PDF anchor handling;
- exact-vector and process-tree measurement boundaries;
- pinned local runtime/model identity checks;
- bounded local inference lifecycle, graceful shutdown, cancellation recovery, and partial-result quarantine.

It does not deliver the complete ingestion, OCR, chunking, hybrid retrieval, evidence-chat, discovery, or packaging layers above. Those remain roadmap work.
