# Public architecture summary

## Scope boundary

This document separates the repository's implemented public behavior from its
intended future architecture. Only capabilities listed in the next section are
implemented; later sections remain design direction unless stated otherwise.

## Implemented public state

- Immutable native-PDF admission, page/span provenance, lexical retrieval,
  semantic retrieval, and deterministic hybrid fusion are implemented.
- Natural-language lexical planning, bounded deterministic multi-query
  acquisition, concern-native semantic queries, additive positional semantic
  acquisition, guarded auxiliary selection, and rank-only parent reranking are
  explicit deterministic services. They do not guarantee retrieval recall or
  representative academic-corpus ranking quality.
- The Phase 2 evidence layer validates explicit candidate references against
  authoritative source/version/generation state and creates deterministic,
  byte- and entry-bounded EvidenceBundle previews.
- `EvidenceGroup` v1 represents exactly two explicit known children and packs
  them atomically. Automatic member discovery is not implemented.
- Analysis proposal models are caller assertions, not final decisions and not
  authorization for factual output.
- The supported-objective verifier is deterministic and supported-only. For
  standalone evidence, each cited value must be exact UTF-8 source text; no
  whitespace, case, Unicode, qualifier, or numeric normalization occurs.
- EvidenceGroup authority and both children are revalidated, after which scalar
  value projection abstains because no deterministic two-child projection is
  authorized.
- Generation is disabled. There is no semantic-entailment or paraphrase
  verifier, no production inference/conflict/`not_reported`/unreadable decision
  engine, and no tokenizer/model/`llama.cpp` answer-generation path.

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

## Earlier Phase 0 evidence relationship

Phase 0 validates individual foundations for this direction:

- deterministic PDF anchor handling;
- exact-vector and process-tree measurement boundaries;
- pinned local runtime/model identity checks;
- bounded local inference lifecycle, graceful shutdown, cancellation recovery, and partial-result quarantine.

Phase 0 did not itself deliver the later document, retrieval, evidence, or
verification slices now listed as implemented above. OCR execution, a complete
Evidence Chat workflow, broader analysis decision engines, discovery,
generation, and release packaging remain intended future work.
