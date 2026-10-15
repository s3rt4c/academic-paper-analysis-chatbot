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
- The bounded MVP implements local `llama.cpp` answer generation through a
  single-paper `ask` CLI, with exact template/tokenizer admission, strict
  structured output, citation validation, and currentness rechecks.
- There is no semantic-entailment or paraphrase verifier and no production
  inference/conflict/`not_reported`/unreadable decision engine.

## Implemented MVP answer flow

```text
single selected paper -> current source admission -> scoped hybrid retrieval
-> EvidenceBundle -> bounded context
-> template/tokenizer admission -> local llama.cpp -> strict output
-> citation validation -> currentness check -> CLI answer
```

Source admission requires exactly one published current FileVersion for the
selected paper. The same source-scoped service used by inspection applies the
exact project/paper/FileVersion/DocumentGeneration tuple before each lexical
and semantic cap, then performs unchanged hybrid fusion. It does not acquire
project-wide top-k and filter afterward. Foreign or inconsistent returned
lineage is an invariant failure. Nonready evidence previews abstain before model
access. Paper text remains untrusted data inside the rendered context.

The verified session renders the template, counts tokens, and generates with
one frozen runtime/model pair. Admission enforces rendered prompt tokens plus
1024 reserved output tokens within the 4096-token context. Strict output must
contain an answer and valid evidence citation labels. The service rebuilds the
authoritative bundle and rechecks the selected source before releasing a result.
The ask coordinator also rechecks the admitted semantic/vector publication and
requires every released citation to match its authoritative bundle entry.
Before abstention or generation, the coordinator validates the canonical bundle
seal and binds source, origin, candidate/disposition order and preview budget to
the acquired request using the existing inspection bundle validator.
These checks validate references and currentness; they do not prove semantic
entailment, evidence completeness, or factual correctness.

This path uses loopback-only local transport and externally provisioned model
artifacts. It adds no persistence migration or dependency. It does not implement
multi-paper synthesis, an API/UI, OCR execution, automatic five-status
classification, or the broader staged-analysis design below. Acceptance evidence
is recorded in [mvp-validation.md](mvp-validation.md).

## Implemented V1.1 scoped inspection

`search --paper-id` adds a selected-source request/result contract without
changing project-wide search. Lexical eligibility is part of the SQL predicate
before LIMIT. Semantic eligibility selects global vector rows before exact
block top-k, using at most 4096 rows per block and the existing memmap store.
Hybrid channel depths, weights, ordering and `rrf-v1` are unchanged.

`evidence-bundle inspect` composes this acquisition with the existing
EvidenceBundle resolver, packer and fingerprint authority. Its canonical
`evidence-inspection-result-v1` envelope carries the bundle and bound retrieval
identity, rather than a parallel evidence representation. Default preview
limits are 8 entries and 16KiB. Ready/partial/insufficient are evidence states;
rejected/failed envelopes suppress evidence on invalid or unavailable authority.
Inspection revalidates source/vector currentness before releasing its envelope.

Lexical inspection uses no embedding model. Semantic/hybrid inspection uses
verified BGE artifacts; all inspection modes avoid answer generation and
llama.cpp startup. `ask` uses that same scoped hybrid acquisition, with unchanged
generation schemas, context rendering, 4096/1024 admission, strict output and
citations. Source admission uses logical read-only SQLite transactions. Ordinary
WAL snapshots can maintain SQLite sidecars; this contract does not promise
byte-identical database-family files.

Native coverage reports visible gaps and omissions, not exhaustive document
coverage. Document and selected-source retrieval completeness are `not_assessed`;
support assessment is `not_performed`. Multi-paper synthesis, semantic entailment
certification, scientific factual verification, OCR execution, UI/API,
conversation memory and automatic five-status decisions remain outside V1.1.
See [v1.1-validation.md](v1.1-validation.md) for the tested functional inventory,
Windows acceptance, measured fixture and public revision binding limitation.

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
multi-paper synthesis, and release packaging remain intended future work.
