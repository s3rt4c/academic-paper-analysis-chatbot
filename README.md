# Local Academic Paper Analysis Chatbot

An evidence-grounded, local-first research-assistant project for deep academic-paper analysis on consumer Windows hardware.

## Why this project exists

Researchers need more than document chat: they need answers that can be traced to source evidence, a workflow that respects private corpora, and predictable operation on local hardware. This repository records the technical foundation for that goal.

## Current status

**Phase 0 — technical feasibility: complete.** The implemented code, tests, manifests, and pinned reports validate selected local components and lifecycle safeguards on a recorded reference profile.

**Phase 1A — local document core: implemented.** The repository now includes a Windows-first, offline native-PDF path from local project storage through evidence-bearing lexical retrieval.

**Phase 1B — native-text semantic retrieval foundation: complete.** It adds a
private, local-only semantic retrieval path to the Phase 1A document core.
This is not hybrid retrieval, a full document workspace, an evidence-chat
interface, or a representative 50–300-paper performance certification.

**Phase 1C — deterministic hybrid retrieval: complete and ready for a separate
publication review.** It composes the accepted lexical and semantic channels
under the frozen `rrf-v1` profile. Reranking, OCR execution, LLM analysis,
an API/UI, academic discovery, background workers, and release packaging
remain outside this completed slice.

## What Phase 0 validates

- Reproducible Python packaging and strict test/tool configuration.
- Reference-hardware facts and resource-bound feasibility checks.
- Immutable native-PDF anchor evidence and deterministic exact-vector retrieval checks.
- A pinned local `llama.cpp` feasibility slice with model/runtime provenance, process-tree containment, shutdown checks, cancellation recovery, and partial-result quarantine.
- Public, hash-addressed benchmark reports. Model files, runtimes, private corpora, and generated credentials are deliberately not included.

## What Phase 1A provides

- Windows-first local Project, Paper, and immutable FileVersion persistence.
- Secure local native-PDF admission into SHA-256 content-addressed originals.
- Canonical native-text parsing with persisted `NativePdfAnchor` evidence.
- Immutable document generations, page-bounded `lexical-chunk-v1` chunks, and SQLite FTS5 indexing.
- Deterministic project-scoped lexical retrieval through a local CLI, returning page, chunk, and range-scoped anchor evidence.
- Offline acceptance coverage for the normal document path, including no-network and no-external-subprocess guards.

OCR execution, embeddings, vector and hybrid retrieval, reranking, LLM analysis/synthesis, FastAPI, a browser UI, and recovery/workers are not implemented in Phase 1A.

## What Phase 1B provides

- A frozen `BAAI/bge-small-en-v1.5` embedding profile at immutable revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
- Verified, administrator-provisioned local model artifacts; model assets are
  neither included in Git nor downloaded by the application at runtime.
- CPU-only ONNX Runtime embeddings with exact tokenizer budgets, explicit
  document/query roles, and no silent truncation.
- Tokenizer-aware semantic spans, immutable project-local vector generations,
  float16 read-only memory-mapped exact-vector storage, and deterministic
  cosine ranking.
- `semantic-index build` and `search --mode semantic`, producing
  evidence-bearing semantic hits tied to the active document generation.
- Fail-closed stale-index behavior, explicit rebuilds, valid empty-generation
  results, and unchanged lexical retrieval.

The accepted reference-class measurements and gate outcomes are recorded in
[`benchmarks/results/semantic-retrieval-phase1b.json`](benchmarks/results/semantic-retrieval-phase1b.json).
They are local acceptance evidence, not general support guarantees.

## What Phase 1C provides

- Deterministic lexical plus semantic retrieval using frozen `rrf-v1`: exact
  unweighted reciprocal-rank fusion with `k=40` and equal channel weights.
- One result per parent Chunk identity, while semantic multi-span candidates
  collapse to one representative vote and retain their exact span evidence.
- Read-only orchestration that preserves separate lexical and semantic raw
  scores and anchors, then resolves honest current parent context.
- Explicit fail-closed behavior for unavailable, stale, or corrupt semantic
  state; hybrid search never silently becomes lexical-only when semantic
  health is required.
- `search --mode hybrid` with explicit semantic profile and model inputs.
  Lexical search remains the default mode, and the application does not
  acquire models at runtime.

The rights-safe synthetic evaluation and final acceptance facts are recorded
in [`benchmarks/results/hybrid-retrieval-phase1c.json`](benchmarks/results/hybrid-retrieval-phase1c.json).
Its lexical, semantic, and hybrid values are deterministic fixture results,
not a claim that hybrid is 100% accurate or generally superior to either
channel. Representative, diverse academic-corpus retrieval quality and
retrieval scale remain unvalidated.

## Architecture direction

The planned system is a local RAG research assistant with PDF ingestion, extraction/OCR, section-aware chunking, FTS plus dense retrieval, evidence spans, citation verification, staged deep analysis, SQLite-backed local state, and a resource governor. See [docs/architecture.md](docs/architecture.md) for the public design summary.

The included [technical design document](docs/Local_Academic_Paper_Analysis_Chatbot_Documentation.docx) is **design documentation**. It describes the intended architecture; it is not evidence that every planned capability is implemented.

## Local-first privacy model

Private papers, indexes, models, runtime assets, credentials, logs, and exports belong in local ignored directories. The intended product keeps analysis local by default; any future networked academic-discovery action must be explicit, bounded, and separate from local document content.

## Tech stack

- Python 3.12
- FastAPI/Pydantic/HTTPX for the planned service boundary
- NumPy and ONNX Runtime for local retrieval-oriented work
- pdfplumber and pypdfium2 for PDF-oriented feasibility work
- llama.cpp as the pinned local inference runtime evaluated in Phase 0

## Phase 0 reference-profile measurements

These are recorded measurements from one controlled reference profile, not general performance guarantees. Both reports use llama.cpp `b10007`, CUDA 12.4 runtime provenance, and 20 CPU samples.

| Profile | Model | CPU first-token p95 | CUDA first token | CPU process-tree peak | GPU offload |
| --- | --- | ---: | ---: | ---: | --- |
| Default | Qwen3-8B Q4_K_M | 4,485 ms | 407 ms | 4.10 GB | 37/37 layers |
| Fallback | Qwen3-4B Q4_K_M | 2,953 ms | 187 ms | 2.51 GB | 37/37 layers |

The machine-readable evidence is in [`benchmarks/results/`](benchmarks/results/). Each report records immutable model/runtime identities, report hashes, lifecycle evidence, and cleanup assertions.

## Repository structure

```text
artifacts/manifests/  Pinned public provenance for models and runtimes
benchmarks/           Phase 0 configuration and verified reports
docs/                 Public architecture/design documentation
src/                  Feasibility and domain implementation
tests/                Unit, integration, and deterministic fixture coverage
tools/                Documentation and fixture support tools
```

## Roadmap

- **Phase 0 — Technical feasibility:** complete
- **Phase 1A — Local document core:** complete
- **Phase 1B — Native-text semantic retrieval foundation:** complete
- **Phase 1C — Deterministic hybrid retrieval:** complete; publication review pending
- **Phase 2 — Deep analysis and Evidence Chat:** planned
- **Phase 3 — Academic discovery:** planned
- **Phase 4 — Packaging and hardening:** planned

## Security considerations

- Never commit private papers, model files, API keys, logs, indexes, or generated exports.
- Verify artifact hashes and provenance before using a local runtime or model.
- Treat PDF content as untrusted input.
- Keep local-only data outside synchronized folders where practical.

## Current limitations

- Phase 0 is a feasibility baseline, not a production application.
- The recorded measurements apply only to the documented reference profile and pinned artifacts.
- Model/runtime downloads are intentionally out of scope for this repository.
- No public hosted service, user interface, or cloud deployment is provided.
- Phase 1B does not implement hybrid fusion, reranking, OCR execution, LLM
  analysis/synthesis, citation-verification analysis, FastAPI, a browser UI,
  background jobs/workers, automatic model acquisition, or ANN/vector
  databases.
- Phase 1B's repeated public synthetic span study does not certify diverse
  50-, 100-, or 300-paper academic-corpus latency, token-length distribution,
  excluded-unembeddable incidence, or large-corpus query P95.
- Phase 1C does not certify representative academic-corpus retrieval quality,
  real-BGE semantic quality, production-scale latency, 50/100/300-paper
  retrieval performance, large-corpus ranking stability, or hybrid
  superiority on arbitrary corpora. OCR execution, reranking, LLM analysis,
  citation-verification analysis, API/UI, jobs/workers, ANN, automatic model
  acquisition, and representative large-corpus certification remain deferred.

## Development setup

From the repository root, create a Python 3.12 environment and install the locked dependencies:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install pip-tools==7.5.3
.venv\Scripts\pip-compile.exe --extra dev --generate-hashes --allow-unsafe --upgrade-package build==1.5.0 --output-file requirements.lock pyproject.toml
.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.lock
.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

Run the non-live baseline checks:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit -q
.venv\Scripts\python.exe -m pytest tests/contract -q
.venv\Scripts\python.exe -m pytest tests/integration -q
.venv\Scripts\python.exe -m pytest tests/e2e -q
.venv\Scripts\python.exe -m pytest tests/security -q
.venv\Scripts\python.exe -m ruff check src tests
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pip check
```

`tools/rasterize_pdf.py` uses Poppler from `PATH` by default. If needed, set `ACADEMIC_CHATBOT_POPPLER_PATH` to the directory containing the Poppler binaries.

## Phase 1B semantic workflow

The model root is selected explicitly and must already contain the verified
frozen artifact inventory. It remains outside Git and outside project data.

```powershell
$profile = "ep-sha256-3f8fd2dbcff088eb61b2ef1ecbc6de57644a425722a586fef32059516146a929"
$modelRoot = "<approved-local-model-root>"
$dataRoot = ".\\local-data"
$maxPdfBytes = 100000000

.venv\Scripts\python.exe -m academic_chatbot --data-root $dataRoot --max-pdf-bytes $maxPdfBytes project create --project-id research --display-name Research
.venv\Scripts\python.exe -m academic_chatbot --data-root $dataRoot --max-pdf-bytes $maxPdfBytes paper create --project-id research --paper-id paper-one
.venv\Scripts\python.exe -m academic_chatbot --data-root $dataRoot --max-pdf-bytes $maxPdfBytes import-pdf --project-id research --paper-id paper-one --source <local-pdf-path>
.venv\Scripts\python.exe -m academic_chatbot --data-root $dataRoot --max-pdf-bytes $maxPdfBytes semantic-index build --project-id research --embedding-profile-id $profile --model-root $modelRoot
.venv\Scripts\python.exe -m academic_chatbot --data-root $dataRoot --max-pdf-bytes $maxPdfBytes search --mode semantic --project-id research --query "accuracy" --embedding-profile-id $profile --model-root $modelRoot
.venv\Scripts\python.exe -m academic_chatbot --data-root $dataRoot --max-pdf-bytes $maxPdfBytes search --mode hybrid --project-id research --query "accuracy" --embedding-profile-id $profile --model-root $modelRoot
```

For the opt-in real frozen-model contract, set
`ACADEMIC_CHATBOT_BGE_ARTIFACT_ROOT` to that approved local root and run:

```powershell
.venv\Scripts\python.exe -m pytest tests/contract/test_offline_embedder.py -m real_embedding -q
```

## License

Project-authored content is licensed under the [Apache License 2.0](LICENSE). Third-party model and runtime artifacts are not bundled; their provenance and upstream licensing remain separate.
