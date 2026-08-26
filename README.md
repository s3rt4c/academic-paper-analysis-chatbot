# Local Academic Paper Analysis Chatbot

An evidence-grounded, local-first research-assistant project for deep academic-paper analysis on consumer Windows hardware.

## Why this project exists

Researchers need more than document chat: they need answers that can be traced to source evidence, a workflow that respects private corpora, and predictable operation on local hardware. This repository records the technical foundation for that goal.

## Current status

**Phase 0 — technical feasibility: complete.** The implemented code, tests, manifests, and pinned reports validate selected local components and lifecycle safeguards on a recorded reference profile.

**Phases 1–4 are planned, not implemented product features.** This repository does not yet provide the full end-user document workspace, hybrid retrieval experience, evidence-chat interface, academic discovery workflow, or release packaging described in the roadmap.

## What Phase 0 validates

- Reproducible Python packaging and strict test/tool configuration.
- Reference-hardware facts and resource-bound feasibility checks.
- Immutable native-PDF anchor evidence and deterministic exact-vector retrieval checks.
- A pinned local `llama.cpp` feasibility slice with model/runtime provenance, process-tree containment, shutdown checks, cancellation recovery, and partial-result quarantine.
- Public, hash-addressed benchmark reports. Model files, runtimes, private corpora, and generated credentials are deliberately not included.

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
- **Phase 1 — Local document core:** planned / next
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

## Development setup

From the repository root, create a Python 3.12 environment and install the locked dependencies:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip pip-tools
.venv\Scripts\pip-compile.exe --extra dev --generate-hashes --output-file requirements.lock pyproject.toml
.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.lock
.venv\Scripts\python.exe -m pip install --no-deps -e .
```

Run the non-live baseline checks:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit -q
.venv\Scripts\python.exe -m ruff check src tests
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pip check
```

`tools/rasterize_pdf.py` uses Poppler from `PATH` by default. If needed, set `ACADEMIC_CHATBOT_POPPLER_PATH` to the directory containing the Poppler binaries.

## License

Project-authored content is licensed under the [Apache License 2.0](LICENSE). Third-party model and runtime artifacts are not bundled; their provenance and upstream licensing remain separate.
