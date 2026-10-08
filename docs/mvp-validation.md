# MVP validation record

## Tested revision and scope

Validation was run from the independent public projection checkout on
2026-10-07 with Python 3.12 and the repository's hash-locked dependencies.
The editable package resolved to this checkout.

- Tested public functional revision: `3c5f07946068e2943cd08c0b75bd10c880ee6a27`.
- Tested tree: `a0ba31c65dd65eb4df26f374bd94a87c2ef5aa1b`.
- Generation profile: `mvp-qwen3-8b-b10007-cuda-v1`.
- Runtime profile: `b10007-win-cuda-12.4-x64`.
- Model profile: `qwen3-8b-q4-k-m`.

The tested revision contains both functional MVP commits. This documentation
commit records those results without changing production code or tests.

## Offline validation

The ordinary suites ran with live opt-in disabled, fresh external test
workspaces, and `--import-mode=importlib`. They required no external model or
runtime artifacts.

| Suite | Passed | Skipped | Duration |
| --- | ---: | ---: | ---: |
| unit | 2692 | 4 | 56.77 s |
| contract | 5 | 1 | 0.90 s |
| integration | 257 | 2 | 76.71 s |
| e2e | 22 | 0 | 11.41 s |
| security | 53 | 3 | 13.72 s |

Ruff, `mypy src`, `pip check`, and Git diff checks passed. The focused
orchestrator/generation/CLI/offline E2E gate passed with 170 tests. The live test
collected successfully and skipped by default with exit code 0.

## Real Windows live acceptance

The fresh run from the tested public revision passed: **1 passed in
123.84 seconds**, exit code 0. Standard error was empty.

The repository-authored synthetic single-paper PDF exercised real BGE retrieval,
authoritative EvidenceBundle preparation, trusted/untrusted context rendering,
the verified local `llama.cpp` session, `/apply-template`, `/tokenize`, token
admission, Qwen3 generation, strict structured output, citation validation,
currentness rechecks, and an answered CLI result. The test asserted a nonblank
answer and citations scoped to the selected paper. It does not establish general
answer accuracy or entailment.

The command shape was:

```text
python -m pytest tests/integration/generation/test_local_ask_live.py -q -m llama_cpp --import-mode=importlib
```

Live execution requires `ACADEMIC_CHATBOT_RUN_MVP_LLAMA_CPP=1` and the five
externally configured artifact inputs identified by these environment names:

- `ACADEMIC_CHATBOT_MVP_EMBEDDING_MODEL_ROOT`
- `ACADEMIC_CHATBOT_MVP_LLAMA_CPP_RUNTIME_DIR`
- `ACADEMIC_CHATBOT_MVP_LLAMA_CPP_RUNTIME_MANIFEST`
- `ACADEMIC_CHATBOT_MVP_LLAMA_CPP_MODEL`
- `ACADEMIC_CHATBOT_MVP_LLAMA_CPP_MODEL_MANIFEST`

Filesystem values and temporary-workspace arguments are intentionally omitted.
Artifacts were already provisioned outside the repository and were not copied
into it. Configuration was supplied only to the child process; no persistent
environment settings were changed. Transport was loopback-only, with inherited
proxy configuration disabled and redirects rejected.

## Limits

This acceptance covers the frozen single-paper CLI MVP and a synthetic positive
live case. It does not certify representative academic-corpus retrieval quality,
large-corpus performance, semantic entailment, factual correctness, evidence
completeness, multi-paper synthesis, an API/UI, OCR execution, automatic
five-status classification, or general production readiness. Historical Phase 2
acceptance remains a separate record for its original tested revision.
