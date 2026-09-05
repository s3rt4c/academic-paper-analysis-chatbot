from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).parents[3]
_REPORT = _ROOT / "benchmarks" / "results" / "hybrid-retrieval-phase1c.json"


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def test_phase1c_hybrid_acceptance_report_is_canonical_hashed_limited_and_sanitized() -> None:
    raw = _REPORT.read_bytes()
    payload = json.loads(raw)

    assert raw == _canonical_bytes(payload) + b"\n"
    unsigned = {key: value for key, value in payload.items() if key != "report_sha256"}
    assert payload["report_sha256"] == hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    assert payload["schema_version"] == "1.0.0"
    assert payload["report_type"] == "phase1c_hybrid_retrieval_acceptance"
    assert payload["phase"] == "1C"
    assert payload["fusion_profile"] == {
        "candidate_depth_policy": "min(100, max(50, final_limit * 5))",
        "canonical_identity": ["project_id", "document_generation_id", "page_id", "chunk_id"],
        "k": 40,
        "lexical_weight": 1.0,
        "profile_id": "rrf-v1",
        "semantic_collapse": "best-span-per-parent-chunk-v1",
        "semantic_weight": 1.0,
    }
    assert payload["accepted_test_suite_summaries"] == {
        "contract": {"passed": 5, "skipped": 1},
        "e2e": {"passed": 4, "skipped": 0},
        "integration": {"passed": 68, "skipped": 1},
        "security": {"passed": 7, "skipped": 1},
        "unit": {"passed": 1996, "skipped": 4},
    }
    assert payload["fixture_evaluation"]["classification"] == "informational_fixture"
    assert payload["fixture_evaluation"]["metrics"] == {
        "hybrid": {"mrr_at_10": 1.0, "ndcg_at_10": 1.0, "recall_at_10": 1.0},
        "lexical": {
            "mrr_at_10": 0.572916666667,
            "ndcg_at_10": 0.519621367951,
            "recall_at_10": 0.625,
        },
        "semantic": {
            "mrr_at_10": 0.5625,
            "ndcg_at_10": 0.482153017388,
            "recall_at_10": 0.5,
        },
    }
    assert payload["fixture_evaluation"]["complementarity_diagnostics"] == {
        "hybrid_relevant": 10,
        "lexical_only_recovered_by_hybrid": [4, 4],
        "lexical_relevant": 6,
        "overlap_relevant": 2,
        "semantic_only_recovered_by_hybrid": [3, 3],
        "semantic_relevant": 5,
    }
    assert payload["hard_acceptance_gates"] and all(
        gate["classification"] == "hard" and gate["result"] == "passed"
        for gate in payload["hard_acceptance_gates"]
    )
    assert {
        "representative academic-corpus retrieval quality",
        "real BGE semantic quality",
        "production-scale latency",
        "50/100/300-paper retrieval performance",
        "large-corpus ranking stability",
        "hybrid superiority on arbitrary corpora",
    }.issubset(set(payload["limitations"]["does_not_certify"]))
    assert payload["limitations"]["validates"] == [
        "deterministic hybrid behavior",
        "rights-safe fixture complementarity",
        "accepted runtime invariants",
    ]
    assert payload["real_model_acceptance"]["decision"] == "reuse_phase1b_accepted_runtime"
    serialized = raw.lower()
    for forbidden in (
        b"c:\\users\\",
        b"/users/",
        b"/home/",
        b"username",
        b"password",
        b"secret",
        b"hf_token",
        b"onedrive",
        b"serta",
    ):
        assert forbidden not in serialized
    for private_publication_reference in (
        b"hybrid_profile_reference",
        b"test_hybrid_profile_reference",
        b"docs/superpowers",
        b".superpowers",
    ):
        assert private_publication_reference not in serialized
