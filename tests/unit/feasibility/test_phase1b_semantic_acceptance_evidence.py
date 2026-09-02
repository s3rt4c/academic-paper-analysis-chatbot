from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).parents[3]
_REPORT = _ROOT / "benchmarks" / "results" / "semantic-retrieval-phase1b.json"


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def test_phase1b_semantic_acceptance_report_is_canonical_hashed_and_sanitized() -> None:
    raw = _REPORT.read_bytes()
    payload = json.loads(raw)

    assert raw == _canonical_bytes(payload) + b"\n"
    unsigned = {key: value for key, value in payload.items() if key != "report_sha256"}
    assert payload["report_sha256"] == hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    assert payload["schema_version"] == "1.0.0"
    assert payload["report_type"] == "phase1b_semantic_retrieval_acceptance"
    assert payload["measurement_scope"] == "current_host_reference_class"
    assert payload["hard_gates"] and all(
        gate["result"] == "passed" for gate in payload["hard_gates"]
    )
    assert all(
        item["classification"] == "informational"
        for item in payload["informational_metrics"]
    )
    assert {item["label"] for item in payload["span_vector_study"]} == {"MEASURED", "PROJECTED"}
    serialized = raw.lower()
    for forbidden in (b"c:\\", b"/users/", b"onedrive", b"username", b"password", b"secret"):
        assert forbidden not in serialized
