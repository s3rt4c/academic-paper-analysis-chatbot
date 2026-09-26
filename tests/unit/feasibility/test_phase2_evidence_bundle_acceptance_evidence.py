from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parents[3]
_REPORT = _ROOT / "benchmarks" / "results" / "evidence-bundle-phase2.json"

_EXPECTED_ROOT_KEYS = {
    "schema_version",
    "report_type",
    "tested_revision",
    "fixtures",
    "pytest_gates",
    "static_gates",
    "resource_limits",
    "semantics",
    "capabilities",
    "limitations",
    "report_sha256",
}

_EXPECTED_TESTED_REVISION = {
    "commit": "a6f7ca48d7707fd9836aeb9aed207e1667f88b04",
    "tree": "f1e7fc80297a04fdc755db67b60365f188419fff",
}

_EXPECTED_FIXTURES = [
    {
        "fixture_id": "evidence-bundle-contracts-v1",
        "repository_relative_path": "tests/fixtures/evidence_bundle/contracts.py",
        "provenance": "repository-authored-public-synthetic",
        "byte_size": 7590,
        "sha256": "15402ace450b73e435ce8b13a6b611fcb3b002abd0acb9b904c1be3f797ac684",
    },
    {
        "fixture_id": "evidence-bundle-database-v1",
        "repository_relative_path": "tests/fixtures/evidence_bundle/database.py",
        "provenance": "repository-authored-public-synthetic",
        "byte_size": 10578,
        "sha256": "69af6d5638d294e86cdf3b96bbb3d181b3b8e3431d4e1008b92a7595f780d5b1",
    },
    {
        "fixture_id": "evidence-bundle-pdfs-v1",
        "repository_relative_path": "tests/fixtures/evidence_bundle/pdfs.py",
        "provenance": "repository-authored-public-synthetic",
        "byte_size": 934,
        "sha256": "2ee6887bb075a1e2ecdb6de23642717b54c556f8f16c549c351995287947c09b",
    },
    {
        "fixture_id": "hybrid-retrieval-evaluation-v1",
        "repository_relative_path": "tests/fixtures/hybrid_retrieval/evaluation.json",
        "provenance": "repository-authored-public-synthetic",
        "byte_size": 7734,
        "sha256": "645b423d1d996ceac7a2733834559699896f7d558c6e263cc73d39b7d25d4cc6",
    },
    {
        "fixture_id": "native-anchor-pdf-v1",
        "repository_relative_path": "tests/fixtures/pdfs/native_anchor.pdf",
        "provenance": "repository-authored-public-synthetic",
        "byte_size": 2470,
        "sha256": "2d9c30592721d5e27f39c6a047f4e10f2577868d0ac1ef836a81dbdb8180175e",
    },
]

_EXPECTED_PYTEST_GATES = [
    {"id": "unit", "status": "passed", "passed": 2487, "failed": 0, "skipped": 4, "deselected": 0},
    {"id": "contract", "status": "passed", "passed": 5, "failed": 0, "skipped": 1, "deselected": 0},
    {
        "id": "integration",
        "status": "passed",
        "passed": 228,
        "failed": 0,
        "skipped": 1,
        "deselected": 0,
    },
    {"id": "e2e", "status": "passed", "passed": 17, "failed": 0, "skipped": 0, "deselected": 0},
    {
        "id": "security",
        "status": "passed",
        "passed": 53,
        "failed": 0,
        "skipped": 3,
        "deselected": 0,
    },
]

_EXPECTED_STATIC_GATES = [
    {"id": "ruff", "status": "passed"},
    {"id": "mypy_src", "status": "passed"},
    {"id": "pip_check", "status": "passed"},
    {"id": "git_diff_check", "status": "passed"},
]

_EXPECTED_RESOURCE_LIMITS = {
    "kind": "configured_limits",
    "request_utf8_bytes": 2097152,
    "json_depth": 16,
    "candidate_groups": 100,
    "contributions_per_group": 2,
    "identifier_characters": 512,
    "chunk_anchors": 120,
    "selected_fileversion_pages": 1000,
    "resolved_page_text_bytes": 1048576,
    "distinct_resolved_page_text_bytes": 16777216,
    "preview_content_bytes": {"minimum": 77, "default": 16384, "maximum": 65536},
    "entries": {"minimum": 1, "default": 8, "maximum": 200},
    "final_audit_bytes": 4194304,
    "vector_rows": 100000,
    "vector_metadata_bytes": 16777216,
    "vector_payload_bytes": 134217728,
    "manifest_and_profile_json_bytes": 65536,
    "source_snapshot_fileversions": 10000,
}

_EXPECTED_SEMANTICS = {
    "completeness": "not_assessed",
    "support": "not_performed",
    "not_reported": "not_permitted",
    "generation": "disabled",
}

_EXPECTED_CAPABILITIES = {
    "source_version_generation_isolation": True,
    "bounded_evidence_preparation": True,
    "explicit_candidate_reference_resolution": True,
    "deterministic_packing": True,
    "citation_location_validation": True,
    "explicit_evidence_group_representation": True,
    "automatic_group_member_discovery": "deferred",
    "retrieval_ranking_quality": "not_guaranteed",
    "additive_candidate_analysis_view": "not_public",
}

_EXPECTED_LIMITATIONS = [
    {
        "id": "retrieval_ranking_quality_not_guaranteed",
        "text": "Retrieval and ranking quality is not complete or guaranteed.",
    },
    {
        "id": "automatic_group_member_discovery_not_implemented",
        "text": "Automatic group-member discovery is not implemented.",
    },
    {"id": "evidence_completeness_not_assessed", "text": "Evidence completeness is not assessed."},
    {
        "id": "support_classification_not_performed",
        "text": "Support classification is not performed.",
    },
    {
        "id": "retrieval_absence_not_not_reported",
        "text": "Retrieval absence must not be interpreted as not_reported.",
    },
    {"id": "generation_disabled", "text": "Generation is disabled."},
    {
        "id": "additive_candidate_analysis_view_not_public",
        "text": (
            "A bounded acquired-candidate view for richer future analysis is not a "
            "public analysis-layer interface."
        ),
    },
]


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise AssertionError(f"nonfinite JSON constant: {value}")


def _load_report() -> tuple[bytes, dict[str, Any]]:
    raw = _REPORT.read_bytes()
    payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    assert isinstance(payload, dict)
    return raw, payload


def _tested_fixture_bytes(repository_relative_path: str) -> bytes:
    revision = _EXPECTED_TESTED_REVISION["commit"]
    return subprocess.check_output(
        ["git", "show", f"{revision}:{repository_relative_path}"], cwd=_ROOT
    )


def test_phase2_acceptance_evidence_matches_public_commit_9() -> None:
    raw, payload = _load_report()

    assert set(payload) == _EXPECTED_ROOT_KEYS
    assert raw == _canonical_bytes(payload) + b"\n"
    unsigned = {key: value for key, value in payload.items() if key != "report_sha256"}
    assert re.fullmatch(r"[0-9a-f]{64}", payload["report_sha256"])
    assert payload["report_sha256"] == hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    assert payload["schema_version"] == "1.0.0"
    assert payload["report_type"] == "phase2_evidence_bundle_acceptance"

    assert payload["tested_revision"] == _EXPECTED_TESTED_REVISION
    actual_tree = subprocess.check_output(
        ["git", "show", "-s", "--format=%T", _EXPECTED_TESTED_REVISION["commit"]],
        cwd=_ROOT,
        text=True,
    ).strip()
    assert actual_tree == _EXPECTED_TESTED_REVISION["tree"]

    assert payload["fixtures"] == _EXPECTED_FIXTURES
    for fixture in _EXPECTED_FIXTURES:
        relative = fixture["repository_relative_path"]
        assert not Path(relative).is_absolute()
        assert "\\" not in relative
        fixture_bytes = _tested_fixture_bytes(relative)
        assert len(fixture_bytes) == fixture["byte_size"]
        assert hashlib.sha256(fixture_bytes).hexdigest() == fixture["sha256"]

    assert payload["pytest_gates"] == _EXPECTED_PYTEST_GATES
    assert payload["static_gates"] == _EXPECTED_STATIC_GATES
    assert payload["resource_limits"] == _EXPECTED_RESOURCE_LIMITS
    assert payload["semantics"] == _EXPECTED_SEMANTICS
    assert payload["capabilities"] == _EXPECTED_CAPABILITIES
    assert payload["limitations"] == _EXPECTED_LIMITATIONS

    serialized = raw.lower()
    forbidden = (
        b"c:" + b"\\users\\",
        b"/" + b"users/",
        b"/" + b"home/",
        b"one" + b"drive",
        b"akademik_belge_chatbot-" + b"private",
        b"w" + b"1",
        b"w" + b"2",
        b"w" + b"3",
        b"w" + b"4",
        b"si" + b"lver",
        b"pi" + b"lot",
        b"engineering-" + b"report",
        b"diagnostics." + b"json",
        b"docs/" + b"superpowers",
    )
    for value in forbidden:
        assert value not in serialized
