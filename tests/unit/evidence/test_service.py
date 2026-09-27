import pytest

from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.service import EvidenceBundleService
from tests.unit.evidence.test_packing import group, resolved


class Resolver:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def resolve(self, request):
        self.calls.append(request)
        return self.value


def build(data):
    resolver = Resolver(data)
    bundle = EvidenceBundleService(resolver=resolver).build(data.request)
    assert resolver.calls == [data.request]
    return bundle


def test_empty_candidates_and_zero_fit_have_distinct_reasons():
    assert build(resolved()).insufficient_reason == "no_candidates"
    assert (
        build(resolved(group(), budget=m.PreviewBudget(max_content_bytes=77))).insufficient_reason
        == "budget_too_small"
    )


def test_unknown_coverage_prevents_preview_ready():
    assert (
        build(resolved(group(), status=m.CoverageStatus.UNKNOWN_METADATA)).state
        == "preview_partial"
    )


def test_production_token_count_is_always_unavailable():
    bundle = build(resolved(group()))
    assert bundle.generation_token_count is None
    assert bundle.generation_enabled is False
    assert bundle.generation_readiness == "not_assessed"
    assert bundle.support_assessment == "not_performed"
    with pytest.raises(ValueError):
        bundle.generation_enabled = True


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (m.CoverageStatus.NO_FLAGGED_NATIVE_GAPS, "preview_ready"),
        (m.CoverageStatus.KNOWN_NATIVE_GAPS, "preview_partial"),
        (m.CoverageStatus.UNKNOWN_METADATA, "preview_partial"),
    ],
)
def test_coverage_state_table(status, expected):
    data = resolved(group(), status=status)
    before = data.model_dump_json()
    bundle = build(data)
    assert bundle.state == expected
    assert data.model_dump_json() == before
    assert bundle.coverage.packed_page_ids == ("page-0",)
    assert bundle.coverage.document_completeness == "not_assessed"


def test_duplicates_alone_do_not_force_partial():
    first = group()
    assert build(resolved(first, first)).state == "preview_ready"


def test_budget_omission_with_evidence_is_partial():
    data = resolved(
        group("alpha"),
        group("x" * 1000, page=1, rank=2),
        budget=m.PreviewBudget(max_content_bytes=200),
    )
    assert build(data).state == "preview_partial"


def test_service_rejects_resolved_input_as_public_request():
    from academic_chatbot.evidence.serialization import EvidencePreparationError

    data = resolved(group())
    resolver = Resolver(data)
    with pytest.raises(EvidencePreparationError) as error:
        EvidenceBundleService(resolver=resolver).build(data)
    assert error.value.error.code == "INVALID_REQUEST"
    assert resolver.calls == []


def test_resolver_failure_propagates_without_bundle():
    from academic_chatbot.evidence.resolver import EvidenceResolutionError

    original = EvidenceResolutionError(m.EvidenceErrorCode.GENERATION_NOT_CURRENT)

    class Broken:
        def resolve(self, request):
            raise original

    with pytest.raises(EvidenceResolutionError) as error:
        EvidenceBundleService(resolver=Broken()).build(resolved().request)
    assert error.value is original


def test_packing_failure_has_no_bundle():
    from academic_chatbot.evidence.serialization import EvidencePreparationError

    with pytest.raises(EvidencePreparationError) as error:
        build(resolved(group(), group(page=1, rank=1)))
    assert error.value.error.code == "INVALID_ORDER"


def test_service_final_audit_admission(monkeypatch):
    from academic_chatbot.evidence.serialization import (
        EvidencePreparationError,
        canonical_bundle_bytes,
    )

    data = resolved(group())
    expected = len(canonical_bundle_bytes(build(data)))
    monkeypatch.setattr(m, "MAX_AUDIT_BYTES", expected)
    with pytest.raises(EvidencePreparationError) as error:
        build(data)
    assert error.value.error.code == "RESOURCE_LIMIT"


def test_service_adds_no_storage_search_model_or_network_io(monkeypatch):
    import socket
    import sqlite3

    from academic_chatbot.documents.native_pdf import NativePdfParser
    from academic_chatbot.embeddings.embedder import OfflineEmbedder
    from academic_chatbot.retrieval.exact_memmap import ExactVectorStore
    from academic_chatbot.retrieval.service import RetrievalService

    data = resolved(group())

    def forbidden(*args, **kwargs):
        pytest.fail("unexpected service I/O")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    for cls, name in (
        (NativePdfParser, "parse"),
        (OfflineEmbedder, "open"),
        (ExactVectorStore, "open"),
        (RetrievalService, "search"),
    ):
        monkeypatch.setattr(cls, name, forbidden)
    assert build(data).entries


def test_pure_modules_have_no_storage_imports():
    import ast
    import importlib
    from pathlib import Path

    forbidden = (
        "sqlite3",
        "academic_chatbot.retrieval",
        "academic_chatbot.library",
        "academic_chatbot.documents",
        "academic_chatbot.ports",
        "academic_chatbot.embeddings",
        "academic_chatbot.evidence.resolver",
        "academic_chatbot.evidence.service",
    )
    for name in ("budget", "packing", "serialization"):
        module = importlib.import_module(f"academic_chatbot.evidence.{name}")
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imports = (
                [node.module]
                if isinstance(node, ast.ImportFrom)
                else ([alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            )
            assert not any(value and value.startswith(forbidden) for value in imports)
