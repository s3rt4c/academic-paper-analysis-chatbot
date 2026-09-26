"""Synthetic publication and real retrieval through the binary preview boundary."""

import builtins
import hashlib
import io
import json
import os
from pathlib import PurePath

import pytest

from academic_chatbot import cli
from academic_chatbot.domain.library import Project
from academic_chatbot.evidence.models import PreviewBudget
from academic_chatbot.evidence.references import candidate_ref_from_semantic
from academic_chatbot.evidence.resolver import EvidenceReadResolver
from academic_chatbot.evidence.service import EvidenceBundleService
from academic_chatbot.retrieval.selection import (
    CanonicalPosition,
    occurrence_identity,
    select_guarded_earliest_auxiliary,
)
from academic_chatbot.retrieval.semantic import SemanticRetrievalService
from academic_chatbot.retrieval.semantic_position import SemanticPositionalAcquisition
from tests.fixtures.evidence_bundle.database import _SyntheticQueryEmbedder, database
from tests.integration.embeddings.test_vector_publication import _profile
from tests.integration.evidence.test_read_only import install_preview_guards


@pytest.fixture(autouse=True)
def documentation_files_are_unavailable(monkeypatch):
    """Block Python file reads under documentation directories throughout setup."""

    def guarded(original):
        def open_public(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                path = PurePath(os.fsdecode(file))
                if "docs" in (part.casefold() for part in path.parts):
                    pytest.fail("public workflow accessed documentation storage")
            return original(file, *args, **kwargs)

        return open_public

    monkeypatch.setattr(builtins, "open", guarded(builtins.open))
    monkeypatch.setattr(io, "open", guarded(io.open))


class BinaryOnly:
    """Reject accidental platform-dependent text transport."""

    def __init__(self, raw=b""):
        self.buffer = io.BytesIO(raw)

    def read(self, *args):
        pytest.fail("preview read text stdin")

    def write(self, *args):
        pytest.fail("preview wrote text output")


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def run_cli(monkeypatch, db, request):
    streams = [BinaryOnly(request.model_dump_json().encode("utf-8")), BinaryOnly(), BinaryOnly()]
    with monkeypatch.context() as transport:
        for name, stream in zip(("stdin", "stdout", "stderr"), streams, strict=True):
            transport.setattr(cli.sys, name, stream)
        code = cli.main(
            [
                "--data-root",
                str(db.paths.data_root),
                "--max-pdf-bytes",
                "1000000",
                "evidence-bundle",
                "preview",
                "--request-stdin",
            ]
        )
    return code, streams[1].buffer.getvalue(), streams[2].buffer.getvalue()


def assert_bundle(raw, request):
    payload = json.loads(raw)
    assert raw == canonical(payload) + b"\n"
    fingerprint = payload["fingerprint"]
    assert (
        fingerprint
        == hashlib.sha256(
            canonical({key: value for key, value in payload.items() if key != "fingerprint"})
        ).hexdigest()
    )
    for key, value in request.scope.model_dump(mode="json").items():
        assert payload["scope"][key] == value
    assert payload["origin"] == request.origin.model_dump(mode="json")
    assert payload["generation_enabled"] is False
    assert payload["generation_readiness"] == "not_assessed"
    assert payload["support_assessment"] == "not_performed"
    assert payload["generation_token_count"] is None
    assert payload["budget"]["used_content_bytes"] == len(canonical(payload["content_preview"]))
    assert payload["budget"]["used_entries"] == len(payload["entries"])
    return payload


def test_guarded_selector_order_survives_resolver_and_packing(tmp_path):
    db = database(tmp_path)
    prepared = db.semantic(maximum_words=1)
    profile = _profile()
    service = SemanticRetrievalService(
        data_root=db.paths.data_root,
        profile=profile,
        embedder=_SyntheticQueryEmbedder(profile),
    )
    hits = service.search(
        Project(project_id="project-one", display_name="Synthetic"),
        "alpha",
        limit=100,
    ).hits
    assert len(hits) >= 9

    acquisition = SemanticPositionalAcquisition(
        project_id=hits[0].project_id,
        query="alpha",
        embedding_profile_id=hits[0].embedding_profile_id,
        vector_generation_id=hits[0].vector_generation_id,
        policy_id="semantic-positional-acquisition-v1",
        concern_id="stated-study-objective-v1",
        requested_limit=1,
        examined_hits=hits[:9],
        baseline_hits=hits[:1],
        positional_hits=hits[1:9],
        ordered_hits=hits[:9],
    )
    positions = {
        occurrence_identity(hit): CanonicalPosition(
            occurrence_identity=occurrence_identity(hit),
            document_generation_id=hit.document_generation_id,
            page_id=hit.page_id,
            absolute_start=hit.start_offset,
            source_policy_id="semantic-positional-acquisition-v1",
        )
        for hit in acquisition.ordered_hits
    }
    selected = select_guarded_earliest_auxiliary(
        acquisition=acquisition,
        current=acquisition.baseline_hits,
        positions=positions,
        selector_policy_id="guarded-earliest-aux-selector-v1",
    )
    references = tuple(candidate_ref_from_semantic(hit) for hit in selected)
    request = prepared.model_copy(
        update={
            "candidates": references,
            "preview_budget": PreviewBudget(),
        }
    )

    bundle = EvidenceBundleService(
        resolver=EvidenceReadResolver(data_root=db.paths.data_root)
    ).build(request)

    assert tuple(disposition.reference for disposition in bundle.dispositions) == references
    assert tuple(entry.citation_label for entry in bundle.entries) == ("E1", "E2")


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_synthetic_pdf_retrieval_to_binary_preview(monkeypatch, tmp_path, mode):
    db = database(tmp_path)
    request = db.retrieved(mode=mode)
    assert request.candidates
    install_preview_guards(monkeypatch)
    code, raw, error = run_cli(monkeypatch, db, request)
    assert code == 0 and error == b""
    bundle = assert_bundle(raw, request)
    assert bundle["state"] == "preview_ready"
    assert bundle["entries"]
    assert bundle["scope"]["source_pdf_sha256"] == db.rows(
        "SELECT sha256 FROM file_versions WHERE file_version_id = ?",
        (request.scope.file_version_id,),
    )[0][0]
    assert [row["reference"] for row in bundle["dispositions"]] == [
        candidate.model_dump(mode="json") for candidate in request.candidates
    ]
    assert all(row["omission"] is None for row in bundle["dispositions"])
    labels = [entry["citation_label"] for entry in bundle["entries"]]
    assert labels == [f"E{index}" for index in range(1, len(labels) + 1)]
    for entry in bundle["entries"]:
        source = entry["source"]
        assert source["scope"] == bundle["scope"]
        references = [
            contribution
            for candidate in request.candidates
            if candidate.parent.model_dump(mode="json") == source["parent"]
            for contribution in (candidate.lexical, candidate.semantic)
            if contribution is not None
        ]
        assert any(
            (ref.start_offset, ref.end_offset, ref.expected_text_sha256)
            == (source["start_offset"], source["end_offset"], source["text_sha256"])
            for ref in references
        )
        page = db.rows(
            "SELECT canonical_text FROM pages WHERE page_id = ?", (source["parent"]["page_id"],)
        )[0][0]
        assert entry["text"] == page[source["start_offset"] : source["end_offset"]]
        assert entry["trust"] == "untrusted_source_data"
        assert source["text_sha256"] == hashlib.sha256(entry["text"].encode()).hexdigest()
        assert source["anchor_ids"] == [anchor["page_anchor_id"] for anchor in entry["anchors"]]
        for anchor in entry["anchors"]:
            assert anchor["anchor_text"] == page[anchor["char_start"] : anchor["char_end"]]
        assert any(
            entry["citation_label"] in disposition["citation_labels"]
            for disposition in bundle["dispositions"]
        )
    for context in bundle["context"]:
        assert context["citable"] is False
        assert set(context["related_citation_labels"]) <= set(labels)
    repeated = run_cli(monkeypatch, db, request)
    assert repeated == (code, raw, error)


def test_zero_fit_returns_complete_insufficient_bundle(monkeypatch, tmp_path):
    db = database(tmp_path)
    request = db.retrieved(mode="lexical")
    request = request.model_copy(update={"preview_budget": PreviewBudget(max_content_bytes=77)})
    install_preview_guards(monkeypatch)
    code, raw, error = run_cli(monkeypatch, db, request)
    assert code == 3 and error == b""
    bundle = assert_bundle(raw, request)
    assert bundle["state"] == "insufficient_evidence"
    assert bundle["insufficient_reason"] == "budget_too_small"
    assert bundle["entries"] == bundle["context"] == []
    assert len(bundle["dispositions"]) == len(request.candidates)
    assert all(
        row["omission"]["reason"] == "preview_bytes_exceeded" for row in bundle["dispositions"]
    )


def test_forged_retrieved_digest_returns_only_safe_error(monkeypatch, tmp_path):
    db = database(tmp_path)
    request = db.retrieved(mode="lexical")
    first = request.candidates[0]
    forged = first.model_copy(
        update={"lexical": first.lexical.model_copy(update={"expected_text_sha256": "0" * 64})}
    )
    request = request.model_copy(update={"candidates": (forged, *request.candidates[1:])})
    install_preview_guards(monkeypatch)
    code, raw, error = run_cli(monkeypatch, db, request)
    assert code == 2 and raw == b""
    payload = json.loads(error)
    assert payload == {
        "code": "TEXT_DIGEST_MISMATCH",
        "input_position": 0,
        "message": "Candidate text digest does not match",
    }
    assert error == canonical(payload) + b"\n"
