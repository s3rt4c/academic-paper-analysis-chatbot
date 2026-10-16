"""Independent opt-in Windows acceptance with existing frozen external artifacts."""

import json
import os
import platform
import socket
import statistics
import threading
import time
from pathlib import Path

import psutil
import pytest

from academic_chatbot import cli
from academic_chatbot.embeddings.artifacts import load_verified_artifacts
from academic_chatbot.embeddings.profile import approved_bge_small_en_v15_profile
from academic_chatbot.evidence.inspection_models import EvidenceInspectionRequest
from academic_chatbot.evidence.inspection_service import EvidenceInspectionService
from academic_chatbot.evidence.resolver import EvidenceReadResolver
from academic_chatbot.evidence.service import EvidenceBundleService
from academic_chatbot.feasibility import llama_slice as runtime
from academic_chatbot.generation import llama_cpp
from academic_chatbot.library.repository import ProjectRepository
from academic_chatbot.retrieval.exact_memmap import ExactVectorStore
from academic_chatbot.retrieval.hybrid_service import HybridRetrievalService
from academic_chatbot.retrieval.scoped_service import SourceScopedRetrievalService
from academic_chatbot.retrieval.semantic import SemanticRetrievalService
from academic_chatbot.retrieval.service import RetrievalService
from academic_chatbot.storage.paths import ProjectPaths
from tests.fixtures.evidence_bundle.database import Database
from tests.fixtures.retrieval.scoped import pdf_pages
from tests.integration.generation.test_local_ask_live import _DIRECTORIES, _PATH_ENV

_RUN_ENV = "ACADEMIC_CHATBOT_RUN_V11_LLAMA_CPP"
_QUERY = "What is the study objective?"


def _live_paths():
    if os.environ.get(_RUN_ENV) != "1":
        pytest.skip("live V1.1 artifacts are not enabled")
    if any(not os.environ.get(name) for name in _PATH_ENV.values()):
        pytest.fail("opted-in live V1.1 artifact configuration is incomplete")
    paths = {role: Path(os.environ[name]) for role, name in _PATH_ENV.items()}
    if any(not path.is_absolute() for path in paths.values()):
        pytest.fail("opted-in live V1.1 artifact paths must be absolute")
    if any(
        not (path.is_dir() if role in _DIRECTORIES else path.is_file())
        for role, path in paths.items()
    ):
        pytest.fail("opted-in live V1.1 artifacts are missing or have the wrong kind")
    return paths


def _samples(operation):
    operation()  # one unmeasured warm-up, then exactly five samples
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1000)
    return dict(
        raw_ms=samples, median_ms=statistics.median(samples), range_ms=[min(samples), max(samples)]
    )


@pytest.mark.llama_cpp
def test_live_windows_v11_scoped_inspection_and_ask(tmp_path, monkeypatch, capsys):
    artifacts = _live_paths()
    profile = approved_bge_small_en_v15_profile()
    verified = load_verified_artifacts(artifacts["embedding_model_root"], profile=profile)
    runtime_manifest = runtime.load_llama_runtime_manifest(artifacts["runtime_manifest"])
    model_manifest = runtime.load_gguf_model_manifest(artifacts["model_manifest"])
    assert runtime_manifest.runtime_id == "b10007-win-cuda-12.4-x64"
    assert model_manifest.profile_id == "qwen3-8b-q4-k-m"
    runtime._preflight_llama_run_artifacts(
        runtime_directory=artifacts["runtime_dir"],
        runtime_manifest=runtime_manifest,
        model_path=artifacts["model"],
        model_manifest=model_manifest,
    )
    owned = {}
    ports = []
    leases = []
    maps = []
    calls = {"apply_template": 0, "tokenize": 0, "source_recheck": 0, "vector_recheck": 0}
    admitted = []
    bundles_seen = []
    asking = False
    generated = False
    release_calls = {"source": 0, "vector": 0}
    peak = [0]
    memory_errors = []
    stop = threading.Event()
    original_create = runtime.CtypesLlamaWindowsProcessApi.create_process

    def create(self, **kwargs):
        information = original_create(self, **kwargs)
        process = psutil.Process(information.process_id)
        owned[process.pid] = process.create_time()
        return information

    monkeypatch.setattr(runtime.CtypesLlamaWindowsProcessApi, "create_process", create)
    original_start = runtime.start_llama_server_windows_session

    def start(**kwargs):
        session = original_start(**kwargs)
        ports.append(session.bound_port)
        return session

    monkeypatch.setattr(runtime, "start_llama_server_windows_session", start)
    original_lease = runtime.open_llama_run_artifact_lease

    def lease(**kwargs):
        value = original_lease(**kwargs)
        leases.append(value)
        return value

    monkeypatch.setattr(runtime, "open_llama_run_artifact_lease", lease)
    original_open = ExactVectorStore.open

    def open_store(*args, **kwargs):
        value = original_open(*args, **kwargs)
        maps.append(value)
        return value

    monkeypatch.setattr(ExactVectorStore, "open", open_store)
    for name in ("apply_template", "tokenize"):
        original = getattr(llama_cpp._Phase0PromptSession, name)

        def observe(self, body, _original=original, _name=name):
            value = _original(self, body)
            calls[_name] += 1
            return value

        monkeypatch.setattr(llama_cpp._Phase0PromptSession, name, observe)
    original_generate = llama_cpp.VerifiedLlamaCppModel.generate

    def generate(self, *args, **kwargs):
        nonlocal generated
        value = original_generate(self, *args, **kwargs)
        generated = True
        return value

    monkeypatch.setattr(llama_cpp.VerifiedLlamaCppModel, "generate", generate)
    original_admit = llama_cpp.count_and_admit

    def admit(*args, **kwargs):
        value = original_admit(*args, **kwargs)
        admitted.append(value.exact_prompt_token_count.token_count)
        return value

    monkeypatch.setattr(llama_cpp, "count_and_admit", admit)
    original_source = ProjectRepository.sole_current_paper_source_read_only

    def source(self, **kwargs):
        value = original_source(self, **kwargs)
        if asking:
            calls["source_recheck"] += 1
            if generated:
                release_calls["source"] += 1
        return value

    monkeypatch.setattr(ProjectRepository, "sole_current_paper_source_read_only", source)
    original_current = SourceScopedRetrievalService.recheck_current

    def current(self, request):
        value = original_current(self, request)
        if asking and request.semantic_identity is not None:
            calls["vector_recheck"] += 1
            if generated:
                release_calls["vector"] += 1
        return value

    monkeypatch.setattr(SourceScopedRetrievalService, "recheck_current", current)
    original_build = EvidenceBundleService.build

    def build(self, request):
        value = original_build(self, request)
        if asking:
            bundles_seen.append(value)
        return value

    monkeypatch.setattr(EvidenceBundleService, "build", build)

    def sample_memory():
        while not stop.wait(0.01):
            total = 0
            for pid, born in tuple(owned.items()):
                try:
                    process = psutil.Process(pid)
                    if process.create_time() != born:
                        continue
                    children = process.children(recursive=True)
                    for child in children:
                        owned.setdefault(child.pid, child.create_time())
                    total += process.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
                except psutil.AccessDenied:
                    memory_errors.append("access_denied")
            peak[0] = max(peak[0], total)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        root = ["--data-root", str(tmp_path / "data"), "--max-pdf-bytes", "1000000"]

        def invoke(args):
            code = cli.main([*root, *args])
            output = capsys.readouterr()
            assert code == 0 and output.err == "", output.out
            return json.loads(output.out)

        invoke(["project", "create", "--project-id", "p", "--display-name", "Synthetic"])
        for paper, texts in (
            (
                "selected",
                (
                    "The study objective is to evaluate a synthetic offline method. "
                    + "padding " * 90,
                )
                * 2,
            ),
            (
                "distractor",
                (
                    "What is the study objective? The study objective is to evaluate "
                    "a synthetic offline method. Study objective study objective.",
                )
                * 64,
            ),
        ):
            invoke(["paper", "create", "--project-id", "p", "--paper-id", paper])
            invoke(
                [
                    "import-pdf",
                    "--project-id",
                    "p",
                    "--paper-id",
                    paper,
                    "--source",
                    str(pdf_pages(tmp_path / (paper + ".pdf"), texts)),
                ]
            )
        index_args = [
            "semantic-index",
            "build",
            "--project-id",
            "p",
            "--embedding-profile-id",
            profile.embedding_profile_id,
            "--model-root",
            str(artifacts["embedding_model_root"]),
        ]
        invoke(index_args)
        paths = ProjectPaths.create(tmp_path / "data", project_id="p")
        db = Database(paths, None)
        semantic = SemanticRetrievalService.open_from_model_root(
            data_root=paths.data_root,
            project_id="p",
            profile_id=profile.embedding_profile_id,
            model_root=artifacts["embedding_model_root"],
            scoped=True,
        )
        scoped = SourceScopedRetrievalService(
            data_root=paths.data_root,
            repository=ProjectRepository(paths),
            lexical_service=RetrievalService(data_root=paths.data_root),
            semantic_service=semantic,
        )
        requests = {
            mode: scoped.prepare_request(
                project_id="p",
                paper_id="selected",
                query=_QUERY,
                mode=mode,
                limit=10,
                embedding_profile_id=profile.embedding_profile_id if mode != "lexical" else None,
            )
            for mode in ("lexical", "semantic", "hybrid")
        }
        from academic_chatbot.domain.library import Project

        old = HybridRetrievalService(data_root=paths.data_root, semantic_service=semantic).search(
            Project(project_id="p", display_name="Synthetic"), _QUERY, limit=10
        )
        assert len(old.hits) == 10 and all(
            h.parent_chunk.paper_id == "distractor" for h in old.hits
        )
        # Independent full-corpus SQL proves the selected native matches below old lexical cap50.
        rows = db.rows(
            "SELECT p.paper_id,bm25(chunk_fts) FROM chunk_fts "
            "JOIN chunks c ON c.chunk_id=chunk_fts.chunk_id "
            "JOIN document_generations dg ON dg.document_generation_id=c.document_generation_id "
            "JOIN file_versions fv ON fv.file_version_id=dg.file_version_id "
            "JOIN papers p ON p.paper_id=fv.paper_id "
            "WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts)",
            ('"study" OR "objective"',),
        )
        first_selected = next(i + 1 for i, row in enumerate(rows) if row[0] == "selected")
        assert first_selected > 50 and sum(row[0] == "distractor" for row in rows) >= 60
        for request in requests.values():
            result = scoped.search_scoped(request)
            assert result.hits and all(
                (h.parent_chunk.paper_id if request.mode == "hybrid" else h.paper_id) == "selected"
                for h in result.hits
            )
        resolver = EvidenceReadResolver(data_root=paths.data_root)
        bundles = EvidenceBundleService(resolver=resolver)
        inspection_request = EvidenceInspectionRequest(retrieval_request=requests["hybrid"])
        inspector = EvidenceInspectionService(retrieval_service=scoped, bundle_service=bundles)
        inspected = inspector.inspect(inspection_request)
        assert inspected.outcome == "ready" and inspected.bundle.entries
        assert all(e.source.scope.paper_id == "selected" for e in inspected.bundle.entries)
        assert not owned

        # Fixture-only writes introduce then restore a source release race.
        publications = [tuple(row) for row in db.rows("SELECT * FROM generation_publications")]

        class Race:
            count = 0

            def __init__(self, mutation):
                self.mutation = mutation

            def build(self, request):
                value = bundles.build(request)
                self.count += 1
                if self.count == 2:
                    self.mutation()
                return value

        source_race = EvidenceInspectionService(
            retrieval_service=scoped,
            bundle_service=Race(
                lambda: db.execute(
                    "DELETE FROM generation_publications WHERE file_version_id=?",
                    (requests["hybrid"].scope.file_version_id,),
                )
            ),
        )
        try:
            failed = source_race.inspect(inspection_request)
            assert failed.outcome == "rejected" and failed.error.code == "SOURCE_NOT_CURRENT"
            assert "bundle" not in failed.model_dump() and not owned
        finally:
            for publication in publications:
                db.execute(
                    "INSERT OR IGNORE INTO generation_publications VALUES (?,?)", publication
                )

        # Build a genuine historical vector publication from a different fixture snapshot.
        foreign = next(
            row for row in publications if row[0] != requests["hybrid"].scope.file_version_id
        )
        db.execute("DELETE FROM generation_publications WHERE file_version_id=?", (foreign[0],))
        try:
            invoke(index_args)
            historical = db.rows("SELECT vector_generation_id FROM vector_generation_publications")[
                0
            ][0]
        finally:
            db.execute("INSERT INTO generation_publications VALUES (?,?)", foreign)
            invoke(index_args)
        current_vector = requests["hybrid"].semantic_identity.vector_generation_id
        assert historical != current_vector
        vector_race = EvidenceInspectionService(
            retrieval_service=scoped,
            bundle_service=Race(
                lambda: db.execute(
                    "UPDATE vector_generation_publications SET vector_generation_id=?",
                    (historical,),
                )
            ),
        )
        try:
            failed = vector_race.inspect(inspection_request)
            assert failed.outcome == "failed" and failed.error.code == "VECTOR_NOT_CURRENT"
            assert "bundle" not in failed.model_dump() and not owned
        finally:
            db.execute(
                "UPDATE vector_generation_publications SET vector_generation_id=?",
                (current_vector,),
            )
        measurements = {
            "lexical": _samples(lambda: scoped.search_scoped(requests["lexical"])),
            "semantic": _samples(lambda: scoped.search_scoped(requests["semantic"])),
            "inspection": _samples(lambda: inspector.inspect(inspection_request)),
        }
        assert not owned
        asking = True
        before_source = calls["source_recheck"]
        before_vector = calls["vector_recheck"]
        started = time.perf_counter()
        payload = invoke(
            [
                "ask",
                "--project-id",
                "p",
                "--paper-id",
                "selected",
                "--question",
                _QUERY,
                "--embedding-profile-id",
                profile.embedding_profile_id,
                "--embedding-model-root",
                str(artifacts["embedding_model_root"]),
                "--runtime-dir",
                str(artifacts["runtime_dir"]),
                "--runtime-manifest",
                str(artifacts["runtime_manifest"]),
                "--model",
                str(artifacts["model"]),
                "--model-manifest",
                str(artifacts["model_manifest"]),
            ]
        )
        measurements["ask_ms"] = (time.perf_counter() - started) * 1000
        assert payload["outcome"] == "answered" and payload["answer"].strip()
        assert payload["generation_profile_id"] == "mvp-qwen3-8b-b10007-cuda-v1"
        assert payload["citation_labels"] and payload["citations"] and bundles_seen
        captured = bundles_seen[0]
        assert captured.fingerprint == inspected.bundle.fingerprint == payload["bundle_fingerprint"]
        entries = {e.citation_label: e for e in captured.entries}
        assert payload["citation_labels"] == [c["citation_label"] for c in payload["citations"]]
        for citation in payload["citations"]:
            entry = entries[citation["citation_label"]]
            ref = entry.source
            assert citation == dict(
                citation_label=entry.citation_label,
                paper_id="selected",
                file_version_id=ref.scope.file_version_id,
                document_generation_id=ref.scope.document_generation_id,
                physical_page_index=ref.physical_page_index,
                display_page_number=ref.display_page_number,
                printed_page_label=ref.printed_page_label,
                page_id=ref.parent.page_id,
                start_offset=ref.start_offset,
                end_offset=ref.end_offset,
                text_sha256=ref.text_sha256,
                anchor_ids=list(ref.anchor_ids),
            )
        assert calls["apply_template"] == 1 and calls["tokenize"] == 1 and len(admitted) == 1
        assert admitted[0] == payload["rendered_prompt_tokens"] and admitted[0] + 1024 <= 4096
        assert calls["source_recheck"] > before_source and calls["vector_recheck"] > before_vector
        assert release_calls["source"] >= 1 and release_calls["vector"] >= 1
        assert maps and all(store._closed for store in maps)
    finally:
        stop.set()
        sampler.join(timeout=5)
        assert not sampler.is_alive()
        for pid, born in owned.items():
            try:
                assert psutil.Process(pid).create_time() != born, "owned process survived cleanup"
            except psutil.NoSuchProcess:
                pass
        for port in ports:
            with socket.socket() as connection:
                assert connection.connect_ex(("127.0.0.1", port)) != 0
        assert all(lease.state == "released" for lease in leases)
        assert all(store._closed for store in maps)
    assert owned and ports and leases
    print(
        "V11_MEASUREMENTS "
        + json.dumps(
            dict(
                measurements=measurements,
                selected_pages=2,
                distractor_pages=64,
                total_chunks=len(rows),
                first_selected_lexical_rank=first_selected,
                old_hybrid_selected_hits=0,
                owned_processes=len(owned),
                closed_leases=len(leases),
                closed_memmaps=len(maps),
                exact_prompt_tokens=admitted[0],
                completion_tokens=payload["completion_tokens"],
                source_rechecks=calls["source_recheck"],
                vector_rechecks=calls["vector_recheck"],
                after_generation_release_checks=release_calls,
                peak_owned_rss_bytes=peak[0] if peak[0] and not memory_errors else None,
                rss_interval_ms=10,
                gpu_memory_included=False,
                python=platform.python_version(),
                machine=platform.machine(),
                cpu=platform.processor(),
                logical_cpus=psutil.cpu_count(),
                physical_memory_bytes=psutil.virtual_memory().total,
                runtime_manifest_sha256=runtime_manifest.manifest_sha256,
                model_manifest_sha256=model_manifest.manifest_sha256,
                embedding_artifact_set_sha256=verified.manifest.embedding_profile.artifact_set_sha256,
            ),
            sort_keys=True,
        )
    )
