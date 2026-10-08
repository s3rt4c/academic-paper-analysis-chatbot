from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from academic_chatbot.cli import main
from academic_chatbot.embeddings.profile import approved_bge_small_en_v15_profile
from tests.fixtures.evidence_bundle.pdfs import write_pdf

_RUN_ENV = "ACADEMIC_CHATBOT_RUN_MVP_LLAMA_CPP"
_PATH_ENV = {
    "embedding_model_root": "ACADEMIC_CHATBOT_MVP_EMBEDDING_MODEL_ROOT",
    "runtime_dir": "ACADEMIC_CHATBOT_MVP_LLAMA_CPP_RUNTIME_DIR",
    "runtime_manifest": "ACADEMIC_CHATBOT_MVP_LLAMA_CPP_RUNTIME_MANIFEST",
    "model": "ACADEMIC_CHATBOT_MVP_LLAMA_CPP_MODEL",
    "model_manifest": "ACADEMIC_CHATBOT_MVP_LLAMA_CPP_MODEL_MANIFEST",
}
_DIRECTORIES = {"embedding_model_root", "runtime_dir"}


def _live_paths() -> dict[str, Path]:
    if os.environ.get(_RUN_ENV) != "1":
        pytest.skip("live MVP artifacts are not enabled")
    missing = tuple(role for role, name in _PATH_ENV.items() if not os.environ.get(name))
    if missing:
        pytest.fail("opted-in live MVP artifact configuration is incomplete")
    paths = {role: Path(os.environ[name]) for role, name in _PATH_ENV.items()}
    if any(not path.is_absolute() for path in paths.values()):
        pytest.fail("opted-in live MVP artifact paths must be absolute")
    if any(
        not (path.is_dir() if role in _DIRECTORIES else path.is_file())
        for role, path in paths.items()
    ):
        pytest.fail("opted-in live MVP artifacts are missing or have the wrong kind")
    return paths


@pytest.mark.llama_cpp
def test_live_windows_single_paper_ask(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifacts = _live_paths()
    data_root = tmp_path / "data"
    source = write_pdf(
        tmp_path / "live-synthetic.pdf",
        text="The study objective is to evaluate a synthetic offline method.",
    )
    root = ["--data-root", str(data_root), "--max-pdf-bytes", "1000000"]
    assert main(
        [*root, "project", "create", "--project-id", "p", "--display-name", "P"]
    ) == 0
    assert main(
        [*root, "paper", "create", "--project-id", "p", "--paper-id", "paper"]
    ) == 0
    assert main(
        [
            *root,
            "import-pdf",
            "--project-id",
            "p",
            "--paper-id",
            "paper",
            "--source",
            str(source),
        ]
    ) == 0
    profile = approved_bge_small_en_v15_profile()
    assert main(
        [
            *root,
            "semantic-index",
            "build",
            "--project-id",
            "p",
            "--embedding-profile-id",
            profile.embedding_profile_id,
            "--model-root",
            str(artifacts["embedding_model_root"]),
        ]
    ) == 0
    capsys.readouterr()

    exit_code = main(
        [
            *root,
            "ask",
            "--project-id",
            "p",
            "--paper-id",
            "paper",
            "--question",
            "What is the study objective?",
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
    captured = capsys.readouterr()
    assert exit_code == 0, captured.out
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["outcome"] == "answered"
    assert payload["generation_profile_id"] == "mvp-qwen3-8b-b10007-cuda-v1"
    assert payload["answer"].strip()
    assert payload["citation_labels"]
    assert payload["citations"]
    assert all(citation["paper_id"] == "paper" for citation in payload["citations"])
