from __future__ import annotations

import os
from pathlib import Path

import pytest

from academic_chatbot.feasibility import llama_slice

_RUN_ENV = "ACADEMIC_CHATBOT_RUN_LLAMA_CPP"
_INPUT_ENV = {
    "cpu_runtime_directory": "ACADEMIC_CHATBOT_LLAMA_CPP_CPU_RUNTIME_DIR",
    "cpu_runtime_manifest": "ACADEMIC_CHATBOT_LLAMA_CPP_CPU_RUNTIME_MANIFEST",
    "selected_runtime_directory": "ACADEMIC_CHATBOT_LLAMA_CPP_CUDA_RUNTIME_DIR",
    "selected_runtime_manifest": "ACADEMIC_CHATBOT_LLAMA_CPP_CUDA_RUNTIME_MANIFEST",
    "model": "ACADEMIC_CHATBOT_LLAMA_CPP_MODEL",
    "model_manifest": "ACADEMIC_CHATBOT_LLAMA_CPP_MODEL_MANIFEST",
    "evidence_report": "ACADEMIC_CHATBOT_LLAMA_CPP_EVIDENCE_REPORT",
    "hardware_facts": "ACADEMIC_CHATBOT_LLAMA_CPP_HARDWARE_FACTS",
}
_DIRECTORY_INPUTS = {"cpu_runtime_directory", "selected_runtime_directory"}


def _require_opted_in_live_paths() -> dict[str, Path]:
    if os.environ.get(_RUN_ENV) != "1":
        pytest.skip(f"set {_RUN_ENV}=1 to run the pinned llama.cpp live probe")

    missing = tuple(name for name in _INPUT_ENV.values() if not os.environ.get(name))
    if missing:
        pytest.fail(
            "llama.cpp live probe was opted in but required environment paths are "
            f"missing: {', '.join(missing)}"
        )

    paths = {
        role: Path(os.environ[environment_name]).expanduser()
        for role, environment_name in _INPUT_ENV.items()
    }
    non_absolute = tuple(
        _INPUT_ENV[role] for role, path in paths.items() if not path.is_absolute()
    )
    if non_absolute:
        pytest.fail(
            "llama.cpp live probe environment paths must be absolute: "
            f"{', '.join(non_absolute)}"
        )

    mismatched = tuple(
        _INPUT_ENV[role]
        for role, path in paths.items()
        if not (path.is_dir() if role in _DIRECTORY_INPUTS else path.is_file())
    )
    if mismatched:
        pytest.fail(
            "llama.cpp live probe environment paths are missing or have the wrong kind: "
            f"{', '.join(mismatched)}"
        )
    return paths


@pytest.mark.llama_cpp
def test_pinned_llama_cpp_live_slice_writes_and_reloads_strict_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _require_opted_in_live_paths()
    output = tmp_path / "llama-slice-live.json"
    assert output.resolve(strict=False).parent == tmp_path.resolve(strict=True)
    assert not output.exists()

    cpu_manifest = llama_slice.load_llama_runtime_manifest(
        paths["cpu_runtime_manifest"]
    )
    selected_manifest = llama_slice.load_llama_runtime_manifest(
        paths["selected_runtime_manifest"]
    )
    model_manifest = llama_slice.load_gguf_model_manifest(paths["model_manifest"])
    if model_manifest.profile_id == llama_slice.DEFAULT_MODEL_PROFILE_ID:
        model_role = "default"
    elif model_manifest.profile_id == llama_slice.FALLBACK_MODEL_PROFILE_ID:
        model_role = "fallback"
    else:  # pragma: no cover - strict loader currently makes this unreachable
        pytest.fail("opted-in model manifest does not name a supported frozen profile")

    result = llama_slice.main(
        [
            "run",
            "--cpu-runtime-dir",
            os.fspath(paths["cpu_runtime_directory"]),
            "--cpu-runtime-manifest",
            os.fspath(paths["cpu_runtime_manifest"]),
            "--runtime-dir",
            os.fspath(paths["selected_runtime_directory"]),
            "--runtime-manifest",
            os.fspath(paths["selected_runtime_manifest"]),
            "--model",
            os.fspath(paths["model"]),
            "--model-manifest",
            os.fspath(paths["model_manifest"]),
            "--evidence-report",
            os.fspath(paths["evidence_report"]),
            "--hardware-facts",
            os.fspath(paths["hardware_facts"]),
            "--model-role",
            model_role,
            "--output",
            os.fspath(output),
        ]
    )
    captured = capsys.readouterr()
    assert result == 0, captured.err
    assert captured.out == ""
    assert captured.err == ""

    report = llama_slice.load_llama_slice_report(
        output,
        cpu_runtime_manifest=cpu_manifest,
        selected_runtime_manifest=selected_manifest,
        model_manifest=model_manifest,
    )
    assert report.model_role == model_role
    assert report.cpu_runtime_manifest_sha256 == cpu_manifest.manifest_sha256
    assert (
        report.selected_runtime_manifest_sha256
        == selected_manifest.manifest_sha256
    )
    assert report.model_manifest_sha256 == model_manifest.manifest_sha256
    assert output.parent == tmp_path
    assert tuple(tmp_path.glob(f".{output.name}.*.tmp")) == ()
