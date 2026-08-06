from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_FROZEN_PATHS = (
    "benchmarks/config/phase0.json",
    "artifacts/manifests/default-model.json",
    "benchmarks/results/llama-slice.json",
    "tests/fixtures/pdfs/native_anchor.pdf",
)


@pytest.mark.parametrize("relative_path", _FROZEN_PATHS)
def test_windows_checkout_filter_preserves_frozen_bytes(relative_path: str) -> None:
    committed = subprocess.run(
        ["git", "cat-file", "blob", f"HEAD:{relative_path}"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    filtered = subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "cat-file",
            "--filters",
            f"--path={relative_path}",
            f"HEAD:{relative_path}",
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
    ).stdout

    assert filtered == committed
