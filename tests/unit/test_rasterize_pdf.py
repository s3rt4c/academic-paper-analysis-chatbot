from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def rasterize_module(monkeypatch: pytest.MonkeyPatch):
    fake_pdf2image = types.ModuleType("pdf2image")
    fake_pdf2image.convert_from_path = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "pdf2image", fake_pdf2image)
    tool_path = Path(__file__).parents[2] / "tools" / "rasterize_pdf.py"
    spec = importlib.util.spec_from_file_location("public_rasterize_pdf", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rasterizer_uses_path_when_no_poppler_directory_is_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rasterize_module: object
) -> None:
    captured: dict[str, object] = {}

    def fake_convert(*args: object, **kwargs: object) -> list[Path]:
        captured.update(kwargs)
        return []

    monkeypatch.delenv("ACADEMIC_CHATBOT_POPPLER_PATH", raising=False)
    monkeypatch.setattr(rasterize_module, "convert_from_path", fake_convert)
    monkeypatch.setattr(sys, "argv", ["rasterize_pdf.py", "input.pdf", str(tmp_path / "out")])

    rasterize_module.main()

    assert captured["poppler_path"] is None


def test_rasterizer_rejects_a_configured_non_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rasterize_module: object
) -> None:
    missing = tmp_path / "missing-poppler"
    monkeypatch.setenv("ACADEMIC_CHATBOT_POPPLER_PATH", str(missing))
    monkeypatch.setattr(sys, "argv", ["rasterize_pdf.py", "input.pdf", str(tmp_path / "out")])

    with pytest.raises(SystemExit, match="ACADEMIC_CHATBOT_POPPLER_PATH"):
        rasterize_module.main()
