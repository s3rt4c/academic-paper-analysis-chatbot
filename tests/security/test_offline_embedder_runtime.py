from __future__ import annotations

from pathlib import Path


def test_embedding_runtime_has_no_hub_download_cache_or_subprocess_imports() -> None:
    root = Path(__file__).parents[2] / "src" / "academic_chatbot" / "embeddings"
    production_source = "\n".join(
        (
            (root / "tokenizer.py").read_text(encoding="utf-8"),
            (root / "embedder.py").read_text(encoding="utf-8"),
        )
    )
    for forbidden in (
        "huggingface_hub",
        "snapshot_download",
        "hf_hub_download",
        "from_pretrained",
        "subprocess",
        "requests",
        "httpx",
    ):
        assert forbidden not in production_source
