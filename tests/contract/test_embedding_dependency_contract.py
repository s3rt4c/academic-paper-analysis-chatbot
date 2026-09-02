from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers


def test_tokenizers_loads_a_local_tiny_artifact_without_network(tmp_path: Path) -> None:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "evidence": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    loaded = Tokenizer.from_file(str(tokenizer_path))

    assert loaded.encode("evidence").ids == [1]
