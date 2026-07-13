import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from academic_chatbot.feasibility import hardware
from academic_chatbot.feasibility.hardware import (
    HardwareFacts,
    MemoryModuleFact,
    ReferenceHardwareRecord,
    build_record,
    main,
)


def _sample_facts() -> HardwareFacts:
    return HardwareFacts(
        cpu_model="Test CPU",
        physical_cores=4,
        logical_cores=8,
        instruction_sets=("AVX", "AVX2"),
        ram_bytes=16 * 1024**3,
        usable_ram_bytes=15 * 1024**3,
        ram_layout=(
            MemoryModuleFact(
                capacity_bytes=16 * 1024**3,
                speed_mhz=5600,
                manufacturer="Test Memory",
                part_number="TEST-16G",
                bank_label="BANK 0",
                device_locator="DIMM 0",
            ),
        ),
        windows_build="10.0.test",
        power_profile="Balanced",
        gpu_model=None,
        vram_bytes=0,
        gpu_offload_available=None,
        storage_kind="NVMe SSD",
        collected_at="2026-07-11T00:00:00Z",
    )


def _sample_record() -> ReferenceHardwareRecord:
    return build_record(
        facts=_sample_facts(),
        gguf_name="research-model-Q4_K_M.gguf",
        gguf_sha256="1" * 64,
        gguf_quantization="Q4_K_M",
        model_manifest_sha256="2" * 64,
        llama_release="b-test",
        llama_flags=("--ctx-size", "4096", "--parallel", "1"),
        runtime_manifest_sha256="3" * 64,
        benchmark_corpus_sha256="4" * 64,
        gpu_offload_available=False,
        collected_at="2026-07-11T00:00:00Z",
    )


def test_reference_record_contains_exact_reproducibility_fields() -> None:
    record = _sample_record()

    assert record.ram_bytes == 16 * 1024**3
    assert record.usable_ram_bytes == 15 * 1024**3
    assert record.gguf_name == "research-model-Q4_K_M.gguf"
    assert record.gguf_sha256 == "1" * 64
    assert record.gguf_quantization == "Q4_K_M"
    assert record.model_manifest_sha256 == "2" * 64
    assert record.runtime_manifest_sha256 == "3" * 64
    assert record.gpu_offload_available is False
    assert len(record.record_sha256) == 64


def test_reference_record_hash_is_deterministic_and_content_bound() -> None:
    first = _sample_record()
    second = _sample_record()
    changed = build_record(
        facts=_sample_facts(),
        gguf_name="research-model-Q4_K_M.gguf",
        gguf_sha256="1" * 64,
        gguf_quantization="Q4_K_M",
        model_manifest_sha256="2" * 64,
        llama_release="b-test-other",
        llama_flags=("--ctx-size", "4096", "--parallel", "1"),
        runtime_manifest_sha256="3" * 64,
        benchmark_corpus_sha256="4" * 64,
        gpu_offload_available=False,
        collected_at="2026-07-11T00:00:00Z",
    )

    assert first.record_sha256 == second.record_sha256
    assert first.record_sha256 == "926639239ab7a7240e4efe40869b4f7d0b9c0f74700a238b49c2a8c81fa35d3d"
    assert first.record_sha256 != changed.record_sha256


def test_reference_record_rejects_tampering_and_unknown_fields() -> None:
    payload = _sample_record().model_dump(mode="json")
    payload["cpu_model"] = "Tampered CPU"
    with pytest.raises(ValidationError, match="record_sha256"):
        ReferenceHardwareRecord.model_validate(payload)

    facts_payload = _sample_facts().model_dump(mode="json")
    facts_payload["cpu_modle"] = "typo"
    with pytest.raises(ValidationError, match="Extra inputs"):
        HardwareFacts.model_validate(facts_payload)


def test_reference_models_are_frozen_and_validate_hashes() -> None:
    record = _sample_record()
    with pytest.raises(ValidationError, match="frozen"):
        record.ram_bytes = 1

    with pytest.raises(ValidationError, match="gguf_sha256"):
        build_record(
            facts=_sample_facts(),
            gguf_name="research-model-Q4_K_M.gguf",
            gguf_sha256="A" * 64,
            gguf_quantization="Q4_K_M",
            model_manifest_sha256="2" * 64,
            llama_release="b-test",
            runtime_manifest_sha256="3" * 64,
            benchmark_corpus_sha256="4" * 64,
            gpu_offload_available=False,
            collected_at="2026-07-11T00:00:00Z",
        )


def test_collect_command_writes_canonical_facts_without_final_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts = _sample_facts()
    monkeypatch.setattr(hardware, "collect_windows_hardware", lambda: facts)
    output = tmp_path / "hardware-facts.json"

    assert main(["collect", "--output", str(output)]) == 0

    raw = output.read_bytes()
    payload = json.loads(raw)
    expected = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert raw == expected
    assert payload["cpu_model"] == "Test CPU"
    assert "record_sha256" not in payload
    assert "gguf_sha256" not in payload
    assert list(tmp_path.glob(".hardware-facts.json.*.tmp")) == []


def test_collector_uses_installed_ram_and_records_usable_ram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = (
        MemoryModuleFact(
            capacity_bytes=8 * 1024**3,
            bank_label="BANK 1",
            device_locator="DIMM 1",
        ),
        MemoryModuleFact(
            capacity_bytes=8 * 1024**3,
            bank_label="BANK 0",
            device_locator="DIMM 0",
        ),
    )
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware.platform, "processor", lambda: "Test CPU")
    monkeypatch.setattr(hardware.platform, "version", lambda: "10.0.test")
    monkeypatch.setattr(hardware.psutil, "cpu_count", lambda logical: 8 if logical else 4)
    monkeypatch.setattr(
        hardware.psutil,
        "virtual_memory",
        lambda: type("Memory", (), {"total": 15 * 1024**3})(),
    )
    monkeypatch.setattr(hardware, "_collect_instruction_sets", lambda: (("AVX2",), None))
    monkeypatch.setattr(hardware, "_collect_memory_layout", lambda: (modules, None))
    monkeypatch.setattr(hardware, "_collect_gpu", lambda: (None, None, None))
    monkeypatch.setattr(hardware, "_collect_storage_kind", lambda: ("SSD / NVMe", None))
    monkeypatch.setattr(hardware, "_collect_power_profile", lambda: ("Balanced", None))

    facts = hardware.collect_windows_hardware()

    assert facts.ram_bytes == 16 * 1024**3
    assert facts.usable_ram_bytes == 15 * 1024**3
    assert facts.instruction_sets == ("AVX2",)
    assert facts.ram_layout is not None
    assert [module.device_locator for module in facts.ram_layout] == ["DIMM 0", "DIMM 1"]
    assert facts.gpu_offload_available is None
    assert any("runtime binding" in item for item in facts.collection_diagnostics)


def test_instruction_set_probe_has_stable_feature_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hardware,
        "_processor_feature_present",
        lambda feature_id: feature_id in {6, 10, 13, 37, 38, 39, 40},
    )

    instruction_sets, diagnostic = hardware._collect_instruction_sets()

    assert instruction_sets == ("SSE", "SSE2", "SSE3", "SSE4.1", "SSE4.2", "AVX", "AVX2")
    assert diagnostic is None


def test_atomic_writer_preserves_existing_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "hardware-facts.json"
    output.write_bytes(b"existing\n")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr(hardware.os, "replace", fail_replace)

    with pytest.raises(OSError, match="cannot replace"):
        hardware.write_hardware_facts(output, _sample_facts())

    assert output.read_bytes() == b"existing\n"
    assert list(tmp_path.glob(".hardware-facts.json.*.tmp")) == []


def test_fixed_command_fails_safely_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def timeout(arguments: list[str], **kwargs: object) -> None:
        captured["arguments"] = arguments
        captured.update(kwargs)
        raise hardware.subprocess.TimeoutExpired(arguments, 10)

    monkeypatch.setattr(hardware.subprocess, "run", timeout)

    output, diagnostic = hardware._run_fixed_command(
        (r"C:\Windows\System32\trusted.exe", "--probe"), field_name="test"
    )

    assert output is None
    assert diagnostic == "test: probe command timed out"
    assert captured["shell"] is False
    assert captured["timeout"] == 10


def test_non_windows_collection_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")

    with pytest.raises(RuntimeError, match="only on Windows"):
        hardware.collect_windows_hardware()


def test_nvidia_probe_overrides_truncated_wmi_vram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hardware,
        "_run_fixed_powershell_json",
        lambda script, *, field_name: (
            [
                {"Name": "NVIDIA Test GPU", "AdapterRAM": 4_293_918_720},
                {"Name": "AMD Test GPU", "AdapterRAM": 536_870_912},
            ],
            None,
        ),
    )
    monkeypatch.setattr(
        hardware,
        "_collect_nvidia_gpu_facts",
        lambda: (("NVIDIA Test GPU",), (8_188 * 1024**2,), None),
    )

    names, vram_bytes, diagnostic = hardware._collect_gpu()

    assert names == "AMD Test GPU; NVIDIA Test GPU"
    assert vram_bytes == 8_188 * 1024**2
    assert diagnostic is None


def test_writer_revalidates_constructed_reference_record(tmp_path: Path) -> None:
    payload = dict(_sample_record().__dict__)
    payload["cpu_model"] = "Tampered CPU"
    forged = ReferenceHardwareRecord.model_construct(**payload)

    with pytest.raises(ValidationError, match="record_sha256"):
        hardware.write_reference_hardware(tmp_path / "reference.json", forged)
