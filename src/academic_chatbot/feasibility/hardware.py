from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import hmac
import io
import json
import locale
import os
import platform
import subprocess
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, Self, cast

import psutil  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_COMMAND_TIMEOUT_SECONDS = 10
_BACKGROUND_LOAD_POLICY = (
    "Close non-essential applications and record the complete application process tree."
)
_GPU_OFFLOAD_PENDING_DIAGNOSTIC = (
    "gpu_offload_available: requires pinned llama.cpp runtime binding"
)
_PROCESSOR_FEATURES = (
    (6, "SSE"),
    (10, "SSE2"),
    (13, "SSE3"),
    (36, "SSSE3"),
    (37, "SSE4.1"),
    (38, "SSE4.2"),
    (39, "AVX"),
    (40, "AVX2"),
    (41, "AVX512F"),
)
_PROCESSOR_FEATURE_ORDER = {
    name: position for position, (_, name) in enumerate(_PROCESSOR_FEATURES)
}


class _NativeFunction(Protocol):
    argtypes: list[object]
    restype: object

    def __call__(self, *args: object) -> object: ...


class _Guid(ctypes.Structure):
    _fields_ = (
        ("data1", ctypes.c_uint32),
        ("data2", ctypes.c_uint16),
        ("data3", ctypes.c_uint16),
        ("data4", ctypes.c_ubyte * 8),
    )


_FOLDERID_PROGRAM_FILES = _Guid(
    0x905E63B6,
    0xC1BF,
    0x494E,
    (ctypes.c_ubyte * 8)(0xB2, 0x9C, 0x65, 0xB7, 0x32, 0xD3, 0xD2, 0x1A),
)
_MAX_WINDOWS_DIRECTORY_CHARACTERS = 32_768


def _validated_existing_absolute_directory(candidate: Path) -> Path | None:
    if not candidate.is_absolute():
        return None
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_absolute() or not resolved.is_dir():
        return None
    return resolved


def _read_windows_directory(
    get_windows_directory: _NativeFunction,
) -> Path | None:
    try:
        buffer = ctypes.create_unicode_buffer(_MAX_WINDOWS_DIRECTORY_CHARACTERS)
        get_windows_directory.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
        get_windows_directory.restype = ctypes.c_uint
        length_value = get_windows_directory(
            buffer,
            _MAX_WINDOWS_DIRECTORY_CHARACTERS,
        )
        if isinstance(length_value, bool) or not isinstance(length_value, int):
            return None
        if (
            length_value == 0
            or length_value >= _MAX_WINDOWS_DIRECTORY_CHARACTERS
            or len(buffer.value) != length_value
        ):
            return None
        return _validated_existing_absolute_directory(Path(buffer.value))
    except (
        AttributeError,
        ctypes.ArgumentError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None


def _read_program_files_directory(
    get_known_folder: _NativeFunction,
    free_memory: _NativeFunction,
) -> Path | None:
    output = ctypes.c_wchar_p()
    candidate: Path | None = None
    cleanup_failed = False
    try:
        get_known_folder.argtypes = [
            ctypes.POINTER(_Guid),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_wchar_p),
        ]
        get_known_folder.restype = ctypes.c_long
        free_memory.argtypes = [ctypes.c_void_p]
        free_memory.restype = None
        result_value = get_known_folder(
            ctypes.byref(_FOLDERID_PROGRAM_FILES),
            0,
            None,
            ctypes.byref(output),
        )
        output_value = output.value
        if (
            not isinstance(result_value, bool)
            and isinstance(result_value, int)
            and result_value == 0
            and isinstance(output_value, str)
            and output_value
        ):
            candidate = _validated_existing_absolute_directory(Path(output_value))
    except (
        AttributeError,
        ctypes.ArgumentError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        candidate = None
    finally:
        if bool(output):
            try:
                free_memory(ctypes.cast(output, ctypes.c_void_p))
            except (
                AttributeError,
                ctypes.ArgumentError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                cleanup_failed = True
    if cleanup_failed:
        return None
    return candidate


def _authoritative_windows_directory() -> Path | None:
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_windows_directory = cast(
            _NativeFunction,
            kernel32.GetWindowsDirectoryW,
        )
    except (
        AttributeError,
        ctypes.ArgumentError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None
    return _read_windows_directory(get_windows_directory)


def _authoritative_program_files_directory() -> Path | None:
    if os.name != "nt":
        return None
    try:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        ole32 = ctypes.WinDLL("ole32", use_last_error=True)
        get_known_folder = cast(
            _NativeFunction,
            shell32.SHGetKnownFolderPath,
        )
        free_memory = cast(_NativeFunction, ole32.CoTaskMemFree)
    except (
        AttributeError,
        ctypes.ArgumentError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None
    return _read_program_files_directory(get_known_folder, free_memory)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MemoryModuleFact(_StrictFrozenModel):
    capacity_bytes: int | None = Field(default=None, gt=0)
    speed_mhz: int | None = Field(default=None, gt=0)
    manufacturer: str | None = None
    part_number: str | None = None
    bank_label: str | None = None
    device_locator: str | None = None


class HardwareFacts(_StrictFrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    cpu_model: str | None = None
    physical_cores: int | None = Field(default=None, gt=0)
    logical_cores: int | None = Field(default=None, gt=0)
    instruction_sets: tuple[str, ...] | None = None
    ram_bytes: int | None = Field(default=None, gt=0)
    usable_ram_bytes: int | None = Field(default=None, gt=0)
    ram_layout: tuple[MemoryModuleFact, ...] | None = None
    windows_build: str | None = None
    power_profile: str | None = None
    gpu_model: str | None = None
    vram_bytes: int | None = Field(default=None, ge=0)
    gpu_offload_available: bool | None = None
    storage_kind: str | None = None
    background_load_policy: str = _BACKGROUND_LOAD_POLICY
    collected_at: str | None = None
    collection_diagnostics: tuple[str, ...] = ()

    @field_validator("instruction_sets")
    @classmethod
    def _normalize_instruction_sets(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        unique = set(value)
        return tuple(
            sorted(
                unique,
                key=lambda item: (_PROCESSOR_FEATURE_ORDER.get(item, 10_000), item),
            )
        )

    @field_validator("ram_layout")
    @classmethod
    def _normalize_ram_layout(
        cls, value: tuple[MemoryModuleFact, ...] | None
    ) -> tuple[MemoryModuleFact, ...] | None:
        if value is None:
            return None
        return tuple(sorted(value, key=_memory_module_sort_key))

    @field_validator("collected_at")
    @classmethod
    def _validate_optional_collected_at(cls, value: str | None) -> str | None:
        return None if value is None else _validate_utc_timestamp(value)


class ReferenceHardwareRecord(_StrictFrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    cpu_model: str = Field(min_length=1)
    physical_cores: int = Field(gt=0)
    logical_cores: int = Field(gt=0)
    instruction_sets: tuple[str, ...] = Field(min_length=1)
    ram_bytes: int = Field(gt=0)
    usable_ram_bytes: int = Field(gt=0)
    ram_layout: tuple[MemoryModuleFact, ...] = Field(min_length=1)
    windows_build: str = Field(min_length=1)
    power_profile: str = Field(min_length=1)
    gpu_model: str | None
    vram_bytes: int | None = Field(default=None, ge=0)
    gpu_offload_available: bool
    storage_kind: str = Field(min_length=1)
    background_load_policy: str = Field(min_length=1)
    collection_diagnostics: tuple[str, ...]
    gguf_name: str = Field(min_length=1)
    gguf_sha256: str = Field(pattern=_SHA256_PATTERN)
    gguf_quantization: str = Field(min_length=1)
    model_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    llama_release: str = Field(min_length=1)
    llama_flags: tuple[str, ...] = Field(min_length=1)
    runtime_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    benchmark_corpus_sha256: str = Field(pattern=_SHA256_PATTERN)
    collected_at: str = Field(min_length=1)
    record_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("collected_at")
    @classmethod
    def _validate_collected_at(cls, value: str) -> str:
        return _validate_utc_timestamp(value)

    @field_validator("gguf_name")
    @classmethod
    def _validate_gguf_name(cls, value: str) -> str:
        if Path(value).name != value or Path(value).suffix.casefold() != ".gguf":
            raise ValueError("gguf_name must be a file name ending in .gguf")
        return value

    @model_validator(mode="after")
    def _validate_record_hash(self) -> Self:
        if self.ram_bytes != 16 * 1024**3:
            raise ValueError("ram_bytes must equal the exact 16 GiB reference target")
        if self.physical_cores > self.logical_cores:
            raise ValueError("physical_cores cannot exceed logical_cores")
        if self.usable_ram_bytes > self.ram_bytes:
            raise ValueError("usable_ram_bytes cannot exceed installed ram_bytes")
        module_capacities = tuple(module.capacity_bytes for module in self.ram_layout)
        if any(capacity is None for capacity in module_capacities):
            raise ValueError("ram_layout capacity_bytes must be complete")
        for module in self.ram_layout:
            if module.speed_mhz is None:
                raise ValueError("ram_layout speed_mhz must be complete")
            if module.bank_label is None:
                raise ValueError("ram_layout bank_label must be complete")
            if module.device_locator is None:
                raise ValueError("ram_layout device_locator must be complete")
        installed_from_modules = sum(
            capacity for capacity in module_capacities if capacity is not None
        )
        if installed_from_modules != self.ram_bytes:
            raise ValueError("ram_bytes must equal the complete ram_layout capacity sum")
        if self.gpu_model is None and self.gpu_offload_available:
            raise ValueError("gpu_offload_available cannot be true without a GPU")
        if self.gpu_model is not None and self.vram_bytes is None:
            raise ValueError("vram_bytes must be known when gpu_model is present")
        if any(
            item.startswith("gpu_offload_available:")
            for item in self.collection_diagnostics
        ):
            raise ValueError(
                "gpu_offload_available diagnostics must be resolved before binding"
            )
        payload = self.model_dump(mode="json", exclude={"record_sha256"})
        expected = canonical_sha256(payload)
        if not hmac.compare_digest(self.record_sha256, expected):
            raise ValueError("record_sha256 does not match the canonical record payload")
        return self


def _validate_utc_timestamp(value: str) -> str:
    if not value.endswith("Z"):
        raise ValueError("collected_at must be an ISO 8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError("collected_at must be a valid ISO 8601 UTC timestamp") from error
    if parsed.utcoffset() != UTC.utcoffset(None):
        raise ValueError("collected_at must use UTC")
    return value


def _memory_module_sort_key(module: MemoryModuleFact) -> tuple[object, ...]:
    return (
        module.device_locator or "",
        module.bank_label or "",
        module.capacity_bytes or 0,
        module.speed_mhz or 0,
        module.manufacturer or "",
        module.part_number or "",
    )


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def build_record(
    *,
    facts: HardwareFacts,
    gguf_name: str,
    gguf_sha256: str,
    gguf_quantization: str,
    model_manifest_sha256: str,
    llama_release: str,
    runtime_manifest_sha256: str,
    benchmark_corpus_sha256: str,
    gpu_offload_available: bool,
    collected_at: str,
    llama_flags: tuple[str, ...] = (),
) -> ReferenceHardwareRecord:
    fact_payload = facts.model_dump(mode="json")
    fact_payload.pop("collected_at", None)
    fact_payload["gpu_offload_available"] = gpu_offload_available
    fact_payload["collection_diagnostics"] = [
        item
        for item in facts.collection_diagnostics
        if item != _GPU_OFFLOAD_PENDING_DIAGNOSTIC
    ]
    payload: dict[str, object] = {
        **fact_payload,
        "gguf_name": gguf_name,
        "gguf_sha256": gguf_sha256,
        "gguf_quantization": gguf_quantization,
        "model_manifest_sha256": model_manifest_sha256,
        "llama_release": llama_release,
        "llama_flags": list(llama_flags),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "benchmark_corpus_sha256": benchmark_corpus_sha256,
        "collected_at": collected_at,
    }
    return ReferenceHardwareRecord.model_validate(
        {
            **payload,
            "record_sha256": canonical_sha256(payload),
        }
    )


def _write_canonical_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(payload))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_hardware_facts(path: Path, facts: HardwareFacts) -> None:
    _write_canonical_json(path, facts.model_dump(mode="json"))


def write_reference_hardware(path: Path, record: ReferenceHardwareRecord) -> None:
    validated = ReferenceHardwareRecord.model_validate(record.model_dump(mode="json"))
    _write_canonical_json(path, validated.model_dump(mode="json"))


def _clean_optional_text(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(
        value, (int, str, bytes, bytearray)
    ):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _run_fixed_command(
    arguments: Sequence[str],
    *,
    field_name: str,
    encoding: str | None = None,
) -> tuple[str | None, str | None]:
    raw_creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    creation_flags = raw_creation_flags if isinstance(raw_creation_flags, int) else 0
    try:
        completed = subprocess.run(
            list(arguments),
            check=False,
            capture_output=True,
            text=False,
            shell=False,
            timeout=_COMMAND_TIMEOUT_SECONDS,
            creationflags=creation_flags,
        )
    except FileNotFoundError:
        return None, f"{field_name}: probe command is unavailable"
    except subprocess.TimeoutExpired:
        return None, f"{field_name}: probe command timed out"
    except OSError:
        return None, f"{field_name}: probe command could not be started"

    if completed.returncode != 0:
        return None, f"{field_name}: probe command failed with exit {completed.returncode}"
    try:
        output = completed.stdout.decode(
            encoding or locale.getpreferredencoding(False), errors="strict"
        ).strip()
    except UnicodeDecodeError:
        return None, f"{field_name}: probe returned undecodable text"
    if not output:
        return None, f"{field_name}: probe returned no data"
    return output, None


def _trusted_existing_executable(
    candidate: Path, *, trusted_root: Path
) -> str | None:
    try:
        resolved_root = trusted_root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
        resolved_candidate.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    if not resolved_candidate.is_file():
        return None
    return str(resolved_candidate)


def _windows_system_executable(*relative_parts: str) -> str | None:
    system_root = _authoritative_windows_directory()
    if system_root is None:
        return None
    return _trusted_existing_executable(
        system_root.joinpath(*relative_parts),
        trusted_root=system_root,
    )


def _powershell_executable() -> str | None:
    return _windows_system_executable(
        "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
    )


def _run_fixed_powershell_json(
    script: str,
    *,
    field_name: str,
) -> tuple[Any | None, str | None]:
    executable = _powershell_executable()
    if executable is None:
        return None, f"{field_name}: PowerShell is unavailable"
    output, diagnostic = _run_fixed_command(
        (
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);"
            "$OutputEncoding=[Console]::OutputEncoding;"
            + script,
        ),
        field_name=field_name,
        encoding="utf-8",
    )
    if output is None:
        return None, diagnostic
    try:
        return json.loads(output), None
    except json.JSONDecodeError:
        return None, f"{field_name}: probe returned invalid JSON"


def _processor_feature_present(feature_id: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    probe = kernel32.IsProcessorFeaturePresent
    probe.argtypes = [ctypes.c_uint32]
    probe.restype = ctypes.c_int
    return bool(probe(feature_id))


def _collect_instruction_sets() -> tuple[tuple[str, ...] | None, str | None]:
    try:
        supported = tuple(
            name
            for feature_id, name in _PROCESSOR_FEATURES
            if _processor_feature_present(feature_id)
        )
    except (AttributeError, OSError):
        return None, "instruction_sets: Windows feature probe is unavailable"
    if not supported:
        return None, "instruction_sets: Windows reported no supported feature"
    return supported, None


def _objects(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _collect_memory_layout() -> tuple[tuple[MemoryModuleFact, ...] | None, str | None]:
    payload, diagnostic = _run_fixed_powershell_json(
        "$ErrorActionPreference='Stop';"
        "@(Get-CimInstance Win32_PhysicalMemory | "
        "Select-Object Capacity,Speed,Manufacturer,PartNumber,BankLabel,DeviceLocator) | "
        "ConvertTo-Json -Compress",
        field_name="ram_layout",
    )
    modules = tuple(
        MemoryModuleFact(
            capacity_bytes=_positive_int(item.get("Capacity")),
            speed_mhz=_positive_int(item.get("Speed")),
            manufacturer=_clean_optional_text(item.get("Manufacturer")),
            part_number=_clean_optional_text(item.get("PartNumber")),
            bank_label=_clean_optional_text(item.get("BankLabel")),
            device_locator=_clean_optional_text(item.get("DeviceLocator")),
        )
        for item in _objects(payload)
    )
    if not modules:
        return None, diagnostic or "ram_layout: probe returned no modules"
    return tuple(sorted(modules, key=_memory_module_sort_key)), diagnostic


def _nvidia_smi_executable() -> str | None:
    system_executable = _windows_system_executable("System32", "nvidia-smi.exe")
    if system_executable is not None:
        return system_executable
    program_files = _authoritative_program_files_directory()
    if program_files is None:
        return None
    return _trusted_existing_executable(
        program_files / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        trusted_root=program_files,
    )


def _collect_nvidia_gpu_facts() -> tuple[tuple[str, ...], tuple[int, ...], str | None]:
    executable = _nvidia_smi_executable()
    if executable is None:
        return (), (), "nvidia_smi: trusted executable is unavailable"
    output, diagnostic = _run_fixed_command(
        (
            executable,
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ),
        field_name="nvidia_smi",
        encoding="utf-8",
    )
    if output is None:
        return (), (), diagnostic

    names: list[str] = []
    memory_values: list[int] = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) < 2:
            continue
        name = _clean_optional_text(row[0])
        memory_mib = _positive_int(row[1].strip())
        if name is not None:
            names.append(name)
        if memory_mib is not None:
            memory_values.append(memory_mib * 1024**2)
    if not names:
        return (), (), "nvidia_smi: probe returned no adapters"
    return tuple(sorted(set(names))), tuple(memory_values), None


def _collect_gpu() -> tuple[str | None, int | None, str | None]:
    payload, wmi_diagnostic = _run_fixed_powershell_json(
        "$ErrorActionPreference='Stop';"
        "@(Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM) | ConvertTo-Json -Compress",
        field_name="gpu",
    )
    adapters = _objects(payload)
    wmi_names = tuple(
        name
        for item in adapters
        if (name := _clean_optional_text(item.get("Name"))) is not None
    )
    memory_values = tuple(
        value
        for item in adapters
        if (value := _positive_int(item.get("AdapterRAM"))) is not None
    )
    diagnostics = [wmi_diagnostic] if wmi_diagnostic is not None else []
    nvidia_names: tuple[str, ...] = ()
    if any("nvidia" in name.casefold() for name in wmi_names):
        nvidia_names, nvidia_memory, nvidia_diagnostic = _collect_nvidia_gpu_facts()
        memory_values += nvidia_memory
        if nvidia_diagnostic is not None:
            diagnostics.append(nvidia_diagnostic)

    names = tuple(sorted(set(wmi_names + nvidia_names)))
    if not names:
        diagnostic = "; ".join(diagnostics) or "gpu: probe returned no adapters"
        return None, None, diagnostic
    return "; ".join(names), max(memory_values, default=None), "; ".join(diagnostics) or None


def _collect_storage_kind() -> tuple[str | None, str | None]:
    payload, diagnostic = _run_fixed_powershell_json(
        "$ErrorActionPreference='Stop';"
        "@(Get-PhysicalDisk | Select-Object MediaType,BusType) | "
        "ConvertTo-Json -Compress",
        field_name="storage_kind",
    )
    kinds = sorted(
        {
            " / ".join(part for part in parts if part)
            for item in _objects(payload)
            if (
                parts := (
                    _clean_optional_text(item.get("MediaType")),
                    _clean_optional_text(item.get("BusType")),
                )
            )
            and any(parts)
        }
    )
    if not kinds:
        return None, diagnostic or "storage_kind: probe returned no disks"
    return "; ".join(kinds), diagnostic


def _collect_power_profile() -> tuple[str | None, str | None]:
    executable = _windows_system_executable("System32", "powercfg.exe")
    if executable is None:
        return None, "power_profile: trusted powercfg is unavailable"
    output, diagnostic = _run_fixed_command(
        (executable, "/GETACTIVESCHEME"),
        field_name="power_profile",
    )
    return _clean_optional_text(output), diagnostic


def _collect_core_count(*, logical: bool, field_name: str) -> tuple[int | None, str | None]:
    try:
        value = _positive_int(psutil.cpu_count(logical=logical))
    except (OSError, RuntimeError, TypeError, ValueError):
        return None, f"{field_name}: psutil probe failed"
    if value is None:
        return None, f"{field_name}: psutil returned no positive value"
    return value, None


def _collect_usable_ram_bytes() -> tuple[int | None, str | None]:
    try:
        value = _positive_int(psutil.virtual_memory().total)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None, "usable_ram_bytes: psutil probe failed"
    if value is None:
        return None, "usable_ram_bytes: psutil returned no positive value"
    return value, None


def collect_windows_hardware() -> HardwareFacts:
    if platform.system() != "Windows":
        raise RuntimeError("Hardware collection is supported only on Windows.")

    diagnostics: list[str] = []
    cpu_model = _clean_optional_text(platform.processor()) or _clean_optional_text(
        os.environ.get("PROCESSOR_IDENTIFIER")
    )
    if cpu_model is None:
        diagnostics.append("cpu_model: platform returned no data")

    instruction_sets, instruction_diagnostic = _collect_instruction_sets()
    if instruction_diagnostic is not None:
        diagnostics.append(instruction_diagnostic)

    physical_cores, physical_diagnostic = _collect_core_count(
        logical=False, field_name="physical_cores"
    )
    logical_cores, logical_diagnostic = _collect_core_count(
        logical=True, field_name="logical_cores"
    )
    if physical_diagnostic is not None:
        diagnostics.append(physical_diagnostic)
    if logical_diagnostic is not None:
        diagnostics.append(logical_diagnostic)

    usable_ram_bytes, usable_ram_diagnostic = _collect_usable_ram_bytes()
    if usable_ram_diagnostic is not None:
        diagnostics.append(usable_ram_diagnostic)

    ram_layout, ram_diagnostic = _collect_memory_layout()
    if ram_diagnostic is not None:
        diagnostics.append(ram_diagnostic)
    module_capacities = (
        tuple(module.capacity_bytes for module in ram_layout)
        if ram_layout is not None
        else ()
    )
    ram_bytes = (
        sum(capacity for capacity in module_capacities if capacity is not None)
        if module_capacities and all(capacity is not None for capacity in module_capacities)
        else None
    )
    if ram_bytes is None:
        diagnostics.append(
            "ram_bytes: installed capacity is unavailable; usable RAM is recorded separately"
        )

    gpu_model, vram_bytes, gpu_diagnostic = _collect_gpu()
    if gpu_diagnostic is not None:
        diagnostics.append(gpu_diagnostic)
    diagnostics.append(_GPU_OFFLOAD_PENDING_DIAGNOSTIC)

    storage_kind, storage_diagnostic = _collect_storage_kind()
    if storage_diagnostic is not None:
        diagnostics.append(storage_diagnostic)

    power_profile, power_diagnostic = _collect_power_profile()
    if power_diagnostic is not None:
        diagnostics.append(power_diagnostic)

    windows_build = _clean_optional_text(platform.version())
    if windows_build is None:
        diagnostics.append("windows_build: platform returned no data")

    return HardwareFacts(
        cpu_model=cpu_model,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        instruction_sets=instruction_sets,
        ram_bytes=ram_bytes,
        usable_ram_bytes=usable_ram_bytes,
        ram_layout=ram_layout,
        windows_build=windows_build,
        power_profile=power_profile,
        gpu_model=gpu_model,
        vram_bytes=vram_bytes,
        gpu_offload_available=None,
        storage_kind=storage_kind,
        collected_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        collection_diagnostics=tuple(diagnostics),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m academic_chatbot.feasibility.hardware",
        description="Collect canonical local Windows hardware facts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect", help="Collect hardware facts.")
    collect_parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "collect":
        facts = collect_windows_hardware()
        write_hardware_facts(arguments.output, facts)
        return 0
    raise RuntimeError("Unsupported hardware command.")


if __name__ == "__main__":
    raise SystemExit(main())
