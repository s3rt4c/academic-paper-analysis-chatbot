from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from academic_chatbot.feasibility.hardware import (
    HardwareFacts,
    ReferenceHardwareRecord,
    canonical_sha256,
    collect_windows_hardware,
)
from academic_chatbot.feasibility.process_tree import (
    ProcessTreePeak,
    ProcessTreePeakSampler,
    evaluate_memory_gate,
)
from academic_chatbot.retrieval.exact_memmap import ExactVectorStore, VectorHit

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_REFERENCE_RAM_BYTES = 17_179_869_184
_WRONG_REFERENCE_RAM_SENTINEL = _REFERENCE_RAM_BYTES - 1
_PROCESS_TREE_LIMIT_BYTES = 12_884_901_888
_STABLE_HARDWARE_FIELDS = tuple(
    sorted(
        (
            "cpu_model",
            "physical_cores",
            "logical_cores",
            "instruction_sets",
            "ram_bytes",
            "usable_ram_bytes",
            "ram_layout",
            "windows_build",
            "power_profile",
            "gpu_model",
            "vram_bytes",
            "storage_kind",
            "background_load_policy",
        )
    )
)

MeasurementStatus = Literal["provisional", "bound"]
MemoryGateStatus = Literal[
    "passed",
    "failed",
    "not_evaluated_non_reference_hardware",
    "not_evaluated_invalid_measurement",
    "not_evaluated_wrong_reference_ram",
]


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_file_bytes(payload: Mapping[str, object]) -> bytes:
    return _canonical_json_bytes(payload) + b"\n"


def _canonical_payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _load_json_object(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{label} file could not be read.") from error
    try:
        payload: Any = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} file must contain valid UTF-8 JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} file must contain a JSON object.")
    stable_payload = cast(dict[str, object], payload)
    if raw != _canonical_json_file_bytes(stable_payload):
        raise ValueError(f"{label} file must be canonical UTF-8 JSON.")
    return stable_payload, raw


def _require_exact_validated_model_bytes(
    *,
    raw: bytes,
    model: BaseModel,
    label: str,
) -> None:
    validated_payload = cast(dict[str, object], model.model_dump(mode="json"))
    if raw != _canonical_json_file_bytes(validated_payload):
        raise ValueError(
            f"{label} file does not match its validated canonical model."
        )


class VectorBenchmarkProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    profile_id: Literal["phase0-exact-vector-v1"] = "phase0-exact-vector-v1"
    row_count: int = Field(default=100_000, gt=0)
    dimension: int = Field(default=384, gt=0)
    query_count: int = Field(default=200, gt=0)
    warmup_query_count: int = Field(default=10, ge=0)
    top_k: int = Field(default=10, gt=0)
    block_rows: int = Field(default=4096, gt=0)
    row_seed: int = 20_260_711
    query_seed: int = 20_260_712
    source_generator: Literal["numpy-pcg64-standard-normal-v1"] = (
        "numpy-pcg64-standard-normal-v1"
    )
    normalization_atol: float = Field(default=1e-5, gt=0.0)
    process_tree_sample_interval_ms: int = Field(default=50, gt=0)
    required_reference_ram_bytes: Literal[17_179_869_184] = 17_179_869_184
    peak_process_tree_limit_bytes: Literal[12_884_901_888] = (
        12_884_901_888
    )
    p95_threshold_ms: None = None
    top_k_agreement_threshold: None = None

    @model_validator(mode="after")
    def _validate_top_k(self) -> Self:
        if self.top_k > self.row_count:
            raise ValueError("top_k cannot exceed row_count")
        if self.row_seed == self.query_seed:
            raise ValueError("row_seed and query_seed must be independent")
        return self


def compute_vector_profile_sha256(profile: VectorBenchmarkProfile) -> str:
    validated = VectorBenchmarkProfile.model_validate(profile.model_dump(mode="json"))
    return _canonical_payload_sha256(validated.model_dump(mode="json"))


def load_vector_profile(path: Path) -> VectorBenchmarkProfile:
    payload, _ = _load_json_object(Path(path), label="Vector profile")
    if payload.get("schema_version") != "1.0.0":
        raise ValueError("Vector profile schema_version must equal 1.0.0.")
    vector_payload = payload.get("vector_exact")
    if not isinstance(vector_payload, dict):
        raise ValueError("Vector profile must contain a vector_exact object.")
    try:
        return VectorBenchmarkProfile.model_validate(vector_payload, strict=True)
    except ValidationError as error:
        raise ValueError("Vector profile vector_exact object is invalid.") from error


class RuntimeHardwareAttestation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    compared_fields: tuple[str, ...]
    runtime_hardware_facts_sha256: str = Field(pattern=_SHA256_PATTERN)
    matches_expected: bool
    mismatch_fields: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_attestation(self) -> Self:
        if self.compared_fields != _STABLE_HARDWARE_FIELDS:
            raise ValueError("compared_fields must contain the stable hardware identity")
        if self.mismatch_fields != tuple(sorted(set(self.mismatch_fields))):
            raise ValueError("mismatch_fields must be sorted and unique")
        if any(field not in self.compared_fields for field in self.mismatch_fields):
            raise ValueError("mismatch_fields must be a subset of compared_fields")
        if self.matches_expected == bool(self.mismatch_fields):
            raise ValueError("matches_expected must agree with mismatch_fields")
        return self


def attest_runtime_hardware(
    expected: HardwareFacts | ReferenceHardwareRecord,
    actual: HardwareFacts,
) -> RuntimeHardwareAttestation:
    if isinstance(expected, HardwareFacts):
        validated_expected: HardwareFacts | ReferenceHardwareRecord = (
            HardwareFacts.model_validate(expected.model_dump(mode="json"))
        )
    elif isinstance(expected, ReferenceHardwareRecord):
        validated_expected = ReferenceHardwareRecord.model_validate(
            expected.model_dump(mode="json")
        )
    else:
        raise TypeError("expected must be HardwareFacts or ReferenceHardwareRecord")
    validated_actual = HardwareFacts.model_validate(actual.model_dump(mode="json"))
    mismatch_fields = tuple(
        field
        for field in _STABLE_HARDWARE_FIELDS
        if getattr(validated_expected, field) != getattr(validated_actual, field)
    )
    actual_payload = cast(
        dict[str, object], validated_actual.model_dump(mode="json")
    )
    return RuntimeHardwareAttestation(
        compared_fields=_STABLE_HARDWARE_FIELDS,
        runtime_hardware_facts_sha256=canonical_sha256(actual_payload),
        matches_expected=not mismatch_fields,
        mismatch_fields=mismatch_fields,
    )


class LatencySummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_count: int = Field(gt=0)
    warmup_count: int = Field(ge=0)
    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)
    maximum_ms: float = Field(ge=0.0)
    clock_name: Literal["time.perf_counter_ns"] = "time.perf_counter_ns"
    percentile_method: Literal["linear"] = "linear"


class TopKAgreement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    oracle: Literal["float32_exhaustive_cosine"] = "float32_exhaustive_cosine"
    candidate: Literal["float16_exhaustive_dot_normalized_source"] = (
        "float16_exhaustive_dot_normalized_source"
    )
    top_k: int = Field(gt=0)
    mean_recall_at_k: float = Field(ge=0.0, le=1.0)
    minimum_recall_at_k: float = Field(ge=0.0, le=1.0)
    exact_order_match_rate: float = Field(ge=0.0, le=1.0)


class VectorReportStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    exact_16_gib_reference_verified: bool
    reference_profile_verified: bool
    correctness_status: Literal["passed", "failed"]
    memory_gate_status: MemoryGateStatus
    p95_gate_status: Literal["not_evaluated_no_approved_threshold"] = (
        "not_evaluated_no_approved_threshold"
    )
    agreement_gate_status: Literal["not_evaluated_no_approved_threshold"] = (
        "not_evaluated_no_approved_threshold"
    )
    gate_eligible: bool
    failure_reasons: tuple[str, ...]


def evaluate_report_status(
    *,
    profile: VectorBenchmarkProfile,
    measurement_status: MeasurementStatus,
    reference_ram_bytes: int | None,
    generation_integrity_verified: bool,
    deterministic_tie_break_verified: bool,
    runtime_hardware_match: bool,
    measurement_valid: bool,
    peak_bytes: int,
) -> VectorReportStatus:
    validated_profile = VectorBenchmarkProfile.model_validate(
        profile.model_dump(mode="json")
    )
    if measurement_status not in ("provisional", "bound"):
        raise ValueError("measurement_status must be provisional or bound")
    reference_profile_verified = validated_profile == VectorBenchmarkProfile()
    exact_reference = (
        measurement_status == "bound"
        and reference_ram_bytes == validated_profile.required_reference_ram_bytes
    )
    correctness_passed = (
        generation_integrity_verified and deterministic_tie_break_verified
    )

    if measurement_status == "provisional":
        memory_status: MemoryGateStatus = "not_evaluated_non_reference_hardware"
    elif not exact_reference:
        memory_status = "not_evaluated_wrong_reference_ram"
    elif not measurement_valid:
        memory_status = "not_evaluated_invalid_measurement"
    else:
        memory_status = evaluate_memory_gate(
            peak_bytes,
            limit_bytes=validated_profile.peak_process_tree_limit_bytes,
        )

    gate_eligible = (
        measurement_status == "bound"
        and reference_profile_verified
        and exact_reference
        and runtime_hardware_match
        and correctness_passed
        and memory_status == "passed"
    )
    failure_reasons: list[str] = []
    if not reference_profile_verified:
        failure_reasons.append(
            "Benchmark profile does not match the committed reference profile."
        )
    if measurement_status == "bound" and not exact_reference:
        failure_reasons.append(
            "Reference hardware RAM does not equal the exact 16 GiB target."
        )
    if not runtime_hardware_match:
        failure_reasons.append("Runtime hardware attestation failed.")
    if not generation_integrity_verified:
        failure_reasons.append("Vector generation integrity verification failed.")
    if not deterministic_tie_break_verified:
        failure_reasons.append(
            "Deterministic exact-search correctness verification failed."
        )
    if not measurement_valid:
        failure_reasons.append("Process-tree memory measurement is invalid.")
    if memory_status == "failed":
        failure_reasons.append(
            "Peak process-tree memory is not strictly below the approved limit."
        )
    return VectorReportStatus(
        exact_16_gib_reference_verified=exact_reference,
        reference_profile_verified=reference_profile_verified,
        correctness_status="passed" if correctness_passed else "failed",
        memory_gate_status=memory_status,
        gate_eligible=gate_eligible,
        failure_reasons=tuple(failure_reasons),
    )


class VectorExactReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    report_type: Literal["vector_exact"] = "vector_exact"
    measurement_status: MeasurementStatus
    measured_at_utc: str = Field(min_length=1)
    profile: VectorBenchmarkProfile
    profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    python_version: str = Field(min_length=1)
    numpy_version: str = Field(min_length=1)
    hardware_source_kind: Literal["hardware_facts", "reference_hardware"]
    hardware_payload_sha256: str = Field(pattern=_SHA256_PATTERN)
    reference_hardware_record_sha256: str | None = Field(
        default=None, pattern=_SHA256_PATTERN
    )
    runtime_hardware_facts_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_hardware_match: Literal[True]
    runtime_hardware_mismatch_fields: tuple[str, ...]
    exact_16_gib_reference_verified: bool
    reference_profile_verified: bool
    generation_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    generation_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    row_count: int = Field(gt=0)
    dimension: int = Field(gt=0)
    dtype: Literal["float16"] = "float16"
    source_float32_bytes: int = Field(gt=0)
    vector_payload_bytes: int = Field(gt=0)
    vectors_file_bytes: int = Field(gt=0)
    query_count: int = Field(gt=0)
    warmup_query_count: int = Field(ge=0)
    top_k: int = Field(gt=0)
    block_rows: int = Field(gt=0)
    search_latency_ms: LatencySummary
    float16_vs_float32: TopKAgreement
    process_tree_peak: ProcessTreePeak
    generation_integrity_verified: bool
    deterministic_tie_break_verified: bool
    correctness_status: Literal["passed", "failed"]
    memory_gate_status: MemoryGateStatus
    p95_gate_status: Literal["not_evaluated_no_approved_threshold"]
    agreement_gate_status: Literal["not_evaluated_no_approved_threshold"]
    gate_eligible: bool
    failure_reasons: tuple[str, ...]
    report_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_report(self) -> Self:
        unsigned = self.model_dump(mode="json", exclude={"report_sha256"})
        expected_hash = _canonical_payload_sha256(unsigned)
        if not hmac.compare_digest(self.report_sha256, expected_hash):
            raise ValueError("report_sha256 does not match the canonical report payload")
        if not self.measured_at_utc.endswith("Z"):
            raise ValueError("measured_at_utc must be an ISO 8601 UTC timestamp")
        try:
            measured_at = datetime.fromisoformat(
                self.measured_at_utc.removesuffix("Z") + "+00:00"
            )
        except ValueError as error:
            raise ValueError(
                "measured_at_utc must be an ISO 8601 UTC timestamp"
            ) from error
        if measured_at.utcoffset() != UTC.utcoffset(None):
            raise ValueError("measured_at_utc must use UTC")
        if not hmac.compare_digest(
            self.profile_sha256, compute_vector_profile_sha256(self.profile)
        ):
            raise ValueError(
                "profile_sha256 does not match the canonical vector profile"
            )
        if self.reference_profile_verified != (
            self.profile == VectorBenchmarkProfile()
        ):
            raise ValueError("reference_profile_verified is inconsistent with profile")
        if (self.row_count, self.dimension) != (
            self.profile.row_count,
            self.profile.dimension,
        ):
            raise ValueError("report dimensions are inconsistent with profile")
        if (
            self.query_count,
            self.warmup_query_count,
            self.top_k,
            self.block_rows,
        ) != (
            self.profile.query_count,
            self.profile.warmup_query_count,
            self.profile.top_k,
            self.profile.block_rows,
        ):
            raise ValueError("report query settings are inconsistent with profile")
        if self.source_float32_bytes != self.row_count * self.dimension * 4:
            raise ValueError("source_float32_bytes is inconsistent with report dimensions")
        if self.vector_payload_bytes != self.row_count * self.dimension * 2:
            raise ValueError("vector_payload_bytes is inconsistent with report dimensions")
        if self.search_latency_ms.sample_count != self.query_count:
            raise ValueError("latency sample count must equal query_count")
        if self.search_latency_ms.warmup_count != self.warmup_query_count:
            raise ValueError("latency warmup count must equal warmup_query_count")
        if self.float16_vs_float32.top_k != self.top_k:
            raise ValueError("agreement top_k must equal report top_k")
        if not self.runtime_hardware_match or self.runtime_hardware_mismatch_fields:
            raise ValueError("successful reports require matching runtime hardware")
        if self.measurement_status == "provisional":
            if self.hardware_source_kind != "hardware_facts":
                raise ValueError("provisional reports require hardware_facts")
            if self.reference_hardware_record_sha256 is not None:
                raise ValueError("provisional reports cannot bind reference hardware")
            reference_ram_bytes: int | None = None
        else:
            if self.hardware_source_kind != "reference_hardware":
                raise ValueError("bound reports require reference_hardware")
            if self.reference_hardware_record_sha256 is None:
                raise ValueError("bound reports require a reference record hash")
            reference_ram_bytes = (
                self.profile.required_reference_ram_bytes
                if self.exact_16_gib_reference_verified
                else _WRONG_REFERENCE_RAM_SENTINEL
            )

        derived_status = evaluate_report_status(
            profile=self.profile,
            measurement_status=self.measurement_status,
            reference_ram_bytes=reference_ram_bytes,
            generation_integrity_verified=self.generation_integrity_verified,
            deterministic_tie_break_verified=(
                self.deterministic_tie_break_verified
            ),
            runtime_hardware_match=self.runtime_hardware_match,
            measurement_valid=self.process_tree_peak.measurement_valid,
            peak_bytes=self.process_tree_peak.peak_bytes,
        )
        recorded_status = VectorReportStatus(
            exact_16_gib_reference_verified=(
                self.exact_16_gib_reference_verified
            ),
            reference_profile_verified=self.reference_profile_verified,
            correctness_status=self.correctness_status,
            memory_gate_status=self.memory_gate_status,
            p95_gate_status=self.p95_gate_status,
            agreement_gate_status=self.agreement_gate_status,
            gate_eligible=self.gate_eligible,
            failure_reasons=self.failure_reasons,
        )
        if recorded_status != derived_status:
            raise ValueError(
                "Report status fields do not match the derived report status."
            )
        return self


class _Sampler(Protocol):
    @property
    def result(self) -> ProcessTreePeak: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> object: ...


def _generate_normalized_matrix(
    row_count: int,
    dimension: int,
    *,
    seed: int,
) -> NDArray[np.float32]:
    generator = np.random.Generator(np.random.PCG64(seed))
    matrix = generator.standard_normal((row_count, dimension)).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if bool(np.any(norms == np.float32(0.0))):
        raise RuntimeError("Synthetic vector generation produced a zero vector.")
    matrix /= norms
    return matrix


def _oracle_top_k_rows(
    rows: NDArray[np.float32],
    query: NDArray[np.float32],
    *,
    top_k: int,
) -> tuple[int, ...]:
    scores = np.einsum(
        "ij,j->i",
        rows,
        query,
        dtype=np.float32,
        optimize=False,
    )
    vector_rows = np.arange(rows.shape[0], dtype=np.int64)
    ordering = np.lexsort((vector_rows, -scores))
    result = tuple(int(value) for value in ordering[:top_k])
    del ordering
    del vector_rows
    del scores
    return result


def _summarize_latency(
    latencies_ms: Sequence[float], *, warmup_count: int
) -> LatencySummary:
    samples = np.asarray(latencies_ms, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("At least one measured latency sample is required.")
    return LatencySummary(
        sample_count=int(samples.size),
        warmup_count=warmup_count,
        p50_ms=float(np.percentile(samples, 50, method="linear")),
        p95_ms=float(np.percentile(samples, 95, method="linear")),
        maximum_ms=float(np.max(samples)),
    )


def _summarize_agreement(
    candidate_hits: Sequence[tuple[VectorHit, ...]],
    oracle_rows: Sequence[tuple[int, ...]],
    *,
    top_k: int,
) -> TopKAgreement:
    if len(candidate_hits) != len(oracle_rows) or not candidate_hits:
        raise ValueError("Candidate and oracle results must have equal non-zero counts.")
    recalls: list[float] = []
    exact_matches = 0
    for candidate, oracle in zip(candidate_hits, oracle_rows, strict=True):
        candidate_rows = tuple(hit.vector_row for hit in candidate)
        recalls.append(len(set(candidate_rows).intersection(oracle)) / top_k)
        exact_matches += int(candidate_rows == oracle)
    return TopKAgreement(
        top_k=top_k,
        mean_recall_at_k=float(np.mean(np.asarray(recalls, dtype=np.float64))),
        minimum_recall_at_k=float(min(recalls)),
        exact_order_match_rate=exact_matches / len(candidate_hits),
    )


def _validated_hardware_source(
    source: HardwareFacts | ReferenceHardwareRecord,
) -> HardwareFacts | ReferenceHardwareRecord:
    if isinstance(source, HardwareFacts):
        return HardwareFacts.model_validate(source.model_dump(mode="json"))
    if isinstance(source, ReferenceHardwareRecord):
        return ReferenceHardwareRecord.model_validate(source.model_dump(mode="json"))
    raise TypeError("hardware_source must be HardwareFacts or ReferenceHardwareRecord")


def _ensure_fresh_workspace(workspace: Path) -> None:
    if workspace.exists() and not workspace.is_dir():
        raise ValueError("Benchmark workspace must be a directory.")
    generations = workspace / "generations"
    if generations.exists() and any(generations.iterdir()):
        raise ValueError(
            "Benchmark workspace already contains a published vector generation."
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_benchmark(
    *,
    profile: VectorBenchmarkProfile,
    hardware_source: HardwareFacts | ReferenceHardwareRecord,
    workspace: Path,
    hardware_collector: Callable[[], HardwareFacts] = collect_windows_hardware,
    sampler_factory: Callable[[int], _Sampler] = ProcessTreePeakSampler,
) -> VectorExactReport:
    validated_profile = VectorBenchmarkProfile.model_validate(
        profile.model_dump(mode="json")
    )
    validated_source = _validated_hardware_source(hardware_source)

    collected = hardware_collector()
    if not isinstance(collected, HardwareFacts):
        raise TypeError("hardware_collector must return HardwareFacts")
    runtime_facts = HardwareFacts.model_validate(collected.model_dump(mode="json"))
    attestation = attest_runtime_hardware(validated_source, runtime_facts)
    if not attestation.matches_expected:
        fields = ", ".join(attestation.mismatch_fields)
        raise RuntimeError(f"Runtime hardware does not match expected fields: {fields}.")

    stable_workspace = Path(workspace)
    _ensure_fresh_workspace(stable_workspace)
    sampler = sampler_factory(validated_profile.process_tree_sample_interval_ms)
    with sampler:
        rows = _generate_normalized_matrix(
            validated_profile.row_count,
            validated_profile.dimension,
            seed=validated_profile.row_seed,
        )
        all_queries = _generate_normalized_matrix(
            validated_profile.query_count + validated_profile.warmup_query_count,
            validated_profile.dimension,
            seed=validated_profile.query_seed,
        )
        warmup_queries = all_queries[: validated_profile.warmup_query_count]
        measured_queries = all_queries[validated_profile.warmup_query_count :]
        row_ids = tuple(
            f"benchmark-span-{vector_row:09d}"
            for vector_row in range(validated_profile.row_count)
        )
        profile_sha256 = compute_vector_profile_sha256(validated_profile)
        candidate_hits: list[tuple[VectorHit, ...]] = []
        latencies_ms: list[float] = []
        generation_integrity_verified = False
        deterministic_tie_break_verified = False

        store = ExactVectorStore.build(
            stable_workspace,
            rows=rows,
            row_ids=row_ids,
            profile_sha256=profile_sha256,
            normalization_atol=validated_profile.normalization_atol,
        )
        manifest = store.manifest
        generation_dir = store.generation_dir
        try:
            for query in warmup_queries:
                store.search(
                    query,
                    limit=validated_profile.top_k,
                    block_rows=validated_profile.block_rows,
                )
            for query in measured_queries:
                started_at = time.perf_counter_ns()
                hits = store.search(
                    query,
                    limit=validated_profile.top_k,
                    block_rows=validated_profile.block_rows,
                )
                elapsed_ns = time.perf_counter_ns() - started_at
                candidate_hits.append(hits)
                latencies_ms.append(elapsed_ns / 1_000_000)
            verification_block_rows = max(1, validated_profile.block_rows // 2)
            alternate_hits = store.search(
                measured_queries[0],
                limit=validated_profile.top_k,
                block_rows=verification_block_rows,
            )
            deterministic_tie_break_verified = alternate_hits == candidate_hits[0]
        finally:
            store.close()

        oracle_rows = [
            _oracle_top_k_rows(rows, query, top_k=validated_profile.top_k)
            for query in measured_queries
        ]
        latency_summary = _summarize_latency(
            latencies_ms,
            warmup_count=validated_profile.warmup_query_count,
        )
        agreement = _summarize_agreement(
            candidate_hits,
            oracle_rows,
            top_k=validated_profile.top_k,
        )
        with ExactVectorStore.open(generation_dir) as verified_store:
            generation_integrity_verified = verified_store.manifest == manifest

    peak = sampler.result
    measurement_status: MeasurementStatus
    reference_ram_bytes: int | None
    reference_record_sha256: str | None
    if isinstance(validated_source, ReferenceHardwareRecord):
        measurement_status = "bound"
        hardware_source_kind: Literal["hardware_facts", "reference_hardware"] = (
            "reference_hardware"
        )
        reference_ram_bytes = validated_source.ram_bytes
        reference_record_sha256 = validated_source.record_sha256
    else:
        measurement_status = "provisional"
        hardware_source_kind = "hardware_facts"
        reference_ram_bytes = None
        reference_record_sha256 = None

    status = evaluate_report_status(
        profile=validated_profile,
        measurement_status=measurement_status,
        reference_ram_bytes=reference_ram_bytes,
        generation_integrity_verified=generation_integrity_verified,
        deterministic_tie_break_verified=deterministic_tie_break_verified,
        runtime_hardware_match=attestation.matches_expected,
        measurement_valid=peak.measurement_valid,
        peak_bytes=peak.peak_bytes,
    )
    hardware_payload = cast(
        dict[str, object], validated_source.model_dump(mode="json")
    )
    report_payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "report_type": "vector_exact",
        "measurement_status": measurement_status,
        "measured_at_utc": _utc_now(),
        "profile": validated_profile.model_dump(mode="json"),
        "profile_sha256": profile_sha256,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "hardware_source_kind": hardware_source_kind,
        "hardware_payload_sha256": canonical_sha256(hardware_payload),
        "reference_hardware_record_sha256": reference_record_sha256,
        "runtime_hardware_facts_sha256": (
            attestation.runtime_hardware_facts_sha256
        ),
        "runtime_hardware_match": True,
        "runtime_hardware_mismatch_fields": [],
        "exact_16_gib_reference_verified": (
            status.exact_16_gib_reference_verified
        ),
        "reference_profile_verified": status.reference_profile_verified,
        "generation_id": manifest.generation_id,
        "generation_manifest_sha256": manifest.manifest_sha256,
        "row_count": validated_profile.row_count,
        "dimension": validated_profile.dimension,
        "dtype": "float16",
        "source_float32_bytes": validated_profile.row_count
        * validated_profile.dimension
        * 4,
        "vector_payload_bytes": manifest.vector_payload_bytes,
        "vectors_file_bytes": manifest.vectors_file_bytes,
        "query_count": validated_profile.query_count,
        "warmup_query_count": validated_profile.warmup_query_count,
        "top_k": validated_profile.top_k,
        "block_rows": validated_profile.block_rows,
        "search_latency_ms": latency_summary.model_dump(mode="json"),
        "float16_vs_float32": agreement.model_dump(mode="json"),
        "process_tree_peak": peak.model_dump(mode="json"),
        "generation_integrity_verified": generation_integrity_verified,
        "deterministic_tie_break_verified": deterministic_tie_break_verified,
        "correctness_status": status.correctness_status,
        "memory_gate_status": status.memory_gate_status,
        "p95_gate_status": status.p95_gate_status,
        "agreement_gate_status": status.agreement_gate_status,
        "gate_eligible": status.gate_eligible,
        "failure_reasons": list(status.failure_reasons),
    }
    report_payload["report_sha256"] = _canonical_payload_sha256(report_payload)
    return VectorExactReport.model_validate(report_payload)


def load_vector_report(path: Path) -> VectorExactReport:
    payload, raw = _load_json_object(Path(path), label="Vector exact report")
    provided_hash = payload.get("report_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "report_sha256"
    }
    expected_hash = _canonical_payload_sha256(unsigned)
    if not isinstance(provided_hash, str) or not hmac.compare_digest(
        provided_hash, expected_hash
    ):
        raise ValueError(
            "report_sha256 does not match the raw canonical report payload."
        )
    validated = VectorExactReport.model_validate(payload)
    _require_exact_validated_model_bytes(
        raw=raw,
        model=validated,
        label="Vector exact report",
    )
    return validated


def write_vector_report(path: Path, report: VectorExactReport) -> None:
    validated = VectorExactReport.model_validate(report.model_dump(mode="json"))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                _canonical_json_file_bytes(validated.model_dump(mode="json"))
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_hardware_facts(path: Path) -> HardwareFacts:
    payload, raw = _load_json_object(path, label="Hardware facts")
    try:
        validated = HardwareFacts.model_validate(payload)
    except ValidationError as error:
        raise ValueError("Hardware facts file is invalid.") from error
    _require_exact_validated_model_bytes(
        raw=raw,
        model=validated,
        label="Hardware facts",
    )
    return validated


def _load_reference_hardware(path: Path) -> ReferenceHardwareRecord:
    payload, raw = _load_json_object(path, label="Reference hardware")
    try:
        validated = ReferenceHardwareRecord.model_validate(payload)
    except ValidationError as error:
        raise ValueError("Reference hardware file is invalid.") from error
    _require_exact_validated_model_bytes(
        raw=raw,
        model=validated,
        label="Reference hardware",
    )
    return validated


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m academic_chatbot.feasibility.exact_vector"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser(
        "benchmark", help="Run the exhaustive exact-vector feasibility benchmark."
    )
    benchmark.add_argument("--config", type=Path, required=True)
    hardware = benchmark.add_mutually_exclusive_group(required=True)
    hardware.add_argument("--hardware-facts", type=Path)
    hardware.add_argument("--reference-hardware", type=Path)
    benchmark.add_argument("--workspace", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 2
    try:
        profile = load_vector_profile(arguments.config)
        if arguments.hardware_facts is not None:
            hardware_source: HardwareFacts | ReferenceHardwareRecord = (
                _load_hardware_facts(arguments.hardware_facts)
            )
        else:
            hardware_source = _load_reference_hardware(arguments.reference_hardware)
        report = run_benchmark(
            profile=profile,
            hardware_source=hardware_source,
            workspace=arguments.workspace,
            hardware_collector=collect_windows_hardware,
            sampler_factory=ProcessTreePeakSampler,
        )
        write_vector_report(arguments.output, report)
        if report.correctness_status == "failed":
            print(
                "Exact-vector benchmark correctness verification failed.",
                file=sys.stderr,
            )
            return 1
        if not report.process_tree_peak.measurement_valid:
            print(
                "Exact-vector process-tree memory measurement is invalid.",
                file=sys.stderr,
            )
            return 1
    except (OSError, TypeError, ValueError, RuntimeError, ValidationError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
