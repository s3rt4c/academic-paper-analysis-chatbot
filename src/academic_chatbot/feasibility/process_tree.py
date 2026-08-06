from __future__ import annotations

import math
import os
import platform
import threading
import time
from collections.abc import Callable
from typing import Literal, Protocol, Self, cast

import psutil  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

_DEFAULT_MEMORY_LIMIT_BYTES = 12_884_901_888

ProcessTreeMetric = Literal[
    "process_tree_sum_uss_bytes",
    "process_tree_sum_rss_bytes",
]
MemoryGateStatus = Literal["passed", "failed"]


class _MemoryInfo(Protocol):
    rss: int


class _FullMemoryInfo(Protocol):
    uss: int


class _ProcessLike(Protocol):
    pid: int

    def children(self, *, recursive: bool) -> list[_ProcessLike]: ...

    def is_running(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> int | None: ...

    def memory_full_info(self) -> _FullMemoryInfo: ...

    def memory_info(self) -> _MemoryInfo: ...


_ProcessFactory = Callable[[int], _ProcessLike]
_Clock = Callable[[], float]
_Wait = Callable[[threading.Event, float], bool]


class ProcessTreePeak(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: ProcessTreeMetric
    peak_bytes: int = Field(gt=0)
    sample_interval_ms: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    process_churn_count: int = Field(ge=0)
    access_error_count: int = Field(ge=0)
    measurement_valid: bool

    @model_validator(mode="after")
    def _validate_measurement_state(self) -> Self:
        if self.measurement_valid and self.access_error_count != 0:
            raise ValueError(
                "measurement_valid cannot be true when access_error_count is non-zero"
            )
        return self


def _default_process_factory(pid: int) -> _ProcessLike:
    return cast(_ProcessLike, psutil.Process(pid))


def _default_wait(stop_event: threading.Event, timeout: float) -> bool:
    return stop_event.wait(timeout)


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return value


def evaluate_memory_gate(
    peak_bytes: int,
    *,
    limit_bytes: int = _DEFAULT_MEMORY_LIMIT_BYTES,
) -> MemoryGateStatus:
    validated_peak = _positive_integer(peak_bytes, field_name="peak_bytes")
    validated_limit = _positive_integer(limit_bytes, field_name="limit_bytes")
    return "passed" if validated_peak < validated_limit else "failed"


class ProcessTreePeakSampler:
    def __init__(
        self,
        sample_interval_ms: int,
        *,
        root_pid: int | None = None,
        process_factory: _ProcessFactory = _default_process_factory,
        platform_system: Callable[[], str] = platform.system,
        clock: _Clock = time.monotonic,
        wait: _Wait = _default_wait,
    ) -> None:
        if isinstance(sample_interval_ms, bool) or sample_interval_ms <= 0:
            raise ValueError("sample_interval_ms must be positive")
        if root_pid is not None and (isinstance(root_pid, bool) or root_pid <= 0):
            raise ValueError("root_pid must be positive")

        self._sample_interval_ms = sample_interval_ms
        self._sample_interval_seconds = sample_interval_ms / 1_000
        self._root_pid = os.getpid() if root_pid is None else root_pid
        self._process_factory = process_factory
        self._platform_system = platform_system
        self._clock = clock
        self._wait = wait

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._root_process: _ProcessLike | None = None
        self._metric: ProcessTreeMetric | None = None
        self._peak_bytes = 0
        self._sample_count = 0
        self._process_churn_count = 0
        self._access_error_count = 0
        self._measurement_valid = True
        self._started = False
        self._finished = False
        self._result: ProcessTreePeak | None = None
        self._worker_error: BaseException | None = None

    @property
    def result(self) -> ProcessTreePeak:
        with self._lock:
            if self._result is None:
                raise RuntimeError(
                    "Process-tree result is available only after sampling stops."
                )
            return self._result

    def __enter__(self) -> Self:
        if self._started:
            raise RuntimeError("Process-tree sampler instances are single-use.")
        self._started = True

        try:
            self._root_process = self._process_factory(self._root_pid)
        except Exception:
            self._record_access_error()

        self._metric = self._select_metric()
        self._record_sample()
        self._thread = threading.Thread(
            target=self._sampling_loop,
            name="process-tree-peak-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> Literal[False]:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join()

        with self._lock:
            worker_error = self._worker_error
        if worker_error is not None:
            with self._lock:
                self._finished = True
            raise worker_error

        try:
            self._record_sample()
        except BaseException:
            with self._lock:
                self._finished = True
            raise

        measurement_error: RuntimeError | None = None
        with self._lock:
            if self._peak_bytes <= 0:
                measurement_error = RuntimeError(
                    "No process-tree memory sample could be recorded."
                )
            elif self._metric is None:
                measurement_error = RuntimeError(
                    "Process-tree memory metric was not selected."
                )
            else:
                self._result = ProcessTreePeak(
                    metric=self._metric,
                    peak_bytes=self._peak_bytes,
                    sample_interval_ms=self._sample_interval_ms,
                    sample_count=self._sample_count,
                    process_churn_count=self._process_churn_count,
                    access_error_count=self._access_error_count,
                    measurement_valid=self._measurement_valid,
                )
            self._finished = True
        if measurement_error is not None and exc_type is None:
            raise measurement_error
        return False

    def _select_metric(self) -> ProcessTreeMetric:
        if self._platform_system() != "Windows":
            return "process_tree_sum_rss_bytes"

        root = self._root_process
        if root is None:
            return "process_tree_sum_rss_bytes"
        try:
            _ = root.memory_full_info().uss
        except (AttributeError, NotImplementedError):
            return "process_tree_sum_rss_bytes"
        except Exception:
            self._record_access_error()
            return "process_tree_sum_rss_bytes"
        return "process_tree_sum_uss_bytes"

    def _sampling_loop(self) -> None:
        try:
            self._run_sampling_loop()
        except BaseException as error:
            with self._lock:
                if self._worker_error is None:
                    self._worker_error = error
            self._stop_event.set()

    def _run_sampling_loop(self) -> None:
        next_sample_at = self._clock() + self._sample_interval_seconds
        while not self._stop_event.is_set():
            remaining = max(0.0, next_sample_at - self._clock())
            if self._wait(self._stop_event, remaining):
                return
            if self._stop_event.is_set():
                return
            if self._clock() < next_sample_at:
                continue
            self._record_sample()
            next_sample_at += self._sample_interval_seconds
            current_time = self._clock()
            if next_sample_at <= current_time:
                missed_intervals = (
                    int((current_time - next_sample_at) / self._sample_interval_seconds)
                    + 1
                )
                next_sample_at += missed_intervals * self._sample_interval_seconds
                deadline_tolerance = max(
                    math.ulp(current_time) * 4,
                    self._sample_interval_seconds * 1e-12,
                )
                if next_sample_at - current_time <= deadline_tolerance:
                    next_sample_at += self._sample_interval_seconds

    def _record_sample(self) -> None:
        with self._lock:
            self._sample_count += 1

        root = self._root_process
        if root is None:
            return

        try:
            root_bytes = self._read_memory(root)
        except Exception:
            self._record_access_error()
            return

        total_bytes = root_bytes
        try:
            descendants = root.children(recursive=True)
        except Exception:
            self._record_access_error()
            self._record_peak(total_bytes)
            return

        unique_descendants: dict[int, _ProcessLike] = {}
        try:
            for child in descendants:
                if child.pid != self._root_pid:
                    unique_descendants.setdefault(child.pid, child)
        except Exception:
            self._record_access_error()

        for child_pid in sorted(unique_descendants):
            child = unique_descendants[child_pid]
            try:
                if not child.is_running():
                    self._record_churn()
                    continue
            except (psutil.NoSuchProcess, ProcessLookupError):
                self._record_churn()
            except Exception:
                self._record_access_error()
            else:
                try:
                    total_bytes += self._read_memory(child)
                except (psutil.NoSuchProcess, ProcessLookupError):
                    self._record_churn()
                except (psutil.AccessDenied, PermissionError):
                    self._record_child_memory_access_error(child)
                except Exception:
                    self._record_access_error()

        self._record_peak(total_bytes)

    def _read_memory(self, process: _ProcessLike) -> int:
        metric = self._metric
        if metric == "process_tree_sum_uss_bytes":
            value = process.memory_full_info().uss
        elif metric == "process_tree_sum_rss_bytes":
            value = process.memory_info().rss
        else:
            raise RuntimeError("Process-tree memory metric was not selected.")
        if isinstance(value, bool) or value < 0:
            raise ValueError("Process memory bytes must be non-negative.")
        return value

    def _record_child_memory_access_error(self, child: _ProcessLike) -> None:
        try:
            recheck = child.is_running()
        except (psutil.NoSuchProcess, ProcessLookupError):
            self._record_churn()
            return
        except Exception:
            self._record_access_error()
            return

        if recheck is False:
            self._record_churn()
            return
        if recheck is not True:
            self._record_access_error()
            return
        if self._platform_system() != "Windows":
            self._record_access_error()
            return

        try:
            wait_result = child.wait(timeout=0.0)
        except (psutil.NoSuchProcess, ProcessLookupError):
            self._record_churn()
        except Exception:
            self._record_access_error()
        else:
            if wait_result is None or (
                isinstance(wait_result, int) and not isinstance(wait_result, bool)
            ):
                self._record_churn()
            else:
                self._record_access_error()

    def _record_peak(self, sample_bytes: int) -> None:
        with self._lock:
            self._peak_bytes = max(self._peak_bytes, sample_bytes)

    def _record_churn(self) -> None:
        with self._lock:
            self._process_churn_count += 1

    def _record_access_error(self) -> None:
        with self._lock:
            self._access_error_count += 1
            self._measurement_valid = False
