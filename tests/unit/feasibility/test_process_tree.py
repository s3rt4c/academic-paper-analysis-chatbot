from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Thread
from types import SimpleNamespace

import psutil
import pytest
from pydantic import ValidationError

from academic_chatbot.feasibility.process_tree import (
    ProcessTreePeak,
    ProcessTreePeakSampler,
    evaluate_memory_gate,
)


@dataclass(frozen=True)
class _ProcessState:
    uss: int | BaseException
    rss: int | BaseException
    running: bool | BaseException | tuple[bool | BaseException, ...] = True


@dataclass(frozen=True)
class _Snapshot:
    processes: dict[int, _ProcessState]
    descendants: tuple[int, ...] = ()
    children_error: BaseException | None = None


@dataclass
class _Snapshots:
    frames: tuple[_Snapshot, ...]
    index: int = 0
    uss_calls: list[int] = field(default_factory=list)
    rss_calls: list[int] = field(default_factory=list)

    @property
    def current(self) -> _Snapshot:
        return self.frames[self.index]

    def advance(self) -> None:
        if self.index < len(self.frames) - 1:
            self.index += 1

    def process(self, pid: int) -> _FakeProcess:
        return _FakeProcess(pid, self)


class _FakeProcess:
    def __init__(self, pid: int, snapshots: _Snapshots) -> None:
        self.pid = pid
        self._snapshots = snapshots
        self._running_call_count = 0

    def children(self, *, recursive: bool) -> list[_FakeProcess]:
        assert recursive is True
        error = self._snapshots.current.children_error
        if error is not None:
            raise error
        return [self._snapshots.process(pid) for pid in self._snapshots.current.descendants]

    def is_running(self) -> bool:
        value = self._state().running
        if isinstance(value, tuple):
            value = value[min(self._running_call_count, len(value) - 1)]
            self._running_call_count += 1
        if isinstance(value, BaseException):
            raise value
        return value

    def memory_full_info(self) -> SimpleNamespace:
        self._snapshots.uss_calls.append(self.pid)
        value = self._state().uss
        if isinstance(value, BaseException):
            raise value
        return SimpleNamespace(uss=value)

    def memory_info(self) -> SimpleNamespace:
        self._snapshots.rss_calls.append(self.pid)
        value = self._state().rss
        if isinstance(value, BaseException):
            raise value
        return SimpleNamespace(rss=value)

    def _state(self) -> _ProcessState:
        try:
            return self._snapshots.current.processes[self.pid]
        except KeyError as error:
            raise psutil.NoSuchProcess(self.pid) from error


@dataclass
class _FakeClock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _OneIntervalWait:
    def __init__(self, snapshots: _Snapshots, clock: _FakeClock) -> None:
        self._snapshots = snapshots
        self._clock = clock
        self._allow_interval = Event()
        self.first_wait_started = Event()
        self.second_wait_started = Event()
        self.timeouts: list[float] = []
        self._call_count = 0

    def allow_interval(self) -> None:
        self._allow_interval.set()

    def __call__(self, stop_event: Event, timeout: float) -> bool:
        self.timeouts.append(timeout)
        self._call_count += 1
        if self._call_count == 1:
            self.first_wait_started.set()
            while not self._allow_interval.is_set():
                if stop_event.wait(timeout=0.01):
                    return True
            self._clock.advance(timeout)
            self._snapshots.advance()
            return False

        self.second_wait_started.set()
        stop_event.wait()
        self._snapshots.advance()
        return True


class _ScheduleWait:
    def __init__(self, clock: _FakeClock, overshoots: tuple[float, ...]) -> None:
        self._clock = clock
        self._overshoots = overshoots
        self.timeouts: list[float] = []

    def __call__(self, stop_event: Event, timeout: float) -> bool:
        del stop_event
        self.timeouts.append(timeout)
        wake_index = len(self.timeouts) - 1
        if wake_index >= len(self._overshoots):
            return True
        self._clock.advance(timeout + self._overshoots[wake_index])
        return False


class _SchedulingSampler(ProcessTreePeakSampler):
    def __init__(
        self,
        clock: _FakeClock,
        wait: _ScheduleWait,
        *,
        sample_cost_seconds: float,
    ) -> None:
        super().__init__(10, clock=clock, wait=wait)
        self._test_clock = clock
        self._sample_cost_seconds = sample_cost_seconds
        self.sample_times: list[float] = []

    def _record_sample(self) -> None:
        self.sample_times.append(self._test_clock())
        self._test_clock.advance(self._sample_cost_seconds)


class _BlockingStopWait:
    def __init__(self) -> None:
        self.started = Event()
        self.stopped = Event()

    def __call__(self, stop_event: Event, timeout: float) -> bool:
        del timeout
        self.started.set()
        try:
            return stop_event.wait()
        finally:
            self.stopped.set()


def _state(memory_bytes: int) -> _ProcessState:
    return _ProcessState(uss=memory_bytes, rss=memory_bytes * 10)


def test_injected_wait_stops_even_before_interval_release() -> None:
    snapshots = _Snapshots((_Snapshot({1: _state(100)}),))
    wait = _OneIntervalWait(snapshots, _FakeClock())
    stop_event = Event()
    outcome: list[bool] = []
    worker = Thread(target=lambda: outcome.append(wait(stop_event, 0.01)), daemon=True)
    worker.start()
    assert wait.first_wait_started.wait(timeout=1)

    stop_event.set()
    worker.join(timeout=0.1)
    try:
        assert worker.is_alive() is False
        assert outcome == [True]
    finally:
        wait.allow_interval()
        worker.join(timeout=1)


def test_sampling_deadlines_do_not_drift_by_sample_cost() -> None:
    clock = _FakeClock()
    wait = _ScheduleWait(clock, overshoots=(0.0, 0.0))
    sampler = _SchedulingSampler(clock, wait, sample_cost_seconds=0.004)

    sampler._sampling_loop()

    assert sampler.sample_times == pytest.approx([0.01, 0.02])
    assert wait.timeouts == pytest.approx([0.01, 0.006, 0.006])


def test_late_wakeup_skips_missed_ticks_without_zero_timeout_busy_loop() -> None:
    clock = _FakeClock()
    wait = _ScheduleWait(clock, overshoots=(0.025, 0.0))
    sampler = _SchedulingSampler(clock, wait, sample_cost_seconds=0.002)

    sampler._sampling_loop()

    assert sampler.sample_times == pytest.approx([0.035, 0.04])
    assert wait.timeouts == pytest.approx([0.01, 0.003, 0.008])
    assert all(timeout > 0.0 for timeout in wait.timeouts)


def test_sample_cost_at_interval_multiple_still_schedules_a_future_deadline() -> None:
    clock = _FakeClock()
    wait = _ScheduleWait(clock, overshoots=(0.0, 0.0))
    sampler = _SchedulingSampler(clock, wait, sample_cost_seconds=0.02)

    sampler._sampling_loop()

    assert sampler.sample_times == pytest.approx([0.01, 0.04])
    assert wait.timeouts == pytest.approx([0.01, 0.01, 0.01])
    assert all(timeout > 0.0 for timeout in wait.timeouts)


def test_context_body_exception_is_not_masked_when_no_sample_is_available() -> None:
    wait = _BlockingStopWait()
    sampler = ProcessTreePeakSampler(
        10,
        root_pid=1,
        process_factory=lambda pid: (_ for _ in ()).throw(psutil.NoSuchProcess(pid)),
        platform_system=lambda: "Linux",
        wait=wait,
    )

    with pytest.raises(LookupError, match="body failed"):
        with sampler:
            assert wait.started.wait(timeout=1)
            raise LookupError("body failed")

    assert wait.stopped.is_set()
    with pytest.raises(RuntimeError, match="available only after"):
        _ = sampler.result


def test_sampler_is_single_use_after_clean_exit() -> None:
    snapshots = _Snapshots(
        (
            _Snapshot({1: _state(100)}),
            _Snapshot({1: _state(110)}),
            _Snapshot({1: _state(90)}),
        )
    )
    clock = _FakeClock()
    wait = _OneIntervalWait(snapshots, clock)
    sampler = ProcessTreePeakSampler(
        10,
        root_pid=1,
        process_factory=snapshots.process,
        platform_system=lambda: "Windows",
        clock=clock,
        wait=wait,
    )

    with sampler:
        assert wait.first_wait_started.wait(timeout=1)
        wait.allow_interval()
        assert wait.second_wait_started.wait(timeout=1)

    with pytest.raises(RuntimeError, match="sampler instances are single-use"):
        sampler.__enter__()


def test_normal_exit_without_a_sample_raises_stable_error_after_worker_cleanup() -> None:
    wait = _BlockingStopWait()
    sampler = ProcessTreePeakSampler(
        10,
        root_pid=1,
        process_factory=lambda pid: (_ for _ in ()).throw(psutil.NoSuchProcess(pid)),
        platform_system=lambda: "Linux",
        wait=wait,
    )

    with pytest.raises(RuntimeError, match="No process-tree memory sample could be recorded"):
        with sampler:
            assert wait.started.wait(timeout=1)

    assert wait.stopped.is_set()


def _run_three_samples(
    frames: tuple[_Snapshot, _Snapshot, _Snapshot],
    *,
    platform_name: str = "Windows",
) -> tuple[ProcessTreePeak, _Snapshots, _OneIntervalWait]:
    snapshots = _Snapshots(frames)
    clock = _FakeClock()
    wait = _OneIntervalWait(snapshots, clock)
    sampler = ProcessTreePeakSampler(
        10,
        root_pid=1,
        process_factory=snapshots.process,
        platform_system=lambda: platform_name,
        clock=clock,
        wait=wait,
    )

    with pytest.raises(RuntimeError, match="available only after"):
        _ = sampler.result

    with sampler:
        assert wait.first_wait_started.wait(timeout=1)
        with pytest.raises(RuntimeError, match="available only after"):
            _ = sampler.result
        wait.allow_interval()
        assert wait.second_wait_started.wait(timeout=1)

    return sampler.result, snapshots, wait


def test_sampler_records_peak_sum_for_root_and_unique_descendants() -> None:
    result, snapshots, wait = _run_three_samples(
        (
            _Snapshot({1: _state(100), 2: _state(25)}, descendants=(2, 2)),
            _Snapshot({1: _state(120), 2: _state(40)}, descendants=(2,)),
            _Snapshot({1: _state(90)}),
        )
    )

    assert result.metric == "process_tree_sum_uss_bytes"
    assert result.peak_bytes == 160
    assert result.sample_count == 3
    assert result.process_churn_count == 0
    assert result.access_error_count == 0
    assert result.measurement_valid is True
    assert snapshots.rss_calls == []
    assert wait.timeouts[0] == pytest.approx(0.01)


def test_windows_falls_back_to_rss_once_when_uss_is_unavailable_at_startup() -> None:
    unavailable = AttributeError("uss is unavailable")
    frames = (
        _Snapshot({1: _ProcessState(uss=unavailable, rss=100)}),
        _Snapshot({1: _ProcessState(uss=999, rss=120)}),
        _Snapshot({1: _ProcessState(uss=999, rss=90)}),
    )

    result, snapshots, _ = _run_three_samples(frames)

    assert result.metric == "process_tree_sum_rss_bytes"
    assert result.peak_bytes == 120
    assert snapshots.uss_calls == [1]
    assert snapshots.rss_calls == [1, 1, 1]


def test_startup_access_denied_selects_rss_and_invalidates_measurement() -> None:
    frames = (
        _Snapshot({1: _ProcessState(uss=psutil.AccessDenied(1), rss=100)}),
        _Snapshot({1: _ProcessState(uss=999, rss=120)}),
        _Snapshot({1: _ProcessState(uss=999, rss=90)}),
    )

    result, snapshots, _ = _run_three_samples(frames)

    assert result.metric == "process_tree_sum_rss_bytes"
    assert result.peak_bytes == 120
    assert result.access_error_count == 1
    assert result.measurement_valid is False
    assert snapshots.uss_calls == [1]
    assert snapshots.rss_calls == [1, 1, 1]


def test_non_windows_uses_rss_without_probing_uss() -> None:
    frames = (
        _Snapshot({1: _ProcessState(uss=999, rss=100)}),
        _Snapshot({1: _ProcessState(uss=999, rss=120)}),
        _Snapshot({1: _ProcessState(uss=999, rss=90)}),
    )

    result, snapshots, _ = _run_three_samples(frames, platform_name="Linux")

    assert result.metric == "process_tree_sum_rss_bytes"
    assert result.peak_bytes == 120
    assert snapshots.uss_calls == []


def test_disappearing_child_counts_as_churn_without_invalidating_measurement() -> None:
    result, _, _ = _run_three_samples(
        (
            _Snapshot({1: _state(100)}),
            _Snapshot(
                {1: _state(120), 2: _ProcessState(uss=psutil.NoSuchProcess(2), rss=20)},
                descendants=(2,),
            ),
            _Snapshot({1: _state(90)}),
        )
    )

    assert result.process_churn_count == 1
    assert result.access_error_count == 0
    assert result.measurement_valid is True


def test_zombie_child_counts_as_churn_without_invalidating_measurement() -> None:
    result, _, _ = _run_three_samples(
        (
            _Snapshot({1: _state(100)}),
            _Snapshot(
                {
                    1: _state(120),
                    2: _ProcessState(uss=psutil.ZombieProcess(2), rss=20),
                },
                descendants=(2,),
            ),
            _Snapshot({1: _state(90)}),
        )
    )

    assert result.process_churn_count == 1
    assert result.access_error_count == 0
    assert result.measurement_valid is True


@pytest.mark.parametrize(
    "memory_error",
    [psutil.AccessDenied(2), PermissionError()],
)
def test_child_memory_access_error_with_live_recheck_invalidates_measurement(
    memory_error: BaseException,
) -> None:
    result, _, _ = _run_three_samples(
        (
            _Snapshot({1: _state(100)}),
            _Snapshot(
                {
                    1: _state(120),
                    2: _ProcessState(
                        uss=memory_error,
                        rss=20,
                        running=(True, True),
                    ),
                },
                descendants=(2,),
            ),
            _Snapshot({1: _state(90)}),
        )
    )

    assert result.peak_bytes == 120
    assert result.process_churn_count == 0
    assert result.access_error_count == 1
    assert result.measurement_valid is False


@pytest.mark.parametrize(
    ("memory_error", "stopped_recheck"),
    [
        (psutil.AccessDenied(2), False),
        (psutil.AccessDenied(2), psutil.NoSuchProcess(2)),
        (psutil.AccessDenied(2), ProcessLookupError()),
        (PermissionError(), False),
        (PermissionError(), psutil.NoSuchProcess(2)),
        (PermissionError(), ProcessLookupError()),
    ],
)
def test_child_memory_access_error_after_confirmed_exit_counts_as_churn(
    memory_error: BaseException,
    stopped_recheck: bool | BaseException,
) -> None:
    result, _, _ = _run_three_samples(
        (
            _Snapshot({1: _state(100)}),
            _Snapshot(
                {
                    1: _state(120),
                    2: _ProcessState(
                        uss=memory_error,
                        rss=20,
                        running=(True, stopped_recheck),
                    ),
                },
                descendants=(2,),
            ),
            _Snapshot({1: _state(90)}),
        )
    )

    assert result.peak_bytes == 120
    assert result.process_churn_count == 1
    assert result.access_error_count == 0
    assert result.measurement_valid is True


@pytest.mark.parametrize(
    ("memory_error", "recheck_error"),
    [
        (psutil.AccessDenied(2), psutil.AccessDenied(2)),
        (psutil.AccessDenied(2), PermissionError()),
        (psutil.AccessDenied(2), RuntimeError()),
        (PermissionError(), psutil.AccessDenied(2)),
        (PermissionError(), PermissionError()),
        (PermissionError(), RuntimeError()),
    ],
)
def test_child_memory_access_error_with_uncertain_recheck_invalidates_measurement(
    memory_error: BaseException,
    recheck_error: BaseException,
) -> None:
    result, _, _ = _run_three_samples(
        (
            _Snapshot({1: _state(100)}),
            _Snapshot(
                {
                    1: _state(120),
                    2: _ProcessState(
                        uss=memory_error,
                        rss=20,
                        running=(True, recheck_error),
                    ),
                },
                descendants=(2,),
            ),
            _Snapshot({1: _state(90)}),
        )
    )

    assert result.peak_bytes == 120
    assert result.process_churn_count == 0
    assert result.access_error_count == 1
    assert result.measurement_valid is False


@pytest.mark.parametrize(
    "initial_running_error",
    [psutil.AccessDenied(2), PermissionError()],
)
def test_initial_child_running_access_error_invalidates_without_memory_recheck(
    initial_running_error: BaseException,
) -> None:
    result, _, _ = _run_three_samples(
        (
            _Snapshot({1: _state(100)}),
            _Snapshot(
                {
                    1: _state(120),
                    2: _ProcessState(
                        uss=20,
                        rss=20,
                        running=initial_running_error,
                    ),
                },
                descendants=(2,),
            ),
            _Snapshot({1: _state(90)}),
        )
    )

    assert result.peak_bytes == 120
    assert result.process_churn_count == 0
    assert result.access_error_count == 1
    assert result.measurement_valid is False


def test_children_enumeration_access_denied_invalidates_measurement() -> None:
    result, _, _ = _run_three_samples(
        (
            _Snapshot({1: _state(100)}),
            _Snapshot(
                {1: _state(120)},
                children_error=psutil.AccessDenied(1),
            ),
            _Snapshot({1: _state(90)}),
        )
    )

    assert result.peak_bytes == 120
    assert result.process_churn_count == 0
    assert result.access_error_count == 1
    assert result.measurement_valid is False


def test_root_sampling_failure_invalidates_measurement() -> None:
    result, _, _ = _run_three_samples(
        (
            _Snapshot({1: _state(100)}),
            _Snapshot({1: _ProcessState(uss=psutil.AccessDenied(1), rss=120)}),
            _Snapshot({1: _state(90)}),
        )
    )

    assert result.access_error_count == 1
    assert result.measurement_valid is False


def test_process_tree_peak_is_frozen_and_forbids_extra_fields() -> None:
    result = ProcessTreePeak(
        metric="process_tree_sum_rss_bytes",
        peak_bytes=1,
        sample_interval_ms=10,
        sample_count=1,
        process_churn_count=0,
        access_error_count=0,
        measurement_valid=True,
    )

    with pytest.raises(ValidationError, match="frozen"):
        result.peak_bytes = 2

    payload = result.model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        ProcessTreePeak.model_validate(payload)


def test_process_tree_peak_rejects_valid_measurement_with_access_errors() -> None:
    with pytest.raises(
        ValidationError,
        match="measurement_valid cannot be true when access_error_count is non-zero",
    ):
        ProcessTreePeak(
            metric="process_tree_sum_rss_bytes",
            peak_bytes=1,
            sample_interval_ms=10,
            sample_count=1,
            process_churn_count=0,
            access_error_count=1,
            measurement_valid=True,
        )

    invalid_without_access_error = ProcessTreePeak(
        metric="process_tree_sum_rss_bytes",
        peak_bytes=1,
        sample_interval_ms=10,
        sample_count=1,
        process_churn_count=0,
        access_error_count=0,
        measurement_valid=False,
    )
    assert invalid_without_access_error.measurement_valid is False


@pytest.mark.parametrize("sample_interval_ms", [0, -1])
def test_sampler_rejects_non_positive_sample_interval(sample_interval_ms: int) -> None:
    with pytest.raises(ValueError, match="sample_interval_ms must be positive"):
        ProcessTreePeakSampler(sample_interval_ms)


def test_memory_gate_is_strictly_below_limit() -> None:
    assert evaluate_memory_gate(12_884_901_887) == "passed"
    assert evaluate_memory_gate(12_884_901_888) == "failed"
    assert evaluate_memory_gate(99, limit_bytes=100) == "passed"
    assert evaluate_memory_gate(100, limit_bytes=100) == "failed"


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_memory_gate_rejects_non_integer_peak_bytes(value: object) -> None:
    with pytest.raises(TypeError, match="peak_bytes must be an integer"):
        evaluate_memory_gate(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1])
def test_memory_gate_rejects_non_positive_peak_bytes(value: int) -> None:
    with pytest.raises(ValueError, match="peak_bytes must be positive"):
        evaluate_memory_gate(value)


@pytest.mark.parametrize("value", [False, 1.5, "1"])
def test_memory_gate_rejects_non_integer_limit_bytes(value: object) -> None:
    with pytest.raises(TypeError, match="limit_bytes must be an integer"):
        evaluate_memory_gate(1, limit_bytes=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1])
def test_memory_gate_rejects_non_positive_limit_bytes(value: int) -> None:
    with pytest.raises(ValueError, match="limit_bytes must be positive"):
        evaluate_memory_gate(1, limit_bytes=value)
