from time import perf_counter_ns
from dataclasses import dataclass, field
from typing import Iterator
from types import TracebackType


@dataclass(slots=True)
class Timer:
    """The data structure that holds the start and stop times of a stopwatch,
    and the total."""

    _times: list[int] = field(default_factory=list[int])
    _total: int = 0

    @property
    def running(self) -> bool:
        """Whether the stopwatch is currently running."""
        return len(self._times) % 2 == 1

    @property
    def total(self) -> int:
        """The total time the stopwatch has been running, in nanoseconds."""
        total = self._total
        if self.running:
            total += perf_counter_ns() - self._times[-1]
        return total

    def start(self):
        """Start the stopwatch. If the stopwatch is already running, does nothing."""
        if not self.running:
            self._times.append(perf_counter_ns())

    def stop(self):
        """Stop the stopwatch. If the stopwatch is not running, does nothing."""
        time = perf_counter_ns()
        if self.running:
            self._total += time - self._times[-1]
            self._times.append(time)

    def __repr__(self) -> str:
        return f"Timer(running={self.running}, total={self.total})"


class TimerContextManager:
    """A context manager that starts a stopwatch when entering the context,
    and stops it when exiting."""

    def __init__(self, timer: Timer) -> None:
        self._timer = timer

    def __enter__(self):
        self._timer.start()

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._timer.stop()


class TimerCollection:
    """A class that holds multiple named stopwatches, that can be started,
    stopped and reset independently from one another."""

    def __init__(self) -> None:
        self._timers: dict[str, Timer] = {}

    def get(self, name: str) -> int:
        """Get the time of the stopwatch with the given name. If the stopwatch
        does not exist, raises KeyError."""
        return self._timers[name].total

    def __getitem__(self, name: str) -> int:
        """Get the time of the stopwatch with the given name. If the stopwatch
        does not exist, raises KeyError."""
        return self._timers[name].total

    def start(self, name: str, *, already_started_at: int | None = None):
        """Start the stopwatch with the given name. If the stopwatch does not
        exist, it is created.
        If already_started_at is provided, the stopwatch is considered to have
        been started at that time (in nanoseconds since some arbitrary point in
        time), the current time is ignored and the stopwatch is overwritten."""
        if already_started_at is not None:
            self._timers[name] = Timer([already_started_at], 0)
        self._timers.setdefault(name, Timer()).start()

    def stop(self, name: str, already_started_at: int | None = None):
        """Stop the stopwatch with the given name. If the stopwatch does not
        exist, raises KeyError.
        If already_started_at is provided, the stopwatch is considered to have
        been started at that time. The timer is created and stopped immediately,
        and overwrites the existing timer if it exists."""
        if already_started_at is None:
            self._timers[name].stop()
        else:
            time = perf_counter_ns()
            self._timers[name] = Timer(
                [already_started_at, time], time - already_started_at
            )

    def stop_all(self):
        """Stop all running stopwatches."""
        for timer in self._timers.values():
            timer.stop()

    def reset(self, name: str):
        """Reset the stopwatch with the given name. If the stopwatch does not
        exist, raises KeyError."""
        self._timers[name] = Timer()

    def __iter__(self) -> Iterator[tuple[str, int]]:
        """Returns a sequence of (name, time)."""
        return ((name, timer.total) for name, timer in self._timers.items())

    def time(self, name: str) -> TimerContextManager:
        """Returns a context manager that starts the stopwatch with the given
        name when entering the context, and stops it when exiting."""
        timer = self._timers.setdefault(name, Timer())
        return TimerContextManager(timer)
