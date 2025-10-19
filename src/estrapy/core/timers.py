from time import perf_counter_ns
from dataclasses import dataclass, field
from typing import Iterator, Literal
from types import TracebackType
from collections import OrderedDict
from statistics import mean, stdev

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
    
    @property
    def times(self) -> list[int]:
        """The list of durations of each start/stop cycle, in nanoseconds."""
        if len(self._times) < 2:
            return []
        return [b-a for a,b in zip(self._times[0::2], self._times[1::2])]

    @property
    def mean(self) -> float:
        """The mean time of the stopwatch over the number of start/stop cycles, in nanoseconds."""
        return mean(self.times) if self.cycles else 0.0
    
    @property
    def stdev(self) -> float:
        """The standard deviation of the stopwatch over the number of start/stop cycles, in nanoseconds."""
        return stdev(self.times) if self.cycles > 1 else 0.0
    
    @property
    def cycles(self) -> int:
        """The number of start/stop cycles the stopwatch has undergone."""
        return len(self._times) // 2

    def start(self, at: int | None = None):
        """Start the stopwatch. If the stopwatch is already running, does nothing."""
        time = at or perf_counter_ns()
        if not self.running:
            self._times.append(time)

    def stop(self, *, at: int | None = None, started_at: int | None = None):
        """Stop the stopwatch. If the stopwatch is not running, does nothing."""
        time = at or perf_counter_ns()
        if self.running:
            self._total += time - self._times[-1]
            self._times.append(time)
        elif started_at is not None:
            self._total += time - started_at
            self._times.append(started_at)
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
    stopped and reset independently from one another.
    The stopwatches are organized in a file-like structure, where each stopwatch
    is identified by a string name, separated by slashes (`/`)."""

    def __init__(self) -> None:
        self._timers: OrderedDict[str, Timer] = OrderedDict()
    
    @staticmethod
    def _ancestors(name: str) -> list[str]:
        """Generate a list of all ancestor names of the given name, including the name itself.
        E.g. for "a/b/c", returns ["", "a", "a/b", "a/b/c"]."""
        parts = name.split('/')
        return ['/'.join(parts[:i]) for i in range(len(parts) + 1)]

    def get(self, name: str) -> int:
        """Get the time of the stopwatch with the given name. If the stopwatch
        does not exist, raises KeyError."""
        return self._timers[name].total
    def get_ns(self, name: str) -> int:
        """Get the time of the stopwatch with the given name in nanoseconds.
        If the stopwatch does not exist, raises KeyError."""
        return self._timers[name].total
    def get_ms(self, name: str) -> float:
        """Get the time of the stopwatch with the given name in milliseconds.
        If the stopwatch does not exist, raises KeyError."""
        return self._timers[name].total / 1_000_000
    def get_s(self, name: str) -> float:
        """Get the time of the stopwatch with the given name in seconds.
        If the stopwatch does not exist, raises KeyError."""
        return self._timers[name].total / 1_000_000_000
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
        time = already_started_at or perf_counter_ns()
        for ancestor in self._ancestors(name):
            self._timers.setdefault(ancestor, Timer()).start(at=time)

    def stop(self, name: str | None = None, already_started_at: int | None = None, stopped_at: int | None = None):
        """Stop the stopwatch with the given name. If the stopwatch does not
        exist, raises KeyError.
        If already_started_at is provided, the stopwatch is considered to have
        been started at that time. The timer is created and stopped immediately,
        and overwrites the existing timer if it exists.
        Stops all sub-timers as well."""
        time = stopped_at or perf_counter_ns()
        if already_started_at is not None:
            for ancestor in self._ancestors(name or '')[:-1]:
                self._timers.setdefault(ancestor, Timer()).start(at=already_started_at)
            self._timers.setdefault(name or '', Timer()).stop(at=time, started_at=already_started_at)
        else:
            for subname, timer in self._timers.items():
                if name is None or subname.startswith(name+'/') or subname == name:
                    timer.stop(at=time, started_at=already_started_at)

    def stop_all(self):
        """Stop all running stopwatches."""
        time = perf_counter_ns()
        for timer in self._timers.values():
            timer.stop(at=time)

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

    def table_format(self, unit: Literal["s", "ms", "us", "ns"] = "s") -> str:
        mul = {
            "s": 1_000_000_000,
            "ms": 1_000_000,
            "us": 1_000,
            "ns": 1,
        }[unit]
        """Returns a string representation of the timers in a table format."""
        # Get longest name length
        maxlen = max((len(name) for name in self._timers), default=4)
        lines: list[str] = []
        lines.append(f"{'Name':<{maxlen}} | {f'Total ({unit})':<12} | {f'Mean ({unit})':<12} | {f'Stdev ({unit})':<12} | {'Cycles':<6}")
        lines.append("-" * len(lines[0]))
        for name, timer in self._timers.items():
            lines.append(f"{name or "<total>":<{maxlen}} | {timer.total / mul:>12.2f} | {timer.mean / mul:>12.2f} | {timer.stdev / mul:>12.2f} | {timer.cycles:<6}")
        return "\n".join(lines)
