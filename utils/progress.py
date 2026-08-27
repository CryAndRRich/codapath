"""Progress reporting that survives a Kaggle log.

`tqdm` writes to a terminal with carriage returns. Kaggle keeps the notebook
output, but a log file tee'd from the same stream ends up with either one
enormous line or nothing useful, and the interesting number — when will this
finish — is exactly what is lost.

So: `tqdm` stays on the console for interactive feedback, and a plain periodic
line carrying the ETA also goes to the log. `estimated_finish` is the value
worth reading, since the whole point is deciding whether a Kaggle session has
enough time left.
"""

from __future__ import annotations

import datetime as _datetime
import time
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional

from tqdm.auto import tqdm

__all__ = [
    "progress", "Stopwatch", "format_duration", "format_eta", "quiet_progress",
]

# REFINE calls other samplers thousands of times per budget. Each of those
# would otherwise draw its own bar and print its own ETA, producing tens of
# thousands of lines that hide the outer run's diagnostics. An outer sampler
# wraps its inner calls in `quiet_progress()` and reports its own progress
# instead.
_QUIET_DEPTH = 0


@contextmanager
def quiet_progress():
    """Silence `progress`/`Stopwatch.report` inside nested sampler calls."""
    global _QUIET_DEPTH
    _QUIET_DEPTH += 1
    try:
        yield
    finally:
        _QUIET_DEPTH -= 1


def is_quiet() -> bool:
    return _QUIET_DEPTH > 0


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.1f}min"
    return f"{minutes / 60.0:.2f}h"


def format_eta(seconds_remaining: float) -> str:
    """Absolute wall-clock finish time, not just a remaining duration.

    A Kaggle session has a hard deadline, so "finishes at 03:14" answers the
    real question in a way "2.3h left" does not.
    """
    finish = _datetime.datetime.now() + _datetime.timedelta(
        seconds=max(0.0, float(seconds_remaining))
    )
    return finish.strftime("%H:%M:%S")


class Stopwatch:
    """Elapsed/ETA for a loop whose length is known but which is not iterated.

    Used where the work is a `while` loop or a nested chunk sweep, so a `tqdm`
    wrapper does not fit but progress still has to be reported.

    `report()` is rate-limited rather than printing on every call. A sampler
    like REFINE invokes other samplers thousands of times, and an unthrottled
    line per round buries the run's actual diagnostics in tens of thousands of
    lines of noise — the log then costs more than it explains.
    """

    def __init__(self, total: int, label: str = "", min_interval: float = 30.0) -> None:
        self.total = max(1, int(total))
        self.label = label
        self.started = time.time()
        self.done = 0
        self.min_interval = float(min_interval)
        # Seeded to the start time, not 0: `now - 0` is an epoch-sized interval,
        # which would make the very first `report()` always print regardless of
        # the rate limit.
        self._last_report = self.started

    def advance(self, count: int = 1) -> None:
        self.done += int(count)

    def report(self, force: bool = False) -> None:
        """Print progress at most every `min_interval` seconds.

        Always prints the final update, so a short loop still reports once and
        a finished loop never looks abandoned.
        """
        if is_quiet():
            return
        now = time.time()
        finished = self.done >= self.total
        if not (force or finished or now - self._last_report >= self.min_interval):
            return
        self._last_report = now
        print(f"  [{self.line()}]")

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def remaining(self) -> float:
        if self.done <= 0:
            return float("nan")
        return self.elapsed * (self.total - self.done) / self.done

    def line(self) -> str:
        base = f"{self.label} {self.done}/{self.total} | elapsed {format_duration(self.elapsed)}"
        remaining = self.remaining
        if self.done <= 0 or remaining != remaining:  # NaN check
            return base
        # An ETA extrapolated from a sub-second sample is noise, and printing a
        # wall-clock finish time for it invites reading precision that is not
        # there.
        if self.elapsed < 1.0:
            return base
        return (
            f"{base} | left {format_duration(remaining)} | ETA {format_eta(remaining)}"
        )


def progress(
    iterable: Iterable,
    desc: str,
    total: Optional[int] = None,
    log_every: int = 0,
    leave: bool = False,
) -> Iterator:
    """`tqdm` on the console, plus periodic ETA lines for the log file.

    `log_every=0` disables the log lines, for loops short enough that the
    console bar is the whole story.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

    if is_quiet():
        # Nested call: iterate plainly, draw nothing.
        yield from iterable
        return

    watch = Stopwatch(total or 1, desc)
    bar = tqdm(iterable, desc=desc, total=total, leave=leave)
    for position, item in enumerate(bar, start=1):
        yield item
        watch.advance()
        if log_every and total and (position % log_every == 0 or position == total):
            # `write` keeps the line from being eaten by the active bar.
            tqdm.write(f"  [{watch.line()}]")
