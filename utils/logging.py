"""Capture a run's console output to a file without hiding it from the console."""

import os
import sys
from contextlib import contextmanager


class _Tee:
    """A stdout/stderr stand-in that writes to several streams at once.

    **Anything it does not implement is delegated to the FIRST stream** (the
    real console). Libraries treat `sys.stdout` as a full file object, not as
    something with only `write`/`flush`: `transformers` calls
    `sys.stdout.isatty()` while reporting checkpoint loading, and an earlier
    version of this class -- which defined write and flush and nothing else --
    crashed every `Dinov2Model.from_pretrained` call inside a run with
    `AttributeError: '_Tee' object has no attribute 'isatty'`. Adding
    `isatty` alone would just move the same failure to the next attribute
    some library decides to probe (`fileno`, `encoding`, `buffer`, ...), so
    the fallback is generic.

    Delegating to the console rather than the log file is deliberate: the
    answers callers want (`isatty` -> is this a terminal, `encoding`,
    `fileno`) describe the real destination. A log file would answer
    `isatty() == False` too, but `fileno()` would hand out the file's
    descriptor, which is not what a caller asking `sys.stdout.fileno()`
    means.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        """False: output is being captured to a file, so callers must not
        emit ANSI colour codes -- they would land in the log as escape
        noise. Answered here rather than delegated, because the console half
        of the tee may well be a terminal while the run's output is not
        purely one."""
        return False

    def __getattr__(self, name):
        # Only reached for attributes not defined above; `_streams` itself is
        # set in __init__ so it never recurses here.
        return getattr(self._streams[0], name)


@contextmanager
def tee_stdout(path: str):
    """Mirror stdout and stderr into `path` for the duration of the block.

    Diagnostics printed by the samplers are the only record of what happened
    inside a run, and on Kaggle the console scrollback is lost when the session
    ends, so the log has to land next to the checkpoints.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with open(path, "a", encoding="utf-8") as handle:
        sys.stdout = _Tee(original_stdout, handle)
        sys.stderr = _Tee(original_stderr, handle)
        try:
            yield path
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
