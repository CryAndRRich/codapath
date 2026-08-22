"""Capture a run's console output to a file without hiding it from the console."""

import os
import sys
from contextlib import contextmanager


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


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
