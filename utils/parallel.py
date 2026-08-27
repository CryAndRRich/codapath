"""Run independent variants concurrently, one worker per GPU.

Kaggle's T4 x2 accelerator exposes two devices. A single AL run uses one of
them, so half the hardware idles for the whole session. The variants of a sweep
are independent — each writes its own files under `save_dir` and shares nothing
mutable — so they can be split across both devices.

Why processes, not threads: each worker calls `set_seed`, which mutates global
RNG state in `random`, `numpy` and `torch`, and sets `CUBLAS_WORKSPACE_CONFIG`
in the environment. Two threads doing that in one interpreter would interleave
and silently destroy reproducibility. Separate processes each get their own
global state, so a parallel run selects exactly what a serial run would.

`spawn` rather than `fork`: CUDA context cannot be inherited across a fork, and
`fork` in a process that has already initialised CUDA (as the notebook has,
after any `torch.cuda` call) is undefined behaviour. Consequently the worker
target must be importable and its arguments picklable, so the work is described
by a plain dict of primitives and re-resolved inside the child.
"""

from __future__ import annotations

import multiprocessing as _multiprocessing
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["visible_gpu_count", "run_variants_parallel", "WorkerResult"]


def visible_gpu_count() -> int:
    """Number of usable CUDA devices, without initialising a CUDA context here.

    Importing torch and calling `device_count` is enough; creating a context in
    the parent would be inherited-state trouble for the children.
    """
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


class WorkerResult(dict):
    """`{"label", "ok", "seconds", "error"}` for one variant."""

    @property
    def ok(self) -> bool:
        return bool(self["ok"])


def _worker(
    assigned: Sequence[Tuple[str, Dict[str, Any]]],
    result_queue,
    device_index: int,
    entry_point: Callable[..., Any],
) -> None:
    """Run this worker's assigned variants, all on one GPU.

    The work list is passed as a plain argument rather than pulled from a shared
    queue. `spawn` pickles the arguments, so the child has its whole list before
    it starts; a `multiprocessing.Queue` filled before `start()` relies on a
    background feeder thread that the spawned child does not inherit, so its
    first `get()` can block forever.

    A worker owns exactly one device for its whole life, so `CUDA_VISIBLE_DEVICES`
    is pinned once before torch initialises. Inside the child, that device is
    then always `cuda:0`.
    """
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
    for label, kwargs in assigned:
        started = time.time()
        try:
            entry_point(**kwargs)
            result_queue.put(WorkerResult(
                label=label, ok=True, seconds=time.time() - started,
                error=None, device=device_index,
            ))
        except Exception:
            # The traceback has to cross the process boundary as text; the
            # exception object itself may not be picklable.
            result_queue.put(WorkerResult(
                label=label, ok=False, seconds=time.time() - started,
                error=traceback.format_exc(), device=device_index,
            ))


def run_variants_parallel(
    variants: Sequence[Tuple[str, Dict[str, Any]]],
    entry_point: Callable[..., Any],
    num_workers: Optional[int] = None,
) -> List[WorkerResult]:
    """Run `(label, kwargs)` variants across GPUs, returning one result each.

    `entry_point` must be a module-level function (picklable by reference) that
    accepts the kwargs and does the whole job, e.g. `main.run`. Each kwargs dict
    must NOT carry a `device`: the worker pins its own GPU and passes
    `cuda:0`, because inside the child only one device is visible.

    Failures are collected, not raised: one variant crashing must not lose the
    others' hours of GPU time. Every result is returned with its traceback so
    the caller can report and decide.

    Work is assigned round-robin up front, so a worker that finishes early does
    NOT steal from a slower one. That is a deliberate trade: static assignment
    keeps one process pinned to one device for its whole life, which is what
    makes `CUDA_VISIBLE_DEVICES` reliable. Order the variants so that expensive
    ones alternate (`refine` then `random`, not both `refine` first) if the
    imbalance matters.
    """
    if not variants:
        return []
    for label, kwargs in variants:
        if "device" in kwargs:
            raise ValueError(
                f"Variant {label!r} passes an explicit `device`. The worker pins "
                "one GPU per process and supplies 'cuda:0' itself; passing a "
                "device here would send both workers to the same card."
            )

    # `num_workers` wins when given, so a caller can force the multiprocess
    # path on a machine with no CUDA device (tests, CPU debugging). Clamping to
    # `visible_gpu_count()` unconditionally would yield zero workers there, and
    # a queue with no consumer deadlocks the collection loop below rather than
    # failing.
    workers = int(num_workers) if num_workers else visible_gpu_count()
    workers = min(workers, len(variants))
    if workers <= 1:
        return _run_serially(variants, entry_point)
    context = _multiprocessing.get_context("spawn")
    result_queue = context.Queue()

    # Deal the variants out round-robin, one pre-filled queue per worker,
    # instead of sharing a single queue. With a shared queue whichever process
    # wins the startup race drains every item before the others finish
    # importing torch, so one GPU does all the work and the other sits idle —
    # the exact waste this module exists to remove.
    assignments: List[List[Tuple[str, Dict[str, Any]]]] = [[] for _ in range(workers)]
    for position, item in enumerate(variants):
        assignments[position % workers].append(item)

    print(f"[parallel] {len(variants)} variants over {workers} GPUs")
    processes = []
    for index, assigned in enumerate(assignments):
        processes.append(context.Process(
            target=_worker,
            args=(list(assigned), result_queue, index, entry_point),
            daemon=False,
        ))
        print(f"[parallel] cuda:{index} <- {', '.join(label for label, _ in assigned)}")
    for process in processes:
        process.start()

    results: List[WorkerResult] = []
    # Collect before joining: a full result queue blocks the child at exit,
    # which would deadlock a join-first ordering.
    for _ in range(len(variants)):
        results.append(result_queue.get())
    for process in processes:
        process.join()

    for result in results:
        status = "ok" if result["ok"] else "FAILED"
        print(f"[parallel] {result['label']}: {status} on cuda:{result['device']}")
        if not result["ok"]:
            print(result["error"])
    return results


def _run_serially(
    variants: Sequence[Tuple[str, Dict[str, Any]]],
    entry_point: Callable[..., Any],
) -> List[WorkerResult]:
    """One process, one device. Used when only one GPU or one variant exists."""
    import time

    results: List[WorkerResult] = []
    for label, kwargs in variants:
        started = time.time()
        print(f"[serial] {label}")
        try:
            entry_point(**kwargs)
            results.append(WorkerResult(
                label=label, ok=True, seconds=time.time() - started,
                error=None, device=0,
            ))
        except Exception:
            results.append(WorkerResult(
                label=label, ok=False, seconds=time.time() - started,
                error=traceback.format_exc(), device=0,
            ))
            print(results[-1]["error"])
    return results
