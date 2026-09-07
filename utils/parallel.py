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

That same import is why placement does NOT go through `CUDA_VISIBLE_DEVICES`:
unpickling the target imports its module, which imports torch, so anything the
worker body sets afterwards is too late. Each worker instead names its card
explicitly as `device_string="cuda:<index>"`. See `_worker`.
"""

from __future__ import annotations

import multiprocessing as _multiprocessing
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

    A worker owns exactly one device for its whole life, and it says so with an
    explicit `cuda:<index>` rather than through `CUDA_VISIBLE_DEVICES`.

    **Why the environment variable is not used.** Setting it here is too late,
    because `spawn` must unpickle `entry_point` before this function body
    runs, and unpickling a function reference imports its module -- and
    `main.py` does `import torch` at module level. Torch is
    therefore already imported by the time any line here executes. That was
    enough for both workers to end up on the SAME card on a real Kaggle T4 x2
    run: the OOM named two processes on one 14.56 GiB device (4.12 GiB +
    10.24 GiB), with byte-identical numbers across two separate runs.

    Setting it anyway would be worse than not setting it: with the variable in
    effect the child sees one card as index 0, so an explicit `cuda:1` would
    point past the end of the visible devices. The two mechanisms cannot both
    be half-applied -- this picks the one whose timing does not depend on when
    torch was imported.

    So the device index is passed to the work itself, as `device_string`, and
    a variant that carries that key has it OVERWRITTEN with `cuda:<index>`.
    Overwritten, not defaulted: both run notebooks pass a literal
    `device_string="cuda:0"` in their base kwargs (correct back when
    `CUDA_VISIBLE_DEVICES` made every worker's own card device 0), and
    honouring it is exactly how both shards landed on card 0. The worker is
    the only party that knows which card it owns, so it wins.

    Only rewritten when the key is already there: `entry_point` is any
    callable, and several (the CellViT shard worker, this module's own tests)
    take no such parameter -- injecting it unconditionally makes them raise
    `TypeError` before doing any work.
    """
    import time

    for label, kwargs in assigned:
        if "device_string" in kwargs:
            kwargs = dict(kwargs, device_string=f"cuda:{device_index}")
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
    abort_on_failure: bool = False,
) -> List[WorkerResult]:
    """Run `(label, kwargs)` variants across GPUs, returning one result each.

    `entry_point` must be a module-level function (picklable by reference) that
    accepts the kwargs and does the whole job, e.g. `main.run`. Each kwargs dict
    must NOT carry a `device`: the worker pins its own GPU and passes
    `cuda:0`, because inside the child only one device is visible.

    Failures are collected, not raised: one variant crashing must not lose the
    others' hours of GPU time. Every result is returned with its traceback so
    the caller can report and decide.

    `abort_on_failure=True` inverts that trade, and exists for BUDGET SHARDS.
    Independent variants are worth finishing individually, but two shards of a
    single run are merged afterwards, so losing one makes the other's hours
    worthless -- measured on a CONCH+LoRA run where shard1 died at 6.3 minutes
    and shard0 kept going for 3.35 hours before the caller could assert. With
    this set, the first failure terminates the surviving workers instead. Those
    workers are killed mid-budget, so their results are reported as aborted
    rather than successful: a partial sweep must never merge as if complete.

    Work is assigned round-robin up front, so a worker that finishes early does
    NOT steal from a slower one. That is a deliberate trade: static assignment
    keeps one process pinned to one device for its whole life, which is what
    makes the explicit per-worker device index correct. Order so that expensive
    ones alternate (`refine` then `random`, not both `refine` first) if the
    imbalance matters.
    """
    if not variants:
        return []
    for label, kwargs in variants:
        if "device" in kwargs:
            raise ValueError(
                f"Variant {label!r} passes an explicit `device`. The worker owns "
                "one GPU per process and supplies its own `device_string`; passing a "
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
    aborted = False
    for _ in range(len(variants)):
        result = result_queue.get()
        results.append(result)
        if abort_on_failure and not result["ok"]:
            # Kill the peers now rather than paying for work whose only
            # consumer is a merge that can no longer happen. `terminate` is
            # the right blunt instrument here: the child owns a CUDA context
            # and its own files, and there is no partial state worth draining.
            print(f"[parallel] {result['label']} failed -- aborting the "
                  "remaining shards; their partial sweeps cannot be merged")
            for process in processes:
                if process.is_alive():
                    process.terminate()
            aborted = True
            break
    for process in processes:
        process.join()

    if aborted:
        # Name every variant that never reported, so the caller's failure list
        # is the whole truth. Without this the run looks like one failed shard
        # and one that simply vanished.
        reported = {result["label"] for result in results}
        for label, _kwargs in variants:
            if label not in reported:
                results.append(WorkerResult(
                    label=label, ok=False, seconds=0.0, device=-1,
                    error="aborted: a peer shard of this run failed, so this "
                          "shard was terminated before it could finish",
                ))

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
