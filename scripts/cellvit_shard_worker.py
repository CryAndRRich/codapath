"""Run one CellViT extraction shard as a subprocess pinned to one GPU.

`utils.parallel` spawns one worker per GPU and pins `CUDA_VISIBLE_DEVICES`
before the target runs. The target here re-invokes
`scripts/extract_cellvit_features.py` as a child process rather than importing
it: that script is a `main()`-style CLI whose CellViT model, postprocessor and
DINO encoder all live in module-local state, and running it out-of-process
guarantees every bit of that state — including the CUDA context — is torn down
when the shard finishes.

Module-level so `spawn` can pickle the target by reference.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def _command(options: Dict[str, object], extra: List[str]) -> List[str]:
    command = [sys.executable, os.path.join(SCRIPT_DIR, "extract_cellvit_features.py")]
    for key, value in options.items():
        if value is None:
            continue
        command += [f"--{key}", str(value)]
    return command + extra


def run_cellvit_shard(
    options: Dict[str, object],
    shard_index: Optional[int] = None,
    shard_count: int = 1,
    assemble_only: bool = False,
    overwrite: bool = False,
    save_instance_maps: bool = False,
    skip_crop_dino: bool = False,
    device_string: str = "cuda:0",
) -> None:
    """Extract one shard (or assemble all of them) in a child process.

    `device_string` is accepted and forwarded as `--device` for symmetry with
    the other worker entry points; inside a pinned worker the visible card is
    always `cuda:0`.
    """
    extra: List[str] = ["--device", device_string]
    if shard_index is not None:
        extra += ["--shard_index", str(shard_index), "--shard_count", str(shard_count)]
    if assemble_only:
        extra.append("--assemble_only")
    if overwrite:
        extra.append("--overwrite")
    if save_instance_maps:
        # Boolean flags cannot ride in `options`, which is rendered as
        # --key value pairs; they are appended here alongside the other switches.
        extra.append("--save_instance_maps")
    if skip_crop_dino:
        extra.append("--skip_crop_dino")
    command = _command(options, extra)
    label = "assembly" if assemble_only else f"shard {shard_index}/{shard_count}"
    print(f"[cellvit-shard] {label}:", " ".join(command))

    # Capture rather than inherit stderr. Two shards run concurrently and both
    # write to the notebook's single stream, so an inherited traceback is
    # interleaved with the other shard's progress lines and, in a long run, is
    # scrolled or truncated away entirely — leaving only CalledProcessError with
    # no cause. Re-raising with the child's own output attached is what makes a
    # failed shard diagnosable.
    process = subprocess.run(
        command, cwd=PROJECT_DIR, text=True,
        stdout=None,                 # progress still streams live
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        details = (process.stderr or "").strip() or "(child produced no stderr)"
        raise RuntimeError(
            f"CellViT {label} failed with exit status {process.returncode}.\n"
            f"--- child stderr ---\n{details}\n--- end child stderr ---"
        )


def build_shard_jobs(
    options: Dict[str, object],
    shard_count: int,
    overwrite: bool = False,
    save_instance_maps: bool = False,
    skip_crop_dino: bool = False,
) -> List[Tuple[str, dict]]:
    """`(label, kwargs)` pairs for `run_variants_parallel`, one per shard.

    No `device` key: `run_variants_parallel` rejects it, because the worker pins
    its own card and supplies `cuda:0` itself.
    """
    dataset = options.get("dataset", "dataset")
    seed = options.get("seed", "?")
    return [
        (
            f"{dataset}-seed{seed}-cellvit-shard{index}",
            dict(
                options=dict(options),
                shard_index=index,
                shard_count=shard_count,
                overwrite=overwrite,
                save_instance_maps=save_instance_maps,
                skip_crop_dino=skip_crop_dino,
            ),
        )
        for index in range(shard_count)
    ]
