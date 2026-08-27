"""Sharded DINOv2 visual-feature extraction for a Kaggle T4 x2 session.

`extract_shard_on_worker` is the entry point `utils.parallel` dispatches: one
process per GPU, each extracting a contiguous row range. It must be a
module-level function so `spawn` can pickle it by reference.

Import-only module; the notebook drives it. It exists so the worker target lives
outside the notebook, which `spawn` cannot import from.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def extract_shard_on_worker(
    data_path: str,
    dataset_key: str,
    seed: int,
    vit_name: str,
    shard_index: int,
    shard_count: int,
    cache_dir: str,
    device_string: str = "cuda:0",
) -> None:
    """Extract one shard, seeding and resolving the device inside the child.

    torch is imported here rather than at module scope: `utils.parallel` pins
    `CUDA_VISIBLE_DEVICES` in the child before the target runs, and that only
    takes effect if torch initialises afterwards.

    `set_seed` runs per process. The forward pass itself is deterministic under
    `inference_mode` with a frozen backbone, but the train/test split of an
    ImageFolder dataset is drawn from a seeded generator inside
    `get_data_loaders`, so both workers must seed identically or they would
    shard two different splits and the assembled cache would interleave them.
    """
    import torch

    from data.loaders import get_data_loaders
    from features.visual import extract_features_shard
    from utils import set_seed

    set_seed(seed)
    device = torch.device(device_string)
    train_loader, test_loader, _ = get_data_loaders(data_path, seed, verbose=False)
    print(
        f"[worker] {dataset_key} seed={seed} shard={shard_index}/{shard_count} "
        f"train={len(train_loader.dataset)} test={len(test_loader.dataset)}"
    )
    extract_features_shard(
        train_loader, test_loader, dataset_key, seed, vit_name, device,
        shard_index=shard_index, shard_count=shard_count, cache_dir=cache_dir,
    )


def build_shard_jobs(
    data_path: str,
    dataset_key: str,
    seed: int,
    vit_name: str,
    shard_count: int,
    cache_dir: str,
) -> List[Tuple[str, dict]]:
    """`(label, kwargs)` pairs for `run_variants_parallel`, one per shard.

    No `device` key: the worker pins its own GPU and passes `cuda:0` itself.
    """
    return [
        (
            f"{dataset_key}-seed{seed}-shard{index}",
            dict(
                data_path=data_path,
                dataset_key=dataset_key,
                seed=seed,
                vit_name=vit_name,
                shard_index=index,
                shard_count=shard_count,
                cache_dir=cache_dir,
            ),
        )
        for index in range(shard_count)
    ]
