"""Sharded CONCH (VLM) feature extraction for a Kaggle T4 x2 session.

`extract_vlm_shard_on_worker` is the entry point `utils.parallel` dispatches:
one process per GPU, each extracting a contiguous row range of both splits.
It must be a module-level function so `spawn` can pickle it by reference.

Import-only module; the notebook drives it. It exists so the worker target
lives outside the notebook, which `spawn` cannot import from -- the same
reason `scripts/extract_visual_features.py` exists.

**Each worker loads its own CONCH checkpoint.** Unlike the DINOv2 notebook,
the model cannot be loaded in the parent and passed down: a CUDA-initialised
module does not survive `spawn` pickling, and the parent must not touch CUDA
before forking workers anyway. The checkpoint is downloaded once in the
parent (into the shared HF cache), so each worker's `load_conch` is a local
read, not a second download.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def extract_vlm_shard_on_worker(
    data_path: str,
    dataset_key: str,
    seed: int,
    vlm_name: str,
    shard_index: int,
    shard_count: int,
    cache_dir: str,
    batch_size: int = 64,
    mmap_cache_dir: str = None,
    hf_token: str = None,
    device_string: str = "cuda:0",
) -> None:
    """Extract one shard, seeding and resolving the device inside the child.

    torch is imported here rather than at module scope: `utils.parallel` pins
    `CUDA_VISIBLE_DEVICES` in the child before the target runs, and that only
    takes effect if torch initialises afterwards.

    `set_seed` runs per process. The forward pass is deterministic under
    `inference_mode` with a frozen backbone, but the train/test split of an
    ImageFolder dataset is drawn from a seeded generator inside
    `get_data_loaders`, so both workers must seed identically or they would
    shard two different splits and the assembled cache would interleave them.

    The loaders are built with CONCH's OWN `preprocess` (448x448 + OpenAI CLIP
    normalization), never `get_data_loaders`'s DINOv2 default -- a hand-rolled
    equivalent does not crash, it just silently shifts every embedding
    (features/vlm.py module docstring).

    `mmap_cache_dir` is what keeps two workers inside a Kaggle session's RAM.
    An eager .npz read costs ~15 GiB per process on PathMNIST-224, so two of
    them exceed the ~30 GiB available and one is OOM-killed inside `np.load`
    before it prints anything. The export it points at must already exist:
    building it here would have both workers writing the same files at once.
    """
    import torch

    from data.loaders import get_data_loaders
    from features.vlm import extract_vlm_features_shard, load_conch
    from utils import set_seed

    set_seed(seed)
    device = torch.device(device_string)

    # The model comes FIRST: building a loader with the right pixels needs
    # CONCH's preprocess, which only exists after the checkpoint is loaded.
    model, preprocess = load_conch(vlm_name, device, hf_token=hf_token or None)

    train_loader, test_loader, _ = get_data_loaders(
        data_path, seed, verbose=False, mmap_cache_dir=mmap_cache_dir,
        transform=preprocess,
    )
    # `get_data_loaders`'s own batch size assumes DINOv2's 224x224; CONCH's
    # 448x448 is 4x the pixels, so rebuild at the caller's batch_size. VRAM,
    # not RAM, is the binding constraint here -- a T4 has 16 GiB and this is
    # the knob that fits the forward pass into it.
    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=batch_size, shuffle=False,
        num_workers=train_loader.num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_loader.dataset, batch_size=batch_size, shuffle=False,
        num_workers=test_loader.num_workers, pin_memory=True,
    )

    mapped = getattr(train_loader.dataset, "mmap", None)
    print(
        f"[worker] {dataset_key} seed={seed} shard={shard_index}/{shard_count} "
        f"train={len(train_loader.dataset)} test={len(test_loader.dataset)} "
        f"batch={batch_size} mmap={mapped}"
    )
    extract_vlm_features_shard(
        train_loader, test_loader, dataset_key, seed, vlm_name, device,
        shard_index=shard_index, shard_count=shard_count, cache_dir=cache_dir,
        model=model,
    )


def build_vlm_shard_jobs(
    data_path: str,
    dataset_key: str,
    seed: int,
    vlm_name: str,
    shard_count: int,
    cache_dir: str,
    batch_size: int = 64,
    mmap_cache_dir: str = None,
    hf_token: str = None,
) -> List[Tuple[str, dict]]:
    """`(label, kwargs)` pairs for `run_variants_parallel`, one per shard.

    No `device` key: the worker pins its own GPU and passes `cuda:0` itself.
    """
    return [
        (
            f"{dataset_key}-seed{seed}-vlmshard{index}",
            dict(
                data_path=data_path,
                dataset_key=dataset_key,
                seed=seed,
                vlm_name=vlm_name,
                shard_index=index,
                shard_count=shard_count,
                cache_dir=cache_dir,
                batch_size=batch_size,
                mmap_cache_dir=mmap_cache_dir,
                hf_token=hf_token,
            ),
        )
        for index in range(shard_count)
    ]
