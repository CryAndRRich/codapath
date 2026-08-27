import os
import json
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import Dinov2Model


class DINOv2Extractor(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-base") -> None:
        super().__init__()
        self.encoder = Dinov2Model.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values=x).last_hidden_state[:, 0, :]


@torch.inference_mode()
def extract_image_features(dataloader: DataLoader,
                            extractor: nn.Module,
                            device: torch.device) -> np.ndarray:
    extractor = extractor.to(device)
    extractor.eval()

    all_features = []
    for images, _ in tqdm(dataloader, desc="Extracting image features", leave=False):
        images = images.to(device, non_blocking=True)
        features = extractor(images)
        all_features.append(features.cpu().numpy().astype(np.float32))

    return np.vstack(all_features)


def _feature_cache_paths(cache_dir: str,
                         dataset_key: str,
                         seed: int,
                         vit_name: str):
    """Cache filenames keyed by dataset + seed + backbone.

    Seed is part of the key because ImageFolder datasets are split with a
    seeded generator in get_data_loaders — a different seed reorders samples,
    so features are only reusable when the seed matches.
    """
    safe_vit = vit_name.replace("/", "_")
    base = f"{dataset_key}_seed{seed}_{safe_vit}"
    train_path = os.path.join(cache_dir, f"{base}_train.npy")
    test_path  = os.path.join(cache_dir, f"{base}_test.npy")
    manifest_path = os.path.join(cache_dir, f"{base}_manifest.json")
    return train_path, test_path, manifest_path


def get_or_extract_features(train_loader: DataLoader,
                            test_loader: DataLoader,
                            dataset_key: str,
                            seed: int,
                            vit_name: str,
                            device: torch.device,
                            cache_dir: str = "features",
                            train_fingerprint: str = None,
                            test_fingerprint: str = None):
    """Return (train_features, test_features) for the frozen DINOv2 backbone.

    Loads them from `cache_dir` when a valid cache exists (matching dataset,
    seed, backbone AND sample count); otherwise builds the extractor, computes
    the features, caches them to disk, and frees the model. The DINOv2 model is
    only instantiated on a cache miss, so cached runs never pay download/GPU
    cost.
    """
    train_path, test_path, manifest_path = _feature_cache_paths(
        cache_dir, dataset_key, seed, vit_name
    )

    n_train = len(train_loader.dataset)
    n_test  = len(test_loader.dataset)

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_features = np.load(train_path)
        test_features  = np.load(test_path)
        cache_valid = (
            train_features.shape[0] == n_train
            and test_features.shape[0] == n_test
        )
        if cache_valid and os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            cache_valid = (
                manifest.get("dataset") == dataset_key
                and manifest.get("seed") == seed
                and manifest.get("backbone") == vit_name
                and (
                    train_fingerprint is None
                    or manifest.get("train_fingerprint") == train_fingerprint
                )
                and (
                    test_fingerprint is None
                    or manifest.get("test_fingerprint") == test_fingerprint
                )
            )
        elif cache_valid and (train_fingerprint is not None or test_fingerprint is not None):
            cache_valid = False
            print(
                "[features] Legacy cache has no sample-order manifest; "
                "re-extracting to guarantee exact row/sample alignment."
            )

        if cache_valid:
            print(f"[features] Loaded cache → {train_path} "
                  f"({train_features.shape}) + {test_path} ({test_features.shape})")
            return train_features, test_features
        print("[features] Cache metadata/order mismatch — re-extracting.")

    print(f"[features] Cache miss — extracting DINOv2 features for '{dataset_key}' "
          f"(seed={seed}, backbone={vit_name}).")
    extractor = DINOv2Extractor(model_name=vit_name).to(device)
    train_features = extract_image_features(train_loader, extractor, device)
    test_features  = extract_image_features(test_loader,  extractor, device)
    del extractor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    os.makedirs(cache_dir, exist_ok=True)
    # Write through a process-unique temporary file and rename. Two workers
    # sharing one cache directory (a T4 x2 session running two variants) can
    # both miss the cache and extract at once; a direct `np.save` to the same
    # path would interleave and leave a corrupt array that still loads. Rename
    # is atomic on the same filesystem, so a reader sees either the old file or
    # a complete new one. The manifest is written LAST, because it is what
    # marks a cache as trustworthy.
    suffix = f".tmp{os.getpid()}"
    for path, array in ((train_path, train_features), (test_path, test_features)):
        temporary = path + suffix
        np.save(temporary, array)
        # np.save appends .npy when the name lacks it.
        os.replace(temporary if temporary.endswith(".npy") else temporary + ".npy", path)
    manifest_temporary = manifest_path + suffix
    with open(manifest_temporary, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": 1,
            "dataset": dataset_key,
            "seed": seed,
            "backbone": vit_name,
            "train_fingerprint": train_fingerprint,
            "test_fingerprint": test_fingerprint,
            "train_shape": list(train_features.shape),
            "test_shape": list(test_features.shape),
        }, f, indent=2, sort_keys=True)
    os.replace(manifest_temporary, manifest_path)
    print(f"[features] Saved cache → {train_path} + {test_path}")

    return train_features, test_features


def _shard_bounds(total: int, shard_index: int, shard_count: int):
    """Contiguous [start, stop) row range for one shard of `total` rows.

    Contiguous, not strided: `extract_image_features` batches rows in dataloader
    order, and a strided split would put different samples in a batch than a
    serial run does. For a fixed-size, batch-independent forward pass that makes
    no numerical difference, but it makes the shard outputs impossible to compare
    against a serial run row by row, which is how this split is tested. The
    remainder is spread over the first shards so the two T4s differ by at most
    one row.
    """
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index {shard_index} outside [0, {shard_count})")
    base, remainder = divmod(total, shard_count)
    start = shard_index * base + min(shard_index, remainder)
    stop = start + base + (1 if shard_index < remainder else 0)
    return start, stop


def _shard_paths(cache_dir: str, base: str, split: str, shard_count: int):
    shard_dir = os.path.join(cache_dir, f".shards_{base}_{split}_of{shard_count}")
    return shard_dir


def extract_features_shard(train_loader: DataLoader,
                           test_loader: DataLoader,
                           dataset_key: str,
                           seed: int,
                           vit_name: str,
                           device: torch.device,
                           shard_index: int,
                           shard_count: int,
                           cache_dir: str = "features") -> None:
    """Extract one contiguous row range of train AND test into a shard file.

    Written for a Kaggle T4 x2 session: two processes, one per GPU, each call
    this with its own `shard_index`, then one process calls
    `assemble_feature_shards` to concatenate them into the normal cache layout.
    Nothing is shared between the processes except the output directory, and
    each writes only its own file, so no locking is needed.

    A shard that already exists is skipped, so a session that dies half way
    through resumes instead of recomputing what it already has.
    """
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    train_path, test_path, _ = _feature_cache_paths(
        cache_dir, dataset_key, seed, vit_name
    )
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"[features] shard {shard_index}: complete cache exists, nothing to do")
        return

    base = os.path.basename(train_path)[:-len("_train.npy")]
    extractor = None
    for split, loader in (("train", train_loader), ("test", test_loader)):
        shard_dir = _shard_paths(cache_dir, base, split, shard_count)
        os.makedirs(shard_dir, exist_ok=True)
        shard_path = os.path.join(shard_dir, f"{shard_index:03d}.npy")
        start, stop = _shard_bounds(len(loader.dataset), shard_index, shard_count)
        if os.path.exists(shard_path):
            existing = np.load(shard_path, mmap_mode="r")
            if existing.shape[0] == stop - start:
                print(f"[features] shard {shard_index} {split}: reusing "
                      f"{existing.shape} rows [{start}:{stop})")
                continue
            print(f"[features] shard {shard_index} {split}: stale row count "
                  f"({existing.shape[0]} != {stop - start}), recomputing")
        if extractor is None:
            extractor = DINOv2Extractor(model_name=vit_name).to(device)
        # Subset over the loader's own dataset, so the rows this shard produces
        # are exactly rows [start, stop) of the serial cache.
        subset = Subset(loader.dataset, list(range(start, stop)))
        # Inherit the loader's worker count: whoever built it chose it for this
        # dataset kind (JPEG decode needs workers, a mapped .npz does not).
        shard_workers = int(getattr(loader, "num_workers", 0) or 0)
        shard_loader = DataLoader(
            subset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=shard_workers,
            pin_memory=True,
            **({"persistent_workers": True} if shard_workers > 0 else {}),
        )
        print(f"[features] shard {shard_index}/{shard_count} {split}: "
              f"rows [{start}:{stop}) of {len(loader.dataset)}")
        features = extract_image_features(shard_loader, extractor, device)
        if features.shape[0] != stop - start:
            raise RuntimeError(
                f"shard produced {features.shape[0]} rows, expected {stop - start}"
            )
        temporary = shard_path + f".tmp{os.getpid()}"
        np.save(temporary, features)
        os.replace(
            temporary if temporary.endswith(".npy") else temporary + ".npy",
            shard_path,
        )
    if extractor is not None:
        del extractor
        if device.type == "cuda":
            torch.cuda.empty_cache()


def assemble_feature_shards(dataset_key: str,
                            seed: int,
                            vit_name: str,
                            shard_count: int,
                            n_train: int,
                            n_test: int,
                            cache_dir: str = "features",
                            train_fingerprint: str = None,
                            test_fingerprint: str = None,
                            keep_shards: bool = False):
    """Concatenate per-GPU shards into the standard cache, manifest last.

    The manifest is what `get_or_extract_features` trusts, so it is written only
    after both arrays are complete and their row counts check out. A missing
    shard is an error rather than a short array: silently concatenating 3 of 4
    shards would produce a cache that loads and is wrong.
    """
    train_path, test_path, manifest_path = _feature_cache_paths(
        cache_dir, dataset_key, seed, vit_name
    )
    base = os.path.basename(train_path)[:-len("_train.npy")]
    assembled = {}
    for split, expected_rows, path in (
        ("train", n_train, train_path), ("test", n_test, test_path)
    ):
        shard_dir = _shard_paths(cache_dir, base, split, shard_count)
        parts = []
        for shard_index in range(shard_count):
            shard_path = os.path.join(shard_dir, f"{shard_index:03d}.npy")
            if not os.path.exists(shard_path):
                raise FileNotFoundError(
                    f"Missing {split} shard {shard_index} at {shard_path}. "
                    "Every shard must finish before assembly; re-run the "
                    "extraction cell to fill the gap."
                )
            start, stop = _shard_bounds(expected_rows, shard_index, shard_count)
            part = np.load(shard_path)
            if part.shape[0] != stop - start:
                raise RuntimeError(
                    f"{split} shard {shard_index} has {part.shape[0]} rows, "
                    f"expected {stop - start}"
                )
            parts.append(part)
        features = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        if features.shape[0] != expected_rows:
            raise RuntimeError(
                f"assembled {split} has {features.shape[0]} rows, "
                f"expected {expected_rows}"
            )
        if not np.all(np.isfinite(features)):
            raise RuntimeError(f"assembled {split} features are not all finite")
        assembled[split] = (path, features)

    suffix = f".tmp{os.getpid()}"
    for path, array in assembled.values():
        temporary = path + suffix
        np.save(temporary, array)
        os.replace(
            temporary if temporary.endswith(".npy") else temporary + ".npy", path
        )
    manifest_temporary = manifest_path + suffix
    with open(manifest_temporary, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": 1,
            "dataset": dataset_key,
            "seed": seed,
            "backbone": vit_name,
            "train_fingerprint": train_fingerprint,
            "test_fingerprint": test_fingerprint,
            "train_shape": list(assembled["train"][1].shape),
            "test_shape": list(assembled["test"][1].shape),
            "shard_count": shard_count,
        }, f, indent=2, sort_keys=True)
    os.replace(manifest_temporary, manifest_path)

    if not keep_shards:
        for split in ("train", "test"):
            shutil.rmtree(
                _shard_paths(cache_dir, base, split, shard_count),
                ignore_errors=True,
            )
    print(f"[features] Assembled {shard_count} shards → {train_path} + {test_path}")
    return assembled["train"][1], assembled["test"][1]
