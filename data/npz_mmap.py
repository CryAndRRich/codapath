"""Turn a .npz of image arrays into memory-mappable .npy files.

Why this exists: `np.load(mmap_mode=...)` is silently ignored for a .npz.
`NpzFile.__getitem__` opens the member through `zipfile` and hands it to
`format.read_array`, so every access materialises the whole array in RAM no
matter what `mmap_mode` says. Only a standalone .npy can actually be mapped.

That distinction is worth 15 GiB on PathMNIST-224. Two extraction workers on a
Kaggle T4 x2 session (~30 GiB RAM) each holding their own eager copy sit exactly
on the limit, and the loser is OOM-killed inside `np.load` before it prints a
single line. Exported once to .npy, the pixels stay on disk and the OS page
cache serves both workers from one copy.

The export is written once per .npz and reused, so the cost is paid on the first
run of a session and never again.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional

import numpy as np

# Only these are large enough to be worth mapping; labels are kilobytes.
IMAGE_KEYS = ("train_images", "val_images", "test_images")
LABEL_KEYS = ("train_labels", "val_labels", "test_labels")

SCHEMA_VERSION = 1


def _export_key(npz_path: str) -> str:
    """A directory name that changes when the source .npz changes.

    Keyed on path + size + mtime rather than a content hash: hashing 15 GiB to
    decide whether to re-export 15 GiB defeats the purpose.
    """
    stat = os.stat(npz_path)
    signature = f"{os.path.realpath(npz_path)}|{stat.st_size}|{int(stat.st_mtime)}"
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    return f"{stem}_mmap_{digest}"


def export_dir_for(npz_path: str, cache_root: str) -> str:
    return os.path.join(cache_root, _export_key(npz_path))


def export_npz_to_npy(
    npz_path: str,
    cache_root: str,
    verbose: bool = True,
) -> str:
    """Write each array of `npz_path` as its own .npy under `cache_root`.

    Returns the export directory. A completed export is detected by its manifest
    and reused; the manifest is written last, so an export interrupted part-way
    is not mistaken for a finished one.

    Arrays are copied in row blocks rather than read whole: the point is to end
    up with a mappable file without ever holding the full array in RAM, and
    `np.load(npz)["train_images"]` would defeat that on its own. `NpzFile` gives
    no way to read a member partially, so the peak here is one array, not all
    three — enough to make a single worker fit where two eager copies would not.
    """
    export_dir = export_dir_for(npz_path, cache_root)
    manifest_path = os.path.join(export_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") == SCHEMA_VERSION:
            if verbose:
                print(f"[npz-mmap] reusing export → {export_dir}")
            return export_dir
        if verbose:
            print("[npz-mmap] export schema changed; rewriting")

    os.makedirs(export_dir, exist_ok=True)
    shapes: Dict[str, list] = {}
    suffix = f".tmp{os.getpid()}"
    with np.load(npz_path) as data:
        available = list(data.files)
        for key in available:
            array = data[key]
            target = os.path.join(export_dir, f"{key}.npy")
            temporary = target + suffix
            # open_memmap + block copy keeps the write out of RAM. `array` is
            # already materialised by NpzFile, so this bounds the peak at one
            # array; without the export it would be all of them, twice over.
            mapped = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=array.dtype, shape=array.shape
            )
            block = max(1, 1024 if array.ndim > 1 else len(array))
            for start in range(0, len(array), block):
                mapped[start:start + block] = array[start:start + block]
            mapped.flush()
            del mapped, array
            os.replace(temporary, target)
            shapes[key] = list(np.load(target, mmap_mode="r").shape)
            if verbose:
                size_gib = os.path.getsize(target) / 2**30
                print(f"[npz-mmap] {key:14} {tuple(shapes[key])} {size_gib:.2f} GiB")

    manifest_temporary = manifest_path + suffix
    with open(manifest_temporary, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "source": os.path.realpath(npz_path),
                "keys": sorted(shapes),
                "shapes": shapes,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    os.replace(manifest_temporary, manifest_path)
    if verbose:
        print(f"[npz-mmap] export complete → {export_dir}")
    return export_dir


def load_mmap_arrays(export_dir: str) -> Dict[str, np.ndarray]:
    """Memory-map every array of a completed export."""
    manifest_path = os.path.join(export_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"No completed .npy export at {export_dir}; call export_npz_to_npy first"
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    arrays = {}
    for key in manifest["keys"]:
        path = os.path.join(export_dir, f"{key}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Export declares {key} but {path} is missing")
        arrays[key] = np.load(path, mmap_mode="r")
    return arrays


def open_npz_mmap(
    npz_path: str,
    cache_root: Optional[str],
    verbose: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    """Return memory-mapped arrays for `npz_path`, exporting them if needed.

    Returns None when `cache_root` is None (mapping disabled) or when the export
    cannot be written — a read-only or full disk is a reason to fall back to the
    eager path, not to fail the run.
    """
    if cache_root is None:
        return None
    try:
        export_dir = export_npz_to_npy(npz_path, cache_root, verbose=verbose)
        return load_mmap_arrays(export_dir)
    except (OSError, ValueError) as exc:
        if verbose:
            print(f"[npz-mmap] falling back to eager load: {exc}")
        return None
