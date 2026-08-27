"""Compressed per-patch instance maps, for visualisation.

The instance map is CellViT's raw segmentation output: an integer label per
pixel, 0 for background and the instance id elsewhere. It is what you overlay on
a tile to see which nucleus the model actually found, so it is worth keeping —
but it is also the largest thing the extractor produces.

Stored raw it does not fit. PathMNIST is resized 0.5 -> 0.25 MPP, so a patch is
448x448; at uint16 that is 392 KiB per patch, 37 GiB over 100k patches, against
a ~20 GB Kaggle Output quota. A label image is mostly background and piecewise
constant, which is exactly what DEFLATE handles well: measured 227-298x on
capped patches (~0.15 GiB for 100k) and still 27x in a deliberately dense
worst case (200 nuclei, jagged borders).

So each map is zlib-compressed separately and the blobs are concatenated into
one file with an offsets index. Per-patch rather than one big stream so that
reading patch 40,000 costs one seek and one decompress, not 40,000.

This is deliberately NOT part of the sampler's cache contract. Nothing in
`sampling/` reads these; `load_cellvit_cache` neither requires nor validates
them. They are an optional sidecar, so a cache built before this existed stays
valid and a run that does not need pictures can skip writing them.
"""

from __future__ import annotations

import json
import os
import zlib
from typing import Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 1
BLOB_FILE = "instance_maps.bin"
INDEX_FILE = "instance_maps_index.npy"
META_FILE = "instance_maps.json"

# Level 6 is zlib's default. Level 9 measured under 3% smaller on label images
# for a large CPU cost, and extraction is already the long pole.
COMPRESSION_LEVEL = 6

# uint16 caps ids at 65535 per patch, far above any plausible nucleus count, and
# halves the pre-compression size versus the int32 the postprocessor returns.
STORED_DTYPE = np.uint16


class InstanceMapWriter:
    """Append instance maps to one blob file, recording offsets and shapes.

    Used by the extraction shard workers: each writes its own blob, and the
    assembly pass concatenates them in patch order.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._handle = open(path, "wb")
        self.offsets = [0]
        self.shapes: list = []

    def append(self, instance_map: np.ndarray) -> None:
        array = np.asarray(instance_map)
        if array.ndim != 2:
            raise ValueError(f"instance map must be 2-D, got shape {array.shape}")
        peak = int(array.max()) if array.size else 0
        if peak > np.iinfo(STORED_DTYPE).max:
            raise ValueError(
                f"instance id {peak} exceeds {STORED_DTYPE.__name__} range; "
                "this cache format cannot represent it"
            )
        payload = zlib.compress(
            np.ascontiguousarray(array, dtype=STORED_DTYPE).tobytes(),
            COMPRESSION_LEVEL,
        )
        self._handle.write(payload)
        self.offsets.append(self.offsets[-1] + len(payload))
        self.shapes.append((int(array.shape[0]), int(array.shape[1])))

    def close(self) -> Tuple[np.ndarray, np.ndarray]:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        return (
            np.asarray(self.offsets, dtype=np.int64),
            np.asarray(self.shapes, dtype=np.int32).reshape(-1, 2),
        )

    def __enter__(self) -> "InstanceMapWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if not self._handle.closed:
            self._handle.close()
        return False


def write_index(
    cache_dir: str,
    offsets: np.ndarray,
    shapes: np.ndarray,
    num_patches: int,
) -> None:
    """Write the index and metadata that make a blob file readable.

    Written last, and only after the blob is complete: the metadata file is what
    `read_instance_map` treats as proof the sidecar is usable.
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    shapes = np.asarray(shapes, dtype=np.int32).reshape(-1, 2)
    if len(offsets) != num_patches + 1:
        raise ValueError(
            f"offsets must have {num_patches + 1} entries, got {len(offsets)}"
        )
    if len(shapes) != num_patches:
        raise ValueError(f"shapes must have {num_patches} rows, got {len(shapes)}")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("offsets must start at zero and be non-decreasing")

    blob_size = os.path.getsize(os.path.join(cache_dir, BLOB_FILE))
    if int(offsets[-1]) != blob_size:
        raise ValueError(
            f"index claims {int(offsets[-1])} bytes but {BLOB_FILE} holds "
            f"{blob_size}; the blob is truncated or the index is stale"
        )

    suffix = f".tmp{os.getpid()}"
    index_path = os.path.join(cache_dir, INDEX_FILE)
    np.save(index_path + suffix, np.concatenate([offsets[:, None],
                                                 np.vstack([shapes, [[0, 0]]])], axis=1))
    os.replace(index_path + suffix + ".npy", index_path)

    meta_path = os.path.join(cache_dir, META_FILE)
    with open(meta_path + suffix, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "num_patches": int(num_patches),
                "dtype": STORED_DTYPE.__name__,
                "compression": "zlib",
                "compression_level": COMPRESSION_LEVEL,
                "total_bytes": int(offsets[-1]),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    os.replace(meta_path + suffix, meta_path)


def has_instance_maps(cache_dir: str) -> bool:
    return all(
        os.path.exists(os.path.join(cache_dir, name))
        for name in (BLOB_FILE, INDEX_FILE, META_FILE)
    )


def read_instance_map(cache_dir: str, patch_index: int) -> np.ndarray:
    """Decompress one patch's instance map.

    One seek plus one decompress, independent of `patch_index`, so browsing a
    cache of 100k maps stays interactive.
    """
    if not has_instance_maps(cache_dir):
        raise FileNotFoundError(
            f"No instance-map sidecar in {cache_dir}. Re-run extraction with "
            "--save_instance_maps to produce one."
        )
    with open(os.path.join(cache_dir, META_FILE), "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported instance-map schema: {meta.get('schema_version')}")
    num_patches = int(meta["num_patches"])
    if not 0 <= patch_index < num_patches:
        raise IndexError(f"patch_index {patch_index} outside [0, {num_patches})")

    index = np.load(os.path.join(cache_dir, INDEX_FILE), mmap_mode="r")
    start, stop = int(index[patch_index, 0]), int(index[patch_index + 1, 0])
    height, width = int(index[patch_index, 1]), int(index[patch_index, 2])
    with open(os.path.join(cache_dir, BLOB_FILE), "rb") as handle:
        handle.seek(start)
        payload = handle.read(stop - start)
    flat = np.frombuffer(zlib.decompress(payload), dtype=STORED_DTYPE)
    if flat.size != height * width:
        raise ValueError(
            f"patch {patch_index}: decompressed {flat.size} pixels, "
            f"expected {height * width}"
        )
    return flat.reshape(height, width)


def merge_blobs(
    cache_dir: str,
    shard_paths: Sequence[str],
    shard_indexes: Sequence[Tuple[np.ndarray, np.ndarray]],
    num_patches: int,
) -> None:
    """Concatenate per-shard blobs in patch order into the final sidecar.

    Byte ranges are rebased as each shard's blob is appended, because a shard's
    own offsets start at zero.
    """
    target = os.path.join(cache_dir, BLOB_FILE)
    suffix = f".tmp{os.getpid()}"
    merged_offsets = [0]
    merged_shapes = []
    with open(target + suffix, "wb") as out:
        for path, (offsets, shapes) in zip(shard_paths, shard_indexes):
            with open(path, "rb") as source:
                while True:
                    block = source.read(8 * 1024 * 1024)
                    if not block:
                        break
                    out.write(block)
            for length in np.diff(np.asarray(offsets, dtype=np.int64)):
                merged_offsets.append(merged_offsets[-1] + int(length))
            merged_shapes.extend(np.asarray(shapes, dtype=np.int32).reshape(-1, 2).tolist())
        out.flush()
        os.fsync(out.fileno())
    os.replace(target + suffix, target)
    write_index(
        cache_dir,
        np.asarray(merged_offsets, dtype=np.int64),
        np.asarray(merged_shapes, dtype=np.int32).reshape(-1, 2),
        num_patches,
    )


def overlay_boundaries(
    rgb: np.ndarray,
    instance_map: np.ndarray,
    colour: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Draw instance boundaries onto an RGB copy — the usual QC picture."""
    colour = np.asarray(colour if colour is not None else [255, 0, 0], dtype=np.uint8)
    boundary = np.zeros(instance_map.shape, dtype=bool)
    boundary[1:] |= instance_map[1:] != instance_map[:-1]
    boundary[:, 1:] |= instance_map[:, 1:] != instance_map[:, :-1]
    boundary &= instance_map > 0
    overlay = np.array(rgb, dtype=np.uint8, copy=True)
    overlay[boundary] = colour
    return overlay
