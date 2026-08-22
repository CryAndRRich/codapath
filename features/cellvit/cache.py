"""Portable, order-checked cache for variable-size cell feature sets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np

from data.identity import sample_order_fingerprint


SCHEMA_VERSION = 1


def _all_finite(array: np.ndarray, chunk_size: int = 65536) -> bool:
    """Check large memmaps without allocating a full-size boolean array."""
    array = np.asarray(array)
    if array.ndim == 0:
        return bool(np.isfinite(array))
    return all(
        bool(np.all(np.isfinite(array[start:start + chunk_size])))
        for start in range(0, len(array), chunk_size)
    )


@dataclass(frozen=True)
class CellViTCache:
    offsets: np.ndarray
    cellvit_embeddings: Optional[np.ndarray]
    cell_dino_features: Optional[np.ndarray]
    confidence: np.ndarray
    bboxes: Optional[np.ndarray]
    sample_ids: np.ndarray
    manifest: Dict[str, Any]

    @property
    def num_patches(self) -> int:
        return len(self.offsets) - 1

    @property
    def num_cells(self) -> int:
        return int(self.offsets[-1])

    def features(self, source: str) -> np.ndarray:
        if source == "cellvit_embedding":
            value = self.cellvit_embeddings
        elif source == "crop_dino":
            value = self.cell_dino_features
        else:
            raise ValueError(
                "source must be 'cellvit_embedding' or 'crop_dino', "
                f"got {source!r}"
            )
        if value is None:
            raise ValueError(f"Cache does not contain features for source={source!r}")
        return value


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_arrays(
    offsets: np.ndarray,
    confidence: np.ndarray,
    cellvit_embeddings: Optional[np.ndarray],
    cell_dino_features: Optional[np.ndarray],
    bboxes: Optional[np.ndarray],
    sample_ids: Sequence[str],
) -> None:
    offsets = np.asarray(offsets)
    if offsets.ndim != 1 or len(offsets) != len(sample_ids) + 1:
        raise ValueError("offsets must have shape (num_patches + 1,)")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("offsets must start at zero and be non-decreasing")
    num_cells = int(offsets[-1])
    if len(confidence) != num_cells:
        raise ValueError("confidence length must equal offsets[-1]")
    if not _all_finite(confidence):
        raise ValueError("confidence must contain only finite values")
    for name, array in (
        ("cellvit_embeddings", cellvit_embeddings),
        ("cell_dino_features", cell_dino_features),
        ("bboxes", bboxes),
    ):
        if array is not None and len(array) != num_cells:
            raise ValueError(f"{name} first dimension must equal offsets[-1]")
    for name, array in (
        ("cellvit_embeddings", cellvit_embeddings),
        ("cell_dino_features", cell_dino_features),
    ):
        if array is not None:
            if array.ndim != 2:
                raise ValueError(f"{name} must have shape (num_cells, feature_dim)")
            if not _all_finite(array):
                raise ValueError(f"{name} must contain only finite values")
    if bboxes is not None and (bboxes.ndim != 2 or bboxes.shape[1] != 4):
        raise ValueError("bboxes must have shape (num_cells, 4)")


def save_cellvit_cache(
    cache_dir: str,
    *,
    offsets: np.ndarray,
    confidence: np.ndarray,
    sample_ids: Sequence[str],
    manifest: Dict[str, Any],
    cellvit_embeddings: Optional[np.ndarray] = None,
    cell_dino_features: Optional[np.ndarray] = None,
    bboxes: Optional[np.ndarray] = None,
) -> None:
    """Write a cache directory after validating its ragged-array contract."""
    offsets = np.asarray(offsets, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=np.float32)
    cellvit_embeddings = (
        None if cellvit_embeddings is None else np.asarray(cellvit_embeddings)
    )
    cell_dino_features = (
        None if cell_dino_features is None else np.asarray(cell_dino_features)
    )
    for name, array in (
        ("cellvit_embeddings", cellvit_embeddings),
        ("cell_dino_features", cell_dino_features),
    ):
        if array is not None and array.dtype not in (np.float16, np.float32):
            raise ValueError(f"{name} must use float16 or float32, got {array.dtype}")
    bboxes = None if bboxes is None else np.asarray(bboxes, dtype=np.int32)
    sample_ids_np = np.asarray(list(sample_ids), dtype=np.str_)

    _validate_arrays(
        offsets, confidence, cellvit_embeddings, cell_dino_features,
        bboxes, sample_ids_np,
    )
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "offsets.npy"), offsets)
    np.save(os.path.join(cache_dir, "confidence.npy"), confidence)
    np.save(os.path.join(cache_dir, "sample_ids.npy"), sample_ids_np)
    if cellvit_embeddings is not None:
        np.save(os.path.join(cache_dir, "cellvit_embeddings.npy"), cellvit_embeddings)
    if cell_dino_features is not None:
        np.save(os.path.join(cache_dir, "cell_dino_features.npy"), cell_dino_features)
    if bboxes is not None:
        np.save(os.path.join(cache_dir, "bboxes.npy"), bboxes)

    full_manifest = dict(manifest)
    full_manifest.update({
        "schema_version": SCHEMA_VERSION,
        "num_patches": len(sample_ids_np),
        "num_cells": int(offsets[-1]),
        "sample_fingerprint": sample_order_fingerprint(sample_ids_np.tolist()),
        "has_cellvit_embeddings": cellvit_embeddings is not None,
        "has_cell_dino_features": cell_dino_features is not None,
    })
    with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(full_manifest, f, indent=2, sort_keys=True)


def load_cellvit_cache(
    cache_dir: str,
    *,
    expected_sample_ids: Optional[Sequence[str]] = None,
    mmap_mode: Optional[str] = "r",
) -> CellViTCache:
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Nucleus cache manifest not found: {manifest_path}. "
            "Run scripts/extract_cellvit_features.py first."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported nucleus cache schema: {manifest.get('schema_version')}"
        )

    def optional_load(name: str):
        path = os.path.join(cache_dir, name)
        return np.load(path, mmap_mode=mmap_mode) if os.path.exists(path) else None

    def required_load(name: str):
        value = optional_load(name)
        if value is None:
            raise FileNotFoundError(
                f"Nucleus cache manifest declares {name}, but the file is missing"
            )
        return value

    offsets = np.load(os.path.join(cache_dir, "offsets.npy"), mmap_mode=mmap_mode)
    confidence = np.load(
        os.path.join(cache_dir, "confidence.npy"), mmap_mode=mmap_mode
    )
    sample_ids = np.load(
        os.path.join(cache_dir, "sample_ids.npy"), mmap_mode=mmap_mode,
        allow_pickle=False,
    )
    cellvit_embeddings = (
        required_load("cellvit_embeddings.npy")
        if manifest.get("has_cellvit_embeddings") else None
    )
    cell_dino_features = (
        required_load("cell_dino_features.npy")
        if manifest.get("has_cell_dino_features") else None
    )
    bboxes = optional_load("bboxes.npy")

    _validate_arrays(
        offsets, confidence, cellvit_embeddings, cell_dino_features,
        bboxes, sample_ids,
    )
    actual_fingerprint = sample_order_fingerprint(sample_ids.tolist())
    if manifest.get("sample_fingerprint") != actual_fingerprint:
        raise ValueError("Nucleus cache sample_ids do not match its manifest")
    if expected_sample_ids is not None:
        expected_fingerprint = sample_order_fingerprint(expected_sample_ids)
        if expected_fingerprint != actual_fingerprint:
            raise ValueError(
                "Nucleus cache sample order does not match the current train split"
            )
        if sample_ids.tolist() != list(expected_sample_ids):
            raise ValueError("Nucleus cache sample IDs do not exactly match the split")

    return CellViTCache(
        offsets=offsets,
        cellvit_embeddings=cellvit_embeddings,
        cell_dino_features=cell_dino_features,
        confidence=confidence,
        bboxes=bboxes,
        sample_ids=sample_ids,
        manifest=manifest,
    )
