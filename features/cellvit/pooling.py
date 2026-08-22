"""Pooling helpers for a variable number of cells per tissue patch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class NucleusView:
    patch_features: np.ndarray
    reliability: np.ndarray
    cell_counts: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return self.cell_counts > 0


def pool_ragged_features(
    cell_features: np.ndarray,
    offsets: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    reliability_mode: str = "valid",
) -> NucleusView:
    """Confidence-weighted cell mean with an explicit empty-patch mask.

    Pooling features before a shared linear head is exactly equivalent to
    pooling that head's per-cell logits, while training each patch only once.
    """
    # Keep a float16/memmap cache lazy; only each patch slice is promoted to
    # float32. Casting the full ragged array here can require many gigabytes.
    features = np.asarray(cell_features)
    offsets = np.asarray(offsets, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError("cell_features must have shape (num_cells, feature_dim)")
    if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(features):
        raise ValueError("Invalid ragged offsets")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("offsets must be non-decreasing")

    if confidence is None:
        weights = np.ones(len(features), dtype=np.float32)
    else:
        weights = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
        if len(weights) != len(features):
            raise ValueError("confidence length must match cell_features")

    num_patches = len(offsets) - 1
    pooled = np.zeros((num_patches, features.shape[1]), dtype=np.float32)
    counts = np.diff(offsets).astype(np.int64)
    mean_confidence = np.zeros(num_patches, dtype=np.float32)
    for patch_idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if start == end:
            continue
        patch_weights = weights[start:end]
        weight_sum = float(patch_weights.sum())
        if weight_sum <= 1e-12:
            patch_weights = np.ones(end - start, dtype=np.float32)
            weight_sum = float(end - start)
        patch_features = np.asarray(features[start:end], dtype=np.float32)
        pooled[patch_idx] = (
            patch_features * patch_weights[:, None]
        ).sum(axis=0) / weight_sum
        mean_confidence[patch_idx] = float(weights[start:end].mean())

    if reliability_mode == "valid":
        reliability = (counts > 0).astype(np.float32)
    elif reliability_mode == "mean_confidence":
        reliability = mean_confidence
    else:
        raise ValueError("reliability_mode must be 'valid' or 'mean_confidence'")
    return NucleusView(pooled, reliability, counts)
