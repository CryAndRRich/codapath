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


def _kde_weights(patch_features: np.ndarray, bandwidth_scale: float) -> np.ndarray:
    """Per-cell kernel-density weight inside ONE patch.

    `rho_i = sum_j exp(-||x_i - x_j||^2 / (2 h^2))`, `h = bandwidth_scale *
    median pairwise distance` within the patch. Cells sitting in the patch's
    dominant morphological mode get a high `rho`; lone outliers (a mis-segmented
    fragment, a single odd nucleus) get a low one.

    This is the difference from plain averaging that motivates KDE here: the
    mean gives every detected nucleus equal say, so one segmentation artefact
    shifts the patch descriptor as much as a real, populous cell type. The KDE
    weight makes the descriptor track the MODE of the patch's cell population
    instead of its centroid.
    """
    n = patch_features.shape[0]
    if n == 1:
        return np.ones(1, dtype=np.float32)
    sq = (patch_features ** 2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (patch_features @ patch_features.T), 0.0)
    iu = np.triu_indices(n, k=1)
    med = float(np.median(np.sqrt(d2[iu])))
    if not np.isfinite(med) or med <= 1e-8:
        # Every cell in this patch is (numerically) identical: KDE cannot
        # discriminate, and any bandwidth would divide by ~0. Uniform weights
        # here degrade gracefully to exactly the mean-pooling result.
        return np.ones(n, dtype=np.float32)
    h = bandwidth_scale * med
    return np.exp(-d2 / (2.0 * h * h)).sum(axis=1).astype(np.float32)


def pool_ragged_features(
    cell_features: np.ndarray,
    offsets: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    reliability_mode: str = "valid",
    pool_mode: str = "mean",
    kde_bandwidth_scale: float = 1.0,
) -> NucleusView:
    """Confidence-weighted cell pooling with an explicit empty-patch mask.

    Pooling features before a shared linear head is exactly equivalent to
    pooling that head's per-cell logits, while training each patch only once.

    `pool_mode`
        "mean" (default, unchanged behaviour) — confidence-weighted average.
        "kde"  — confidence weights are multiplied by the per-cell kernel
                 density `_kde_weights` before averaging, so the patch vector
                 follows the dominant cell mode rather than the plain centroid.
                 With one cell per patch, or with numerically identical cells,
                 this reduces exactly to "mean".
    """
    if pool_mode not in ("mean", "kde"):
        raise ValueError(f"pool_mode must be 'mean' or 'kde', got {pool_mode!r}")
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
        if pool_mode == "kde":
            patch_weights = patch_weights * _kde_weights(patch_features, kde_bandwidth_scale)
            weight_sum = float(patch_weights.sum())
            if weight_sum <= 1e-12:
                patch_weights = np.ones(end - start, dtype=np.float32)
                weight_sum = float(end - start)
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