"""Pooling helpers for a variable number of cells per tissue patch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class NucleusView:
    patch_features: np.ndarray
    reliability: np.ndarray
    cell_counts: np.ndarray
    metadata: Optional[Dict[str, object]] = None

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
    return NucleusView(
        pooled, reliability, counts, metadata={"pooling": "mean"}
    )


def pool_ragged_rff(
    cell_features: np.ndarray,
    offsets: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    reliability_mode: str = "valid",
    *,
    output_dim: int = 64,
    bandwidth: Optional[float] = None,
    bandwidth_sample_size: int = 2048,
    transform_batch_size: int = 32768,
    seed: int = 42,
) -> NucleusView:
    """Approximate a Gaussian kernel mean embedding for every cell bag.

    Each CellViT token is L2-normalized and mapped through random Fourier
    features ``sqrt(2/D) cos(Wx+b)``.  Confidence-weighted averaging in this
    feature space approximates the mean embedding of the patch's empirical
    cell distribution.  Unlike averaging the raw tokens, this representation
    can distinguish bags that have the same first moment but different modes.

    The Gaussian bandwidth uses a deterministic sampled-pair median heuristic
    unless supplied explicitly.  Transformation is chunked so the ragged cell
    cache remains memory-mapped rather than being promoted to one giant
    float32 array.
    """
    features = np.asarray(cell_features)
    offsets = np.asarray(offsets, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError("cell_features must have shape (num_cells, feature_dim)")
    if offsets.ndim != 1 or len(offsets) == 0:
        raise ValueError("offsets must be a non-empty one-dimensional array")
    if offsets[0] != 0 or offsets[-1] != len(features):
        raise ValueError("Invalid ragged offsets")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("offsets must be non-decreasing")
    if output_dim <= 0 or bandwidth_sample_size < 2 or transform_batch_size <= 0:
        raise ValueError("RFF dimensions and batch sizes must be positive")

    num_cells, feature_dim = features.shape
    num_patches = len(offsets) - 1
    counts = np.diff(offsets).astype(np.int64)
    if confidence is None:
        weights = np.ones(num_cells, dtype=np.float32)
    else:
        weights = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
        if len(weights) != num_cells:
            raise ValueError("confidence length must match cell_features")

    mean_confidence = np.zeros(num_patches, dtype=np.float32)
    nonempty = counts > 0
    if num_cells:
        starts = offsets[:-1][nonempty]
        sums = np.add.reduceat(weights, starts)
        mean_confidence[nonempty] = sums / counts[nonempty]

    if reliability_mode == "valid":
        reliability = nonempty.astype(np.float32)
    elif reliability_mode == "mean_confidence":
        reliability = mean_confidence
    else:
        raise ValueError("reliability_mode must be 'valid' or 'mean_confidence'")

    pooled = np.zeros((num_patches, output_dim), dtype=np.float32)
    if num_cells == 0:
        effective_bandwidth = float(bandwidth or 1.0)
        return NucleusView(
            pooled,
            reliability,
            counts,
            metadata={
                "pooling": "rff",
                "rff_dim": float(output_dim),
                "rff_bandwidth": effective_bandwidth,
                "rff_seed": float(seed),
            },
        )

    rng = np.random.default_rng(seed)
    sample_size = min(int(bandwidth_sample_size), num_cells)
    sample_indices = rng.choice(num_cells, sample_size, replace=False)
    sample = np.asarray(features[sample_indices], dtype=np.float32)
    sample /= np.clip(np.linalg.norm(sample, axis=1, keepdims=True), 1e-12, None)
    if bandwidth is None:
        num_pairs = min(8192, max(256, sample_size * 4))
        left = rng.integers(0, sample_size, size=num_pairs)
        right = rng.integers(0, sample_size, size=num_pairs)
        keep = left != right
        pair_dist = np.linalg.norm(sample[left[keep]] - sample[right[keep]], axis=1)
        positive = pair_dist[pair_dist > 1e-6]
        effective_bandwidth = (
            float(np.median(positive)) if len(positive) else 1.0
        )
    else:
        effective_bandwidth = float(bandwidth)
        if effective_bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive")

    projection = rng.normal(
        0.0, 1.0 / effective_bandwidth, size=(feature_dim, output_dim)
    ).astype(np.float32)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=output_dim).astype(np.float32)
    scale = np.float32(np.sqrt(2.0 / output_dim))

    # All-zero confidences within a non-empty patch mean "unknown confidence",
    # not "discard every cell".  Match confidence-weighted mean pooling by
    # falling back to uniform weights for exactly those patches.
    effective_weights = weights.copy()
    patch_weight_sums = np.zeros(num_patches, dtype=np.float32)
    np.add.at(
        patch_weight_sums,
        np.repeat(np.arange(num_patches, dtype=np.int64), counts),
        effective_weights,
    )
    for patch_idx in np.flatnonzero(nonempty & (patch_weight_sums <= 1e-12)):
        effective_weights[offsets[patch_idx]:offsets[patch_idx + 1]] = 1.0

    patch_ids = np.repeat(np.arange(num_patches, dtype=np.int64), counts)
    weight_sums = np.zeros(num_patches, dtype=np.float32)
    np.add.at(weight_sums, patch_ids, effective_weights)
    for start in range(0, num_cells, transform_batch_size):
        end = min(start + transform_batch_size, num_cells)
        chunk = np.asarray(features[start:end], dtype=np.float32)
        chunk /= np.clip(np.linalg.norm(chunk, axis=1, keepdims=True), 1e-12, None)
        mapped = scale * np.cos(chunk @ projection + phase)
        mapped *= effective_weights[start:end, None]
        np.add.at(pooled, patch_ids[start:end], mapped)
    pooled[nonempty] /= weight_sums[nonempty, None]

    return NucleusView(
        pooled,
        reliability,
        counts,
        metadata={
            "pooling": "rff",
            "rff_dim": float(output_dim),
            "rff_bandwidth": effective_bandwidth,
            "rff_seed": float(seed),
        },
    )


def pool_ragged_moments(
    cell_features: np.ndarray,
    offsets: np.ndarray,
    confidence: Optional[np.ndarray] = None,
) -> NucleusView:
    """Summarize each cell set by weighted first/second moments and QC.

    The output for a ``d``-dimensional cell embedding has ``2*d + 2``
    columns: weighted mean, weighted standard deviation, ``log1p`` cell
    count, and mean detection confidence. Empty patches are exactly zero and
    have zero reliability. This is intentionally a cheap diagnostic between
    the current mean pooling and a future distribution/set encoder.
    """
    features = np.asarray(cell_features)
    offsets = np.asarray(offsets, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError("cell_features must have shape (num_cells, feature_dim)")
    if offsets.ndim != 1 or len(offsets) == 0:
        raise ValueError("offsets must be a non-empty one-dimensional array")
    if offsets[0] != 0 or offsets[-1] != len(features):
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
    feature_dim = features.shape[1]
    pooled = np.zeros((num_patches, 2 * feature_dim + 2), dtype=np.float32)
    counts = np.diff(offsets).astype(np.int64)
    for patch_idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if start == end:
            continue
        patch_features = np.asarray(features[start:end], dtype=np.float32)
        patch_weights = weights[start:end]
        weight_sum = float(patch_weights.sum())
        if weight_sum <= 1e-12:
            patch_weights = np.ones(end - start, dtype=np.float32)
            weight_sum = float(end - start)
        normalized_weights = patch_weights / weight_sum
        mean = (patch_features * normalized_weights[:, None]).sum(axis=0)
        variance = (
            (patch_features - mean) ** 2 * normalized_weights[:, None]
        ).sum(axis=0)
        pooled[patch_idx, :feature_dim] = mean
        pooled[patch_idx, feature_dim:2 * feature_dim] = np.sqrt(
            np.maximum(variance, 0.0)
        )
        pooled[patch_idx, -2] = np.log1p(end - start)
        pooled[patch_idx, -1] = float(weights[start:end].mean())

    reliability = (counts > 0).astype(np.float32)
    return NucleusView(
        pooled, reliability, counts, metadata={"pooling": "moments"}
    )
