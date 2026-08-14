"""Feature transforms for the nucleus/final-learner alignment diagnostic."""

from __future__ import annotations

import math

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def standardize_l2(
    features: np.ndarray,
    valid: np.ndarray | None = None,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Column-standardize on valid rows, then L2-normalize each row."""
    values = np.asarray(features)
    fit_values = values if valid is None else values[valid]
    if len(fit_values) == 0:
        raise ValueError("Cannot normalize a feature block with no valid rows")
    mean = np.asarray(fit_values.mean(axis=0, dtype=np.float64), dtype=np.float32)
    std = np.asarray(fit_values.std(axis=0, dtype=np.float64), dtype=np.float32)
    std[std < 1e-6] = 1.0
    output = np.empty(values.shape, dtype=np.float32)
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        block = (np.asarray(values[start:stop], dtype=np.float32) - mean) / std
        norm = np.linalg.norm(block, axis=1, keepdims=True)
        output[start:stop] = block / np.maximum(norm, 1e-12)
    if valid is not None:
        output[~valid] = 0.0
    return output


def fit_pca_block(
    features: np.ndarray,
    valid: np.ndarray,
    n_components: int,
    fit_samples: int,
    seed: int,
) -> np.ndarray:
    """Fit an unlabeled PCA summary on a bounded valid-row subsample."""
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 2:
        raise ValueError("Need at least two valid nucleus patches for PCA")
    rng = np.random.default_rng(seed)
    if len(valid_indices) > fit_samples:
        fit_indices = rng.choice(valid_indices, size=fit_samples, replace=False)
    else:
        fit_indices = valid_indices
    scaled = standardize_l2(features, valid=valid)
    components = min(n_components, scaled.shape[1], len(fit_indices) - 1)
    if components < 1:
        raise ValueError("PCA dimension resolved to zero")
    pca = PCA(
        n_components=components, svd_solver="randomized", random_state=seed,
    )
    pca.fit(scaled[fit_indices])
    transformed = np.asarray(pca.transform(scaled), dtype=np.float32)
    transformed[~valid] = 0.0
    return standardize_l2(transformed, valid=valid)


def residualize_cell_block(
    dino_features: np.ndarray,
    cell_features: np.ndarray,
    valid: np.ndarray,
    fit_samples: int,
    ridge_alpha: float,
    seed: int,
) -> np.ndarray:
    """Remove the cell component predicted by a low-rank DINO ridge model."""
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 2:
        raise ValueError("Need at least two valid nucleus patches for residualization")
    rng = np.random.default_rng(seed + 17)
    if len(valid_indices) > fit_samples:
        fit_indices = rng.choice(valid_indices, size=fit_samples, replace=False)
    else:
        fit_indices = valid_indices
    dino_components = min(cell_features.shape[1], 64, len(fit_indices) - 1)
    dino_pca = PCA(
        n_components=dino_components,
        svd_solver="randomized",
        random_state=seed + 1,
    )
    dino_pca.fit(np.asarray(dino_features[fit_indices], dtype=np.float32))
    dino_low = np.asarray(dino_pca.transform(dino_features), dtype=np.float32)
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True, solver="lsqr")
    ridge.fit(dino_low[fit_indices], cell_features[fit_indices])
    residual = cell_features - np.asarray(ridge.predict(dino_low), dtype=np.float32)
    residual[~valid] = 0.0
    return standardize_l2(residual, valid=valid)


def concat_blocks(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Concatenate two unit-normalized views with equal total block weight."""
    if len(left) != len(right):
        raise ValueError("Feature blocks must have the same number of rows")
    scale = np.float32(1.0 / math.sqrt(2.0))
    return np.concatenate((left * scale, right * scale), axis=1).astype(
        np.float32, copy=False
    )


def normalized_auc(budgets, values) -> float:
    """Trapezoidal learning-curve area divided by the budget interval."""
    x = np.asarray(budgets, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if len(x) < 2 or x[-1] <= x[0] or np.any(np.diff(x) <= 0):
        raise ValueError("nAUC needs at least two strictly increasing budgets")
    if y.shape != x.shape:
        raise ValueError("budgets and values must have the same shape")
    return float(np.trapezoid(y, x=x) / (x[-1] - x[0]))
