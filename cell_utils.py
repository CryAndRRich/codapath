"""Utilities for loading and aggregating per-cell CellViT embeddings
into per-image vectors (mean pooling or KDE-weighted pooling)."""

import os

import numpy as np


def load_and_aggregate_cell_embeddings(
    cell_dir: str,
    aggregation: str = "mean",
) -> np.ndarray:
    """Load per-cell CellViT embeddings and aggregate to per-image vectors.

    Args:
        cell_dir: Directory containing cellvit_embeddings.npy and
                  offsets.npy or sample_ids.npy for grouping.
        aggregation: "mean" for simple mean pooling,
                     "kde" for inverse-density-weighted pooling.

    Returns:
        np.ndarray of shape (num_images, embed_dim).
    """
    embeddings = np.load(os.path.join(cell_dir, "cellvit_embeddings.npy"))
    embed_dim = embeddings.shape[1]

    # Determine how cells map to images
    offsets_path = os.path.join(cell_dir, "offsets.npy")
    sample_ids_path = os.path.join(cell_dir, "sample_ids.npy")

    if os.path.exists(offsets_path):
        offsets = np.load(offsets_path)
        num_images = len(offsets) - 1
        groups = [(int(offsets[i]), int(offsets[i + 1])) for i in range(num_images)]
    elif os.path.exists(sample_ids_path):
        sample_ids = np.load(sample_ids_path)
        unique_ids = np.unique(sample_ids)
        num_images = len(unique_ids)
        groups = []
        for sid in unique_ids:
            indices = np.where(sample_ids == sid)[0]
            groups.append((int(indices[0]), int(indices[-1]) + 1))
    else:
        raise FileNotFoundError(
            f"Need offsets.npy or sample_ids.npy in {cell_dir}"
        )

    result = np.zeros((num_images, embed_dim), dtype=np.float32)

    for i, (start, end) in enumerate(groups):
        cells = embeddings[start:end]
        if len(cells) == 0:
            continue
        if len(cells) == 1:
            result[i] = cells[0]
            continue

        if aggregation == "mean":
            result[i] = cells.mean(axis=0)
        elif aggregation == "kde":
            result[i] = _kde_pool(cells)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    print(f"[CellUtils] Loaded {embeddings.shape[0]} cells -> "
          f"{num_images} images ({aggregation} pooling), dim={embed_dim}")
    return result


def _kde_pool(cells: np.ndarray) -> np.ndarray:
    """Inverse-density-weighted pooling: cells in sparse regions of the
    embedding space contribute more, encouraging diversity."""
    # Cosine distance matrix
    norms = np.linalg.norm(cells, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalised = cells / norms
    sim = normalised @ normalised.T
    dist_sq = np.maximum(2.0 - 2.0 * sim, 0.0)

    # Bandwidth: median heuristic
    upper = dist_sq[np.triu_indices_from(dist_sq, k=1)]
    bandwidth_sq = max(float(np.median(upper)), 1e-6)

    # Density estimate at each cell
    densities = np.exp(-dist_sq / (2.0 * bandwidth_sq)).sum(axis=1)

    # Inverse-density weights (normalised)
    weights = 1.0 / (densities + 1e-8)
    weights = weights / weights.sum()

    return (weights[:, None] * cells).sum(axis=0)
