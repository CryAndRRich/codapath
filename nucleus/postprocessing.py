"""Compatibility helpers around CellViT's official post-processor."""

from __future__ import annotations

import numpy as np


def stack_prediction_maps_numpy(
    nuclei_type_map: np.ndarray,
    nuclei_binary_map: np.ndarray,
    hv_map: np.ndarray,
) -> np.ndarray:
    """Reproduce CellViT ``stack_pred_maps`` without its fragile Numba JIT.

    CellViT 1.0.9 JIT-compiles a mixed-dtype ``np.stack``. Current Kaggle
    Python/Numba fails during type inference because ``argmax`` produces int64
    while the HV maps are floating-point. The post-processing algorithm only
    needs a four-channel numeric array, so an explicit float32 buffer preserves
    the exact values and avoids both JIT compilation and float64 promotion.
    """
    nuclei_type_map = np.asarray(nuclei_type_map)
    nuclei_binary_map = np.asarray(nuclei_binary_map)
    hv_map = np.asarray(hv_map)
    if not (
        nuclei_type_map.ndim == nuclei_binary_map.ndim == hv_map.ndim == 4
    ):
        raise ValueError("CellViT prediction maps must all be 4-D")
    if not (
        nuclei_type_map.shape[:3]
        == nuclei_binary_map.shape[:3]
        == hv_map.shape[:3]
    ):
        raise ValueError("CellViT prediction maps must align on B/H/W")
    if nuclei_binary_map.shape[-1] != 2 or hv_map.shape[-1] != 2:
        raise ValueError("Binary and HV prediction maps must have two channels")

    output = np.empty((*hv_map.shape[:3], 4), dtype=np.float32)
    output[..., 0] = np.argmax(nuclei_type_map, axis=-1)
    output[..., 1] = np.argmax(nuclei_binary_map, axis=-1)
    output[..., 2:] = hv_map.astype(np.float32, copy=False)
    return output
