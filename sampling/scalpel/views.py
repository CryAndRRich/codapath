"""The two feature views SCALPEL selects on.

Both are returned L2-normalized, because `sampling.kernels` reads a dot product
AS a cosine and a non-unit row silently flattens the coverage kernel.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

MISSING_IMPUTE_MODES = ("mean", "zero")


def normalize_rows(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return F.normalize(
        torch.as_tensor(values, device=device, dtype=torch.float32), p=2, dim=1
    )


def build_cell_view(
    cell_features: np.ndarray,
    reliability: np.ndarray,
    missing_impute: str,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """Return `(normalized cell view, fraction of patches with no nucleus)`.

    A patch where CellViT found nothing has no cell vector at all, and the
    choice of what to put there changes selection materially:

    * `"mean"` (default) — the mean direction of all valid cell vectors. Such a
      patch becomes maximally typical rather than maximally novel, so it stops
      winning coverage on the strength of being unlike everything.
    * `"zero"` — leave the row at zero. Ablation only: a zero row has cosine 0
      with every other row INCLUDING other zero rows, so covering one does not
      cover the next and the greedy over-selects them. Always read the reported
      `missing=` fraction before trusting a `"zero"` run.
    """
    if missing_impute not in MISSING_IMPUTE_MODES:
        raise ValueError(f"missing_impute must be one of {list(MISSING_IMPUTE_MODES)}")

    reliability_t = torch.as_tensor(reliability, device=device, dtype=torch.float32)
    valid = reliability_t > 0.0
    missing = ~valid
    missing_fraction = float(missing.float().mean().item())

    raw = torch.as_tensor(cell_features, device=device, dtype=torch.float32)
    view = torch.zeros_like(raw)
    if valid.any():
        view[valid] = F.normalize(raw[valid], p=2, dim=1)
        if missing_impute == "mean" and missing.any():
            mean_direction = view[valid].mean(dim=0, keepdim=True)
            mean_norm = float(mean_direction.norm(p=2).item())
            if mean_norm > 1e-8:
                view[missing] = mean_direction / mean_norm
            # A ~zero mean direction means the valid vectors cancel out, so
            # there is nothing meaningful to impute; leave those rows at zero.
    return view, missing_fraction
