"""Probe checkpoint IO.

Only the linear layer is stored, not the module object, so a checkpoint stays
loadable when the surrounding code changes and `torch.load` can run with
`weights_only=True`.
"""

import os
from typing import Any, Dict, Optional

import torch

from .probe import LinearProbe


def save_probe(
    probe: LinearProbe,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Store the linear layer, plus optional metadata describing the run.

    `metadata` makes a checkpoint self-describing: which run, budget, seed and
    dataset produced it. Without it, a directory of `*_probe_budget_*.pt` files
    can only be interpreted through their filenames, which is how a checkpoint
    gets attributed to the wrong run after a rename.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {
        "feat_dim": probe.fc.in_features,
        "num_classes": probe.fc.out_features,
        "weight": probe.fc.weight.data.cpu(),
        "bias": probe.fc.bias.data.cpu(),
    }
    if metadata:
        # Kept under one key so `weights_only=True` loading stays possible for
        # the tensors: callers that want the metadata opt into a full load.
        payload["metadata"] = metadata
    torch.save(payload, path)


def load_probe(path: str, device: torch.device) -> LinearProbe:
    # `weights_only=False` because a checkpoint may carry a metadata dict. The
    # files are produced by this project's own runs.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    probe = LinearProbe(checkpoint["feat_dim"], checkpoint["num_classes"])
    probe.fc.weight.data.copy_(checkpoint["weight"].to(device))
    probe.fc.bias.data.copy_(checkpoint["bias"].to(device))
    return probe.to(device).eval()
