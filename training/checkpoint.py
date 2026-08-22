"""Probe checkpoint IO.

Only the linear layer is stored, not the module object, so a checkpoint stays
loadable when the surrounding code changes and `torch.load` can run with
`weights_only=True`.
"""

import os

import torch

from .probe import LinearProbe


def save_probe(probe: LinearProbe, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "feat_dim": probe.fc.in_features,
            "num_classes": probe.fc.out_features,
            "weight": probe.fc.weight.data.cpu(),
            "bias": probe.fc.bias.data.cpu(),
        },
        path,
    )


def load_probe(path: str, device: torch.device) -> LinearProbe:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    probe = LinearProbe(checkpoint["feat_dim"], checkpoint["num_classes"])
    probe.fc.weight.data.copy_(checkpoint["weight"].to(device))
    probe.fc.bias.data.copy_(checkpoint["bias"].to(device))
    return probe.to(device).eval()
