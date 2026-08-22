"""Linear probes on frozen features.

`train_probe` is the shared evaluation protocol: every sampler is scored by
the same single-layer probe on the same frozen backbone, so a difference in
accuracy can only come from which samples were selected.

`train_dual_probe` fits the visual and cell probes together. It exists because
`scalpel` needs both heads in the same round to measure how much they disagree,
and because an optional consistency term couples them during training.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


CONSISTENCY_MODES = ("symmetric_js", "visual_teacher", "cell_teacher")


class LinearProbe(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @torch.inference_mode()
    def predict_logits(
        self,
        features: np.ndarray,
        device: torch.device,
        batch_size: int = 8192,
    ) -> np.ndarray:
        self.eval()
        outputs = []
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                features[start:start + batch_size], device=device, dtype=torch.float32
            )
            outputs.append(self(batch).cpu().numpy().astype(np.float32))
        if not outputs:
            return np.empty((0, self.fc.out_features), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    @torch.inference_mode()
    def predict_proba(
        self,
        features: np.ndarray,
        device: torch.device,
        batch_size: int = 8192,
    ) -> np.ndarray:
        logits = self.predict_logits(features, device, batch_size=batch_size)
        if len(logits) == 0:
            return logits
        return F.softmax(torch.from_numpy(logits), dim=1).numpy().astype(np.float32)


def class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency weights; classes absent from `labels` get zero.

    The division is masked rather than computed everywhere and filtered after,
    which is what `np.where` would do -- an absent class has count 0 and would
    warn about dividing by zero on every call.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    np.divide(counts.sum(), num_classes * counts, out=weights, where=present)
    return weights


def train_probe(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    num_epochs: int,
    lr: float,
    device: torch.device,
    weight_decay: float = 0.0,
) -> LinearProbe:
    if len(labels) == 0:
        raise ValueError("Cannot train a linear probe with zero labeled samples")
    probe = LinearProbe(features.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights(labels, num_classes), device=device)
    )
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=min(64, len(labels)), shuffle=True)

    probe.train()
    for _ in range(num_epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return probe


def _consistency_penalty(
    visual_probs: torch.Tensor,
    cell_probs: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "symmetric_js":
        middle = 0.5 * (visual_probs + cell_probs)
        return 0.5 * (
            (visual_probs * (visual_probs.log() - middle.log())).sum(dim=1)
            + (cell_probs * (cell_probs.log() - middle.log())).sum(dim=1)
        )
    if mode == "visual_teacher":
        teacher = visual_probs.detach()
        return (teacher * (teacher.log() - cell_probs.log())).sum(dim=1)
    teacher = cell_probs.detach()
    return (teacher * (teacher.log() - visual_probs.log())).sum(dim=1)


def train_dual_probe(
    visual_features: np.ndarray,
    cell_features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    num_epochs: int,
    lr: float,
    device: torch.device,
    cell_valid: np.ndarray,
    cell_reliability: Optional[np.ndarray] = None,
    consistency_weight: float = 0.0,
    consistency_mode: str = "symmetric_js",
    weight_decay: float = 0.0,
) -> Tuple[LinearProbe, LinearProbe]:
    """Fit the visual and cell probes jointly, optionally coupling them.

    The visual cross-entropy uses every labeled patch. The cell cross-entropy
    and the consistency term use only patches that actually contain a detected
    nucleus, weighted by that patch's reliability. Updates are full-batch on
    purpose: active-learning budgets here are a few hundred points, and keeping
    two differently-sized labeled views aligned is simpler than pairing them
    inside mini-batches.

    `consistency_weight=0` is the baseline and the default. A positive value is
    an ablation that works AGAINST acquisition: forcing the two heads to agree
    shrinks exactly the divergence `scalpel` selects on. Raise it only when
    measuring that trade-off deliberately.
    """
    visual_features = np.asarray(visual_features, dtype=np.float32)
    cell_features = np.asarray(cell_features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    cell_valid = np.asarray(cell_valid, dtype=bool)

    if visual_features.ndim != 2 or cell_features.ndim != 2:
        raise ValueError("visual_features and cell_features must be 2-D")
    if not len(visual_features) == len(cell_features) == len(labels) == len(cell_valid):
        raise ValueError("Dual-probe arrays must align by labeled patch")
    if len(labels) == 0:
        raise ValueError("Cannot train dual probes with zero labeled samples")
    if not cell_valid.any():
        raise ValueError("Cannot train a cell probe without a valid cell view")
    if consistency_weight < 0.0:
        raise ValueError("consistency_weight must be non-negative")
    if consistency_mode not in CONSISTENCY_MODES:
        raise ValueError(f"consistency_mode must be one of {list(CONSISTENCY_MODES)}")

    if cell_reliability is None:
        reliability = np.ones(len(labels), dtype=np.float32)
    else:
        reliability = np.clip(np.asarray(cell_reliability, dtype=np.float32), 0.0, 1.0)
        if len(reliability) != len(labels):
            raise ValueError("cell_reliability must align by labeled patch")

    visual_probe = LinearProbe(visual_features.shape[1], num_classes).to(device)
    cell_probe = LinearProbe(cell_features.shape[1], num_classes).to(device)
    visual_criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights(labels, num_classes), device=device)
    )
    cell_criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights(labels[cell_valid], num_classes), device=device)
    )
    optimizer = optim.Adam(
        list(visual_probe.parameters()) + list(cell_probe.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    visual_x = torch.as_tensor(visual_features, dtype=torch.float32, device=device)
    cell_x = torch.as_tensor(cell_features[cell_valid], dtype=torch.float32, device=device)
    visual_y = torch.as_tensor(labels, dtype=torch.long, device=device)
    cell_y = torch.as_tensor(labels[cell_valid], dtype=torch.long, device=device)
    valid_mask = torch.as_tensor(cell_valid, dtype=torch.bool, device=device)
    rho = torch.as_tensor(reliability[cell_valid], dtype=torch.float32, device=device)

    visual_probe.train()
    cell_probe.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        visual_logits = visual_probe(visual_x)
        cell_logits = cell_probe(cell_x)
        loss = (
            visual_criterion(visual_logits, visual_y)
            + cell_criterion(cell_logits, cell_y)
        )
        if consistency_weight > 0.0:
            visual_probs = F.softmax(visual_logits[valid_mask], dim=1).clamp_min(1e-8)
            cell_probs = F.softmax(cell_logits, dim=1).clamp_min(1e-8)
            penalty = _consistency_penalty(visual_probs, cell_probs, consistency_mode)
            loss = loss + consistency_weight * (rho * penalty).sum() / rho.sum().clamp_min(1e-8)
        loss.backward()
        optimizer.step()
    return visual_probe, cell_probe
