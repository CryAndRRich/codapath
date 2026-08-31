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

# Every training loop in this project -- the selection-round probes, the dual
# probe, and the final-training pass -- runs the same budget: up to
# `probe_epochs` epochs, stopping early once the epoch's mean TRAINING loss
# has not improved for `EARLY_STOP_PATIENCE` consecutive epochs.
EARLY_STOP_PATIENCE = 20

# An "improvement" smaller than this is noise, not progress. Without a
# threshold, a loss drifting by 1e-9 per epoch resets the patience counter
# forever and early stopping never fires.
EARLY_STOP_MIN_DELTA = 1e-5


class _EarlyStopper:
    """Patience counter over per-epoch mean TRAINING loss.

    Training loss, not validation loss, and deliberately so: at this
    project's budgets a held-out split is both tiny and expensive. Budget 25
    over 9 classes leaves under one validation sample per class after a 70/30
    split, so the signal would be mostly noise -- and, worse, holding out 30%
    removes 7 of 25 labeled points from training, making low-budget accuracy
    worse for a reason that has nothing to do with which sampler chose them.

    So this is a COMPUTE saver, not a regulariser: it stops once the fit has
    converged, and cannot detect overfitting (training loss keeps falling
    when a probe overfits). `patience=20` out of 100 epochs is deliberately
    loose -- it should end runs that plateaued, not truncate ones still
    learning.

    **Measured behaviour: on this project's actual budgets it rarely fires.**
    At `lr=1e-3` with <=200 labeled points the probe's mean training loss is
    still falling steeply at epoch 100 (2.92 -> 0.25, still improving ~6e-3
    per epoch on a 9-class synthetic fit), so patience never accumulates and
    all 100 epochs run. Verified at budgets 25/50/100/200: identical data and
    identical init, early-stopped vs forced-full-100, give the SAME accuracy
    to three decimals. Treat this as a guard that costs nothing and would cut
    a genuinely plateaued run short -- not as a speed-up to budget for. If a
    future change (a larger lr, a much bigger labeled set) does make these
    fits converge early, this starts paying off without any further edit.
    """

    def __init__(self, patience: int = EARLY_STOP_PATIENCE,
                 min_delta: float = EARLY_STOP_MIN_DELTA) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0
        self.stopped_epoch: Optional[int] = None

    def step(self, loss: float, epoch: int) -> bool:
        """Record one epoch's mean loss; return True when training should stop."""
        if loss < self.best - self.min_delta:
            self.best = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False


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
    stopper = _EarlyStopper()
    for epoch in range(num_epochs):
        running, batches = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            batches += 1
        if batches and stopper.step(running / batches, epoch):
            break
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
    # Full-batch here (no DataLoader), so one step IS one epoch and the loss
    # fed to the stopper is already the epoch mean.
    stopper = _EarlyStopper()
    for epoch in range(num_epochs):
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
        if stopper.step(float(loss.detach()), epoch):
            break
    return visual_probe, cell_probe
