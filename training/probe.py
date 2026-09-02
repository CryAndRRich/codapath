"""Linear probes on frozen features.

`train_probe` is the shared evaluation protocol: every sampler is scored by
the same single-layer probe on the same frozen backbone, so a difference in
accuracy can only come from which samples were selected.

`train_dual_probe` fits the visual and cell probes together. It exists because
`scalpel` needs both heads in the same round to measure how much they disagree,
and because an optional consistency term couples them during training.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Upper bound on how many unlabeled pool rows the semi-supervised term uses
# per epoch. See `train_dual_probe` for why it is half the pool, capped.
POOL_SUBSAMPLE_CAP = 20000

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
) -> torch.Tensor:
    """Per-row symmetric Jensen-Shannon divergence between the two heads.

    Symmetric on purpose, and the only shape offered: the two teacher/student
    variants that used to live here (`visual_teacher`, `cell_teacher`) made
    one view authoritative, which is the wrong prior for this method -- the
    whole premise of the cell view is that it sees something the visual one
    does not, so neither should be the target the other is pulled toward.
    """
    middle = 0.5 * (visual_probs + cell_probs)
    return 0.5 * (
        (visual_probs * (visual_probs.log() - middle.log())).sum(dim=1)
        + (cell_probs * (cell_probs.log() - middle.log())).sum(dim=1)
    )


def train_dual_probe(
    visual_features: np.ndarray,
    cell_features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    num_epochs: int,
    lr: float,
    device: torch.device,
    cell_valid: np.ndarray,
    weight_decay: float = 0.0,
    pool_visual_features: Optional[np.ndarray] = None,
    pool_cell_features: Optional[np.ndarray] = None,
    pool_consistency_weight: float = 0.0,
    pool_confidence_quantile: float = 0.5,
    pool_seed: int = 42,
) -> Tuple[LinearProbe, LinearProbe, Dict[str, float]]:
    """Fit the visual and cell probes jointly, optionally coupling them.

    The visual cross-entropy uses every labeled patch; the cell cross-entropy
    uses only patches that actually contain a detected nucleus. Updates are
    full-batch on purpose: active-learning budgets here are a few hundred
    points, and keeping two differently-sized labeled views aligned is simpler
    than pairing them inside mini-batches.

    **`pool_consistency_weight` is the one coupling term**, and it is
    semi-supervised: it uses the ~22k UNLABELED pool rows the probes never
    otherwise see, instead of the <=200 labeled ones. With only a few hundred
    labels, that pool is the largest unused resource in the round.

    A labeled-slice version (`consistency_weight`) existed here and was
    removed: it coupled the heads on the very rows they were already fitting,
    measured as doing almost nothing below weight ~5, and unlike the pool term
    it had no mask keeping it off `scalpel`'s acquisition signal.

    It is deliberately restricted to points where BOTH probes are already
    confident. That restriction is what keeps it from cannibalising
    acquisition: the pool-wide `JS(visual, cell)` IS `scalpel`'s per-point
    weight, so an unmasked pool consistency term would directly minimise the
    very quantity the sampler ranks on. Masking to the confident region means
    the term only sharpens agreement where the two views already agree, and
    leaves the contested region -- exactly the region `scalpel` wants to
    sample -- untouched. This is the standard confidence-thresholded
    consistency of semi-supervised learning (FixMatch and relatives), used
    here for the same reason.

    **Both knobs are RELATIVE, because this project runs three datasets whose
    scales differ.** `pool_confidence_quantile` is the share of each head's
    own confidence distribution that may enter the loss, not an absolute
    max-softmax cut: measured at budget 200, an absolute 0.9 admitted 67% of
    pathmnist (9 classes), 49% of histoset (14) and 42% of skintissue (16), so
    one number would have meant three different experiments.
    How many rows the term sees is not configurable at all -- it is half the
    pool, capped at `POOL_SUBSAMPLE_CAP`. The pools differ by 4.6x (22,400 vs
    100,000 and 103,495), so a fixed row count covered 36.6% of one and 8% of
    the others, while a fixed fraction cost 4.4x more on the large ones.

    The subsample is drawn ONCE, deterministically from `pool_seed`, not
    re-drawn per epoch: a fresh draw each epoch would make the loss
    non-stationary and defeat the early stopper, which compares consecutive
    epoch losses. Full-batch over a whole pool costs ~112x the labeled-only
    forward on histoset; 0.35 keeps that near 40x there.
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
    if pool_consistency_weight < 0.0:
        raise ValueError("pool_consistency_weight must be non-negative")
    if not 0.0 < pool_confidence_quantile <= 1.0:
        raise ValueError("pool_confidence_quantile must be in (0, 1]")
    if pool_consistency_weight > 0.0 and (
        pool_visual_features is None or pool_cell_features is None
    ):
        # Silently ignoring the weight would produce a run whose name and
        # config claim a semi-supervised term that never ran -- the same
        # mislabeled-baseline failure `aux_loss` without LoRA already refuses.
        raise ValueError(
            "pool_consistency_weight > 0 needs both pool_visual_features and "
            "pool_cell_features (the unlabeled rows the term is computed on)"
        )

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

    # Unlabeled pool rows for the semi-supervised term. Drawn ONCE, not per
    # epoch: a fresh subsample each epoch changes the loss surface between
    # epochs, and `_EarlyStopper` compares consecutive epoch losses, so it
    # would read sampling noise as failure to improve.
    pool_visual_x = pool_cell_x = None
    if pool_consistency_weight > 0.0:
        pv = np.asarray(pool_visual_features, dtype=np.float32)
        pc = np.asarray(pool_cell_features, dtype=np.float32)
        if len(pv) != len(pc):
            raise ValueError("pool_visual_features and pool_cell_features must align by row")
        if pv.shape[1] != visual_features.shape[1] or pc.shape[1] != cell_features.shape[1]:
            raise ValueError(
                "pool features must have the same width as the labeled features "
                "the probes are trained on"
            )
        # How many pool rows to use, derived from the pool itself rather than
        # configured. Neither fixed rule works alone across this project's
        # three datasets (pools of 22,400 / 100,000 / 103,495):
        #
        #   * a fixed ROW COUNT (8192) covered 36.6% of the small pool but
        #     only ~8% of the large ones -- three different experiments under
        #     one setting;
        #   * a fixed FRACTION (0.35) equalised coverage but cost 4.4x more on
        #     the large pools than on the small one.
        #
        # Half the pool, capped: coverage lands at 50% / 20% / 19% for at most
        # 2.4x the old cost. Generous where it is cheap, bounded where it is
        # not, and nothing for a caller to tune per dataset.
        take_n = min(len(pv) // 2, POOL_SUBSAMPLE_CAP)
        if 0 < take_n < len(pv):
            take = np.random.default_rng(pool_seed).choice(len(pv), take_n, replace=False)
            pv, pc = pv[take], pc[take]
        pool_visual_x = torch.as_tensor(pv, dtype=torch.float32, device=device)
        pool_cell_x = torch.as_tensor(pc, dtype=torch.float32, device=device)

    # Reported back to the caller so a run can never silently be a no-op: an
    # absolute confidence threshold is very dataset-dependent (a linear probe
    # on a few hundred labels over 14 classes may never exceed 0.14 on noisy
    # features, while histoset's real probe clears 0.9 on 54% of the pool), so
    # "the term was configured" and "the term did anything" are different
    # facts and the caller needs both.
    pool_mask_fraction = float("nan")

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
        if pool_visual_x is not None:
            pool_visual_probs = F.softmax(visual_probe(pool_visual_x), dim=1).clamp_min(1e-8)
            pool_cell_probs = F.softmax(cell_probe(pool_cell_x), dim=1).clamp_min(1e-8)
            # BOTH heads must be confident. The mask is detached: it selects
            # WHERE the penalty applies and must not itself be a path the
            # gradient can take -- otherwise the probes could lower the loss
            # by becoming less confident and shrinking the mask, rather than
            # by agreeing.
            # A QUANTILE of each head's own confidence, not an absolute cut.
            # Max-softmax scale depends on the number of classes and on how
            # separable they are: measured at budget 200, an absolute 0.9
            # admitted 67% of pathmnist (9 classes), 49% of histoset (14) and
            # 42% of skintissue (16) -- the same number meaning three
            # different things. A quantile admits the same SHARE of each
            # head's confident tail everywhere, so one setting is comparable
            # across datasets. (The intersection of the two heads' tails is
            # smaller than the quantile itself, which is intended: the term
            # should apply only where BOTH views are sure.)
            with torch.no_grad():
                visual_conf = pool_visual_probs.max(dim=1).values
                cell_conf = pool_cell_probs.max(dim=1).values
                cut = 1.0 - pool_confidence_quantile
                confident = (
                    (visual_conf >= torch.quantile(visual_conf, cut))
                    & (cell_conf >= torch.quantile(cell_conf, cut))
                )
            pool_mask_fraction = float(confident.float().mean())
            if confident.any():
                pool_penalty = _consistency_penalty(
                    pool_visual_probs[confident],
                    pool_cell_probs[confident],
                )
                loss = loss + pool_consistency_weight * pool_penalty.mean()
            # No `else`: early on, no pool point clears the threshold and the
            # term is simply absent for that epoch. Raising there would kill a
            # run for a condition that resolves itself as the probes train.
        loss.backward()
        optimizer.step()
        if stopper.step(float(loss.detach()), epoch):
            break
    return visual_probe, cell_probe, {"pool_mask_fraction": pool_mask_fraction}
