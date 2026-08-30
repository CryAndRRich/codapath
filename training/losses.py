"""Auxiliary representation losses for the final-training pass.

Every loss shares one signature -- `(features, logits, labels) -> scalar` --
so `training/finetune.py` can dispatch on `AUX_LOSS` without a special case
per loss. `logits` is accepted even though only `center_loss` ignores it,
because a caller building this call site once (not once per loss kind)
should not have to know which losses use which arguments.

**`supcon_loss` and `triplet_loss` RAISE on a batch with fewer than 2
samples of some present class**, rather than returning ~0. A silent 0 is the
failure this project has hit before with a different mechanism (see
CLAUDE.md's minmax-on-a-constant-vector lesson): the training loop would run
to completion, print a normal-looking loss curve, and the auxiliary loss
would have contributed nothing the whole time, with nothing to say so. This
project's budgets (25-200 labeled points) make a thin batch a realistic risk,
not a hypothetical one, which is exactly why `run_al_main.ipynb`'s assert
cell also checks `min(budget) >= 2 * num_classes` before a supcon/triplet run
is allowed to start (PLAN_IMPLEMENT.md §6.1) -- this raise is the second,
mechanism-level line of defense for the same problem, active on every batch,
not just the smallest configured budget.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["center_loss", "supcon_loss", "triplet_loss"]


def center_loss(
    features: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Mean squared distance from each feature to its OWN class's mean
    feature in the current batch (a per-batch center, not a maintained
    running center -- this project's final-training batches are the whole
    labeled set at once, so a running center across steps would need state
    this function has no reason to carry).

    Never raises on a thin batch: a class with exactly one member has
    distance zero to its own (single-point) mean, which is the geometrically
    correct answer, not a degenerate one -- unlike supcon/triplet, this loss
    needs no SECOND sample of the same class to have a well-defined value.
    """
    labels = labels.long()
    unique_labels = torch.unique(labels)
    centers = torch.stack([features[labels == c].mean(dim=0) for c in unique_labels])
    center_per_sample = centers[torch.searchsorted(unique_labels, labels)]
    return F.mse_loss(features, center_per_sample)


def supcon_loss(
    features: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020), single-view form:
    for each anchor, pulls it toward every OTHER sample of the same class
    and pushes it away from every sample of a different class, in the
    (L2-normalized) feature space.

    Raises `ValueError` if any class present in `labels` has fewer than 2
    members: an anchor with no positive (no other same-class sample in the
    batch) has an undefined "pull toward" term. The failure is silent
    without this check -- the pairwise-similarity sum over an empty positive
    set is exactly 0, so the training loop keeps running and reports a
    non-`nan` loss that quietly ignores that anchor's class the whole time.
    """
    labels = labels.long()
    counts = torch.bincount(labels)
    present = counts[counts > 0]
    if (present < 2).any():
        thin = [int(c) for c in torch.unique(labels) if counts[c] < 2]
        raise ValueError(
            f"supcon_loss: class(es) {thin} have fewer than 2 samples in this "
            f"batch (batch size {len(labels)}) -- an anchor with no same-class "
            "positive has no defined pull term. Use AUX_LOSS='center' if the "
            "budget cannot guarantee 2+ samples/class."
        )

    normalized = F.normalize(features, dim=1)
    similarity = torch.matmul(normalized, normalized.T) / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()  # stability

    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = same_class & ~self_mask

    exp_sim = torch.exp(similarity) * (~self_mask)
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts
    return -mean_log_prob_pos.mean()


def triplet_loss(
    features: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Batch-hard triplet loss (Hermans et al. 2017): for each anchor, the
    HARDEST positive (farthest same-class sample) and the HARDEST negative
    (closest different-class sample), in Euclidean feature space.

    Raises `ValueError` under the same condition as `supcon_loss` (some
    present class has fewer than 2 members) -- an anchor with no positive
    cannot select a hardest positive, and the same silent-zero failure mode
    applies: a masked-out max over an empty set is `-inf` unless guarded, and
    a naive guard that clamps it to 0 would silently drop that anchor from
    the loss rather than raising.
    """
    labels = labels.long()
    counts = torch.bincount(labels)
    if (counts[counts > 0] < 2).any():
        thin = [int(c) for c in torch.unique(labels) if counts[c] < 2]
        raise ValueError(
            f"triplet_loss: class(es) {thin} have fewer than 2 samples in this "
            f"batch (batch size {len(labels)}) -- an anchor with no same-class "
            "positive has no hardest-positive to select. Use AUX_LOSS='center' "
            "if the budget cannot guarantee 2+ samples/class."
        )

    pairwise = torch.cdist(features, features, p=2)
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = same_class & ~self_mask
    negative_mask = ~same_class

    hardest_positive = (pairwise * positive_mask).max(dim=1).values
    # Negatives use +inf where masked out so `.min` never picks a same-class
    # (or self) pair -- 0 would be wrong here (it is the diagonal's value,
    # not "far away").
    masked_negative = pairwise.masked_fill(~negative_mask, float("inf"))
    hardest_negative = masked_negative.min(dim=1).values

    return F.relu(hardest_positive - hardest_negative + margin).mean()
