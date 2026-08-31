"""Auxiliary representation losses for the final-training pass.

Every loss shares one signature -- `(features, logits, labels) -> scalar` --
so `training/finetune.py` can dispatch on `AUX_LOSS` without a special case
per loss. **All three currently ignore `logits`**; it is in the signature so
that a loss which does need it can be added without changing the one call
site, and so a caller building that site does not have to know which losses
read which arguments.

That every loss reads ONLY `features` has a consequence worth stating
plainly: an auxiliary loss can only do something if `features` carries a
gradient. On `finetune_and_evaluate`'s frozen-encoder path the features are
produced under `torch.no_grad()`, so an aux term there has no `grad_fn` at
all -- it is a constant added to the loss, contributing exactly nothing.
`finetune_and_evaluate` therefore refuses that combination rather than
running it; see its `aux_loss`/`use_lora` guard.

**Thin batches are normal here, and are handled by DROPPING the anchors that
have no positive -- not by raising.** `supcon_loss` and `triplet_loss` need a
same-class partner per anchor. An earlier version raised whenever any present
class had fewer than 2 members in the batch, which made both losses
unrunnable at this project's budgets: histoset has 14 classes, budget 25
selects only ~10 of them, and a random batch of 32 tripped the raise in 99%
of draws (measured). No batching strategy fixes that -- a class with one
sample in the whole labeled set has no positive in ANY batch. So those
anchors are excluded from the mean (the standard Khosla et al. formulation,
which sums over a positive set `P(i)` that is empty for them) while still
serving as negatives for other anchors. The losses raise only when NO anchor
has a positive, where the value is genuinely undefined.
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

    **Anchors with no positive are DROPPED from the mean, not silently
    counted as zero.** An anchor whose class has exactly one member in the
    batch has no "pull toward" term at all; including it would add a hard 0
    to the mean and quietly bias the loss downward in proportion to how many
    such anchors there are. Averaging only over anchors that HAVE a positive
    is the standard formulation (Khosla et al. 2020 Eq. 2 sums over
    `P(i)`, which is empty for those anchors).

    An earlier version raised `ValueError` instead. That was wrong for this
    project: at these budgets a singleton class is the NORM, not an error.
    histoset has 14 classes and budget 25 selects only ~10 of them, so with
    `batch_size=32` a randomly-drawn batch hit the raise in 99% of cases --
    making `supcon` and `triplet` unrunnable on the very sweep they were
    added for. And no batching strategy can fix it: a class with exactly one
    sample in the whole labeled set has no positive in ANY batch.

    Raises only when NO anchor in the batch has a positive (every class is a
    singleton), where the loss is genuinely undefined rather than merely
    thin.
    """
    labels = labels.long()

    normalized = F.normalize(features, dim=1)
    similarity = torch.matmul(normalized, normalized.T) / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()  # stability

    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = same_class & ~self_mask

    exp_sim = torch.exp(similarity) * (~self_mask)
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_counts = positive_mask.sum(dim=1)
    has_positive = positive_counts > 0
    if not bool(has_positive.any()):
        raise ValueError(
            f"supcon_loss: no anchor in this batch (size {len(labels)}) has a "
            "same-class positive -- every present class is a singleton, so the "
            "loss is undefined. Use AUX_LOSS='center', which needs no pair."
        )

    mean_log_prob_pos = (
        (positive_mask * log_prob).sum(dim=1)[has_positive]
        / positive_counts[has_positive]
    )
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

    **Anchors with no positive are DROPPED from the mean**, for the same
    reason and by the same rule as `supcon_loss` -- see its docstring for why
    raising here was wrong at this project's budgets (a singleton class is
    normal, not an error, and no batching strategy can manufacture a positive
    for a class with one sample in the whole labeled set).

    The drop is explicit, not the accidental `max`-over-empty-set value: a
    masked max would return 0 (the diagonal's value), which reads as "the
    hardest positive is at distance 0" and silently makes that anchor's
    triplet trivially satisfied.

    Raises only when NO anchor has a positive.
    """
    labels = labels.long()

    pairwise = torch.cdist(features, features, p=2)
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = same_class & ~self_mask
    negative_mask = ~same_class

    has_positive = positive_mask.any(dim=1)
    if not bool(has_positive.any()):
        raise ValueError(
            f"triplet_loss: no anchor in this batch (size {len(labels)}) has a "
            "same-class positive -- every present class is a singleton, so the "
            "loss is undefined. Use AUX_LOSS='center', which needs no pair."
        )

    hardest_positive = (pairwise * positive_mask).max(dim=1).values
    # Negatives use +inf where masked out so `.min` never picks a same-class
    # (or self) pair -- 0 would be wrong here (it is the diagonal's value,
    # not "far away").
    masked_negative = pairwise.masked_fill(~negative_mask, float("inf"))
    hardest_negative = masked_negative.min(dim=1).values

    triplets = F.relu(hardest_positive - hardest_negative + margin)
    return triplets[has_positive].mean()
