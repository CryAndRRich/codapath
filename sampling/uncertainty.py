"""Numerically stable uncertainty functions used by nucleus-aware AL."""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def margin_uncertainty_from_logits(logits: np.ndarray) -> np.ndarray:
    probabilities = softmax(logits)
    ordered = np.sort(probabilities, axis=1)
    return (1.0 - (ordered[:, -1] - ordered[:, -2])).astype(np.float32)


def js_disagreement_from_logits(
    first_logits: np.ndarray,
    second_logits: np.ndarray,
) -> np.ndarray:
    """Jensen-Shannon divergence normalized to [0, 1]."""
    first = np.clip(softmax(first_logits), 1e-8, 1.0)
    second = np.clip(softmax(second_logits), 1e-8, 1.0)
    middle = 0.5 * (first + second)
    kl_first = np.sum(first * (np.log(first) - np.log(middle)), axis=1)
    kl_second = np.sum(second * (np.log(second) - np.log(middle)), axis=1)
    jsd = 0.5 * (kl_first + kl_second) / np.log(2.0)
    return np.clip(jsd, 0.0, 1.0).astype(np.float32)


def row_layer_norm(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return ((values - mean) / (std + eps)).astype(np.float32)


def rank_normalize(values: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Map `values` onto [0, 1] by RANK, optionally within a sub-group.

    Returns an array the same length as `values`; entries outside `mask` are
    left at 0.0 and take no part in the ranking.

    **Why rank and not min-max.** PACT's acquisition weight mixes two
    quantities that live on wildly different scales: Jensen-Shannon
    disagreement between two probes (measured at 0.012-0.025 on a real
    14-class histoset run) and the visual margin (~0.7 for the same probe).
    Min-max was the obvious fix and is not enough -- it rescales the RANGE
    but not the SHAPE. JS is long-tailed, so a handful of outliers set its
    maximum and the bulk of the pool still lands near 0 after rescaling,
    while the near-uniform margin still averages ~0.5. Measured on the real
    pool shape (22400 patches, 9.24% with no nucleus), the fraction of the
    top-200 weights that were nucleus-free patches was:

        raw mixture          200/200      (a fair share would be ~18)
        min-max normalized   193/200
        rank normalized       17/200

    Ranking equalizes the distributions themselves, which is what makes the
    two terms comparable at all.

    **Ties share their average rank**, so a constant vector maps to a
    constant 0.5 rather than to an arbitrary ordering or to all-zeros. That
    matters here for the same reason CLAUDE.md's min-max-on-a-constant-vector
    lesson does: a degenerate input must produce a NEUTRAL weight, not a zero
    one that silently makes every marginal gain vanish and turns `argmax`
    into "return index order".
    """
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros(len(values), dtype=np.float32)
    if mask is None:
        selected = np.ones(len(values), dtype=bool)
    else:
        selected = np.asarray(mask, dtype=bool)
        if len(selected) != len(values):
            raise ValueError("mask must align with values")
    count = int(selected.sum())
    if count == 0:
        return out
    if count == 1:
        out[selected] = 0.5
        return out

    subset = values[selected]
    order = np.argsort(subset, kind="stable")
    ranks = np.empty(count, dtype=np.float64)
    ranks[order] = np.arange(count, dtype=np.float64)

    # Average the ranks inside each run of equal values, so ties -- and a
    # fully constant vector as the limiting case -- are neutral rather than
    # ordered by array position.
    ordered_values = subset[order]
    start = 0
    for index in range(1, count + 1):
        if index == count or ordered_values[index] != ordered_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index

    out[selected] = (ranks / (count - 1)).astype(np.float32)
    return out
