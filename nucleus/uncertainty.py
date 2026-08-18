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