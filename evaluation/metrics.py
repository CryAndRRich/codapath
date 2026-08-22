"""Test-set metrics for a trained probe.

Every sampler is scored through this one function so the numbers stay
comparable: same frozen features, same probe, same macro-averaged metrics.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from training.probe import LinearProbe

__all__ = ["evaluate_probe"]


def evaluate_probe(
    probe: LinearProbe,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[float, float, float, float]:
    """Return `(accuracy, macro precision, macro recall, macro F1)`."""
    predictions = np.argmax(probe.predict_proba(test_features, device), axis=1)

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average="macro", zero_division=0)
    recall = recall_score(test_labels, predictions, average="macro", zero_division=0)
    f1 = f1_score(test_labels, predictions, average="macro", zero_division=0)

    if verbose:
        print(
            f"Accuracy {accuracy * 100:.2f}% | Precision {precision * 100:.2f}% | "
            f"Recall {recall * 100:.2f}% | Macro F1 {f1 * 100:.2f}%"
        )
    return accuracy, precision, recall, f1
