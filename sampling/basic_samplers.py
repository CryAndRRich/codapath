"""
Basic / classic active learning samplers.
Merged into one file to keep the sampling/ directory clean and avoid
naming conflicts with Python's built-in `random` module.

Methods:
  - random    : uniform random selection
  - margin    : iterative margin uncertainty (lowest top-2 gap)
  - entropy   : iterative entropy uncertainty
"""
from typing import List

import numpy as np

from . import register_sampler


@register_sampler("random")
def random_sampling(**kwargs) -> List[int]:
    """Uniform random selection — sliceable (run once at max_budget)."""
    image_embeddings = kwargs["image_embeddings"]
    max_budget = kwargs["max_budget"]
    num_samples = image_embeddings.shape[0]
    return np.random.choice(num_samples, max_budget, replace=False).tolist()


@register_sampler("margin")
def margin_sampling(**kwargs) -> List[int]:
    """
    Iterative margin uncertainty sampling.
    Warm-start with random step_budget samples, then greedily select the samples
    with smallest top-2 softmax gap (most uncertain) using an internal LinearProbe.
    """
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 100)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]

    from trainer import train_linear

    num_samples = image_embeddings.shape[0]
    step_budget = max(1, int(0.2 * max_budget))

    selected_indices: List[int] = []
    unlabeled_indices = list(range(num_samples))

    while len(selected_indices) < max_budget:
        current_need = min(step_budget, max_budget - len(selected_indices))

        if len(selected_indices) == 0:
            chosen_local = np.random.choice(len(unlabeled_indices), current_need, replace=False)
            chosen = [unlabeled_indices[i] for i in chosen_local]
        else:
            probe = train_linear(
                image_embeddings[selected_indices],
                oracle_labels[selected_indices],
                num_classes, probe_epochs, probe_lr, device,
            )
            probs = probe.predict_proba(image_embeddings[unlabeled_indices], device)
            sorted_probs = np.sort(probs, axis=1)
            margin_scores = sorted_probs[:, -1] - sorted_probs[:, -2]
            best_local = np.argsort(margin_scores)[:current_need]
            chosen = [unlabeled_indices[i] for i in best_local]
            del probe

        selected_indices.extend(chosen)
        chosen_set = set(chosen)
        unlabeled_indices = [i for i in unlabeled_indices if i not in chosen_set]

    return selected_indices


@register_sampler("entropy")
def entropy_sampling(**kwargs) -> List[int]:
    """
    Iterative entropy uncertainty sampling.
    Warm-start with random step_budget samples, then greedily select the samples
    with highest predictive entropy using an internal LinearProbe.
    """
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 100)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]

    from trainer import train_linear

    num_samples = image_embeddings.shape[0]
    step_budget = max(1, int(0.2 * max_budget))

    selected_indices: List[int] = []
    unlabeled_indices = list(range(num_samples))

    while len(selected_indices) < max_budget:
        current_need = min(step_budget, max_budget - len(selected_indices))

        if len(selected_indices) == 0:
            chosen_local = np.random.choice(len(unlabeled_indices), current_need, replace=False)
            chosen = [unlabeled_indices[i] for i in chosen_local]
        else:
            probe = train_linear(
                image_embeddings[selected_indices],
                oracle_labels[selected_indices],
                num_classes, probe_epochs, probe_lr, device,
            )
            probs = probe.predict_proba(image_embeddings[unlabeled_indices], device)
            entropy_scores = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            best_local = np.argsort(entropy_scores)[::-1][:current_need]
            chosen = [unlabeled_indices[i] for i in best_local]
            del probe

        selected_indices.extend(chosen)
        chosen_set = set(chosen)
        unlabeled_indices = [i for i in unlabeled_indices if i not in chosen_set]

    return selected_indices
