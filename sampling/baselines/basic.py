"""Random, margin and entropy sampling.

No reference implementation needed: these are the textbook baselines. Margin
takes the smallest top1-top2 gap, entropy the largest predictive entropy, both
from a probe refit on all labels revealed so far. Round 1 has no model, so it
draws at random.
"""

from typing import List

import numpy as np

from ..registry import register_sampler


@register_sampler("random")
def random_sampling(**kwargs) -> List[int]:
    """Uniform sample without replacement.

    Taken as a permutation prefix rather than `np.random.choice(..., B)` so
    that prefix-exactness holds BY CONSTRUCTION: the first B1 of a
    max-budget draw is exactly the B1 draw at the same seed. `choice` happens
    to behave the same way today, but only as an undocumented internal detail,
    and `sampling.specs` declares this sampler prefix-exact.
    """
    max_budget = kwargs["max_budget"]
    num_samples = kwargs["image_embeddings"].shape[0]
    return np.random.permutation(num_samples)[:max_budget].tolist()


@register_sampler("margin")
def margin_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 100)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]

    from training.probe import train_probe

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
            probe = train_probe(
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
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 100)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]

    from training.probe import train_probe

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
            probe = train_probe(
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
