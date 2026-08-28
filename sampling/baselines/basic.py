"""Random, margin and entropy sampling.

No reference implementation needed: these are the textbook baselines. Margin
takes the smallest top1-top2 gap, entropy the largest predictive entropy, both
from a probe refit on all labels revealed so far. Round 1 has no model, so it
draws at random.
"""

import time
from typing import List

import numpy as np

from utils.progress import Stopwatch
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
    trace = kwargs.get("trace")

    from training.probe import train_probe

    num_samples = image_embeddings.shape[0]
    step_budget = max(1, int(0.2 * max_budget))

    selected_indices: List[int] = []
    unlabeled_indices = list(range(num_samples))
    watch = Stopwatch(max_budget, "Margin")
    round_index = 0

    while len(selected_indices) < max_budget:
        current_need = min(step_budget, max_budget - len(selected_indices))
        round_started = time.time()
        if trace is not None:
            trace.start_round(round_index)
        round_scores = None

        if len(selected_indices) == 0:
            chosen_local = np.random.choice(len(unlabeled_indices), current_need, replace=False)
            chosen = [unlabeled_indices[i] for i in chosen_local]
            # Round 1 has no model, so there is no score to record. Leaving it
            # absent is the honest record; a placeholder would read downstream
            # as a real uncertainty of that value.
            chosen_scores = [None] * len(chosen)
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
            # Uncertainty, oriented so larger = more uncertain, matching every
            # other sampler here. The raw top1-top2 gap is the opposite.
            round_scores = 1.0 - margin_scores
            chosen_scores = [float(1.0 - margin_scores[i]) for i in best_local]
            del probe

        if trace is not None:
            for index, score in zip(chosen, chosen_scores):
                # `extra` values are coerced with float(), so an absent score
                # must be omitted rather than passed as None.
                extra = {} if score is None else {"uncertainty": score}
                trace.add_step(int(index), score=score, **extra)
            trace.add_round(
                num_selected=len(chosen),
                seconds=time.time() - round_started,
                scores=round_scores,
                weights=round_scores,
            )

        selected_indices.extend(chosen)
        chosen_set = set(chosen)
        unlabeled_indices = [i for i in unlabeled_indices if i not in chosen_set]
        watch.advance(len(chosen))
        watch.report()
        round_index += 1

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
    trace = kwargs.get("trace")

    from training.probe import train_probe

    num_samples = image_embeddings.shape[0]
    step_budget = max(1, int(0.2 * max_budget))

    selected_indices: List[int] = []
    unlabeled_indices = list(range(num_samples))
    watch = Stopwatch(max_budget, "Entropy")
    round_index = 0

    while len(selected_indices) < max_budget:
        current_need = min(step_budget, max_budget - len(selected_indices))
        round_started = time.time()
        if trace is not None:
            trace.start_round(round_index)
        round_scores = None

        if len(selected_indices) == 0:
            chosen_local = np.random.choice(len(unlabeled_indices), current_need, replace=False)
            chosen = [unlabeled_indices[i] for i in chosen_local]
            # No model in round 1, so no score exists to record -- see margin.
            chosen_scores = [None] * len(chosen)
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
            # Entropy is already oriented larger = more uncertain.
            round_scores = entropy_scores
            chosen_scores = [float(entropy_scores[i]) for i in best_local]
            del probe

        if trace is not None:
            for index, score in zip(chosen, chosen_scores):
                extra = {} if score is None else {"uncertainty": score}
                trace.add_step(int(index), score=score, **extra)
            trace.add_round(
                num_selected=len(chosen),
                seconds=time.time() - round_started,
                scores=round_scores,
                weights=round_scores,
            )

        selected_indices.extend(chosen)
        chosen_set = set(chosen)
        unlabeled_indices = [i for i in unlabeled_indices if i not in chosen_set]
        watch.advance(len(chosen))
        watch.report()
        round_index += 1

    return selected_indices
