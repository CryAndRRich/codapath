"""TCM: TypiClust-then-Margin (ICLR 2024 Workshop).

No public reference implementation, so this follows the paper text directly
(`references/TCM.md`, and `pdfs/TCM_2403.03728.pdf`).

Phase 1 is TypiClust, phase 2 is Margin refit after every step. The paper's
"tiny" regime -- the one this project's budgets fall in -- sets "the initial
sample size equal to the number of classes" with "a step size equal to the
number of initial samples", and its heuristic performs "3 steps of TypiClust
before switching to Margin". Hence a transition at `3 * num_classes` and a
Margin step of `num_classes`, both budget-independent.

The TypiClust phase is a separate implementation from `typiclust.py` because it
starts from an empty labeled set: clusters are ranked by size alone, since the
"fewest existing labels first" key is constant at zero.
"""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans

from utils.runtime import clear_memory
from ..registry import register_sampler


def _typicality(features: torch.Tensor, indices: List[int], k_nn: int,
                chunk_size: int) -> np.ndarray:
    # Recomputed on the current (shrinking) candidate pool of a single cluster —
    # matches upstream TypiClust, where density estimates are relative to the
    # still-unlabeled members only, not a stale global/whole-cluster snapshot.
    n_c = len(indices)
    if n_c <= 1:
        return np.zeros(n_c, dtype=np.float32)

    feats = features[indices]
    k_val = max(1, min(k_nn, n_c // 2))  # upstream: min(K_NN, len(indices) // 2)
    mean_nn_dist = torch.zeros(n_c, device=features.device)

    for cs in range(0, n_c, chunk_size):
        ce = min(cs + chunk_size, n_c)
        chunk = feats[cs:ce]
        sim = torch.matmul(chunk, feats.T)
        dist = 1.0 - sim

        k_search = min(k_val + 1, n_c)
        topk_dist, _ = torch.topk(dist, k=k_search, dim=1, largest=False)
        mean_nn_dist[cs:ce] = topk_dist[:, 1:].mean(dim=1)

        del sim, dist, topk_dist

    return (1.0 / (mean_nn_dist.cpu().numpy() + 1e-5))


def _typiclust_phase(features: torch.Tensor,
                     num_select: int,
                     k_nn: int = 20,
                     chunk_size: int = 5000) -> List[int]:
    num_samples = features.shape[0]
    feats_np = features.cpu().numpy()

    if num_select <= 50:
        km = KMeans(n_clusters=num_select, n_init=10, random_state=42)
    else:
        km = MiniBatchKMeans(n_clusters=num_select, batch_size=5000,
                             n_init=3, random_state=42)

    cluster_ids = km.fit_predict(feats_np)
    cluster_sizes = np.bincount(cluster_ids, minlength=num_select)
    sorted_clusters = np.argsort(cluster_sizes)[::-1]
    valid_clusters = [c for c in sorted_clusters if cluster_sizes[c] > 5]  # upstream MIN_CLUSTER_SIZE
    if not valid_clusters:
        # see typiclust.py: MIN_CLUSTER_SIZE=5 assumes a large pool; on a small
        # pool/subsample every cluster can legitimately be smaller than that.
        valid_clusters = [c for c in sorted_clusters if cluster_sizes[c] > 0]
        print(f"[TCM WARNING] no cluster exceeds MIN_CLUSTER_SIZE=5 "
              f"(pool={num_samples}, num_clusters={num_select}); "
              f"falling back to all non-empty clusters")

    cluster_members = {
        c: [int(idx) for idx in np.where(cluster_ids == c)[0]]
        for c in valid_clusters
    }

    selected: List[int] = []
    selected_set: set = set()
    i = 0
    with tqdm(total=num_select, desc="TCM Phase-1 (TypiClust)", leave=False) as pbar:
        while len(selected) < num_select:
            cluster = valid_clusters[i % len(valid_clusters)]
            available = cluster_members[cluster]
            if available:
                typ = _typicality(features, available, k_nn, chunk_size)
                best = available[int(np.argmax(typ))]
                selected.append(best)
                selected_set.add(best)
                cluster_members[cluster] = [idx for idx in available if idx != best]
                pbar.update(1)
            i += 1
            if i > len(valid_clusters) * num_select:
                break

    if len(selected) < num_select:
        remaining = list(set(range(num_samples)) - selected_set)
        missing = num_select - len(selected)
        selected.extend(np.random.choice(remaining, missing, replace=False).tolist())

    return selected


@register_sampler("tcm")
def tcm_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 50)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]
    k_nn = kwargs.get("k_nn", 20)
    chunk_size = kwargs.get("chunk_size", 5000)
    # Paper Sec. 4.1: the "tiny" budget regime it defines has "the initial
    # sample size equal to the number of classes" and "a step size equal to the
    # number of initial samples". Its heuristic (Sec. 3.3) then performs "3
    # steps of TypiClust before switching to Margin" in the Tiny and Low
    # settings. This project's budgets (25..200 over 9-16 classes) sit squarely
    # in that regime, so the transition lands at 3 * num_classes and the Margin
    # phase steps by num_classes.
    #
    # Both multiples are class-relative and independent of max_budget, which is
    # what keeps the selection order budget-independent (see sampling.specs:
    # prefix-exact once budget >= 3 * num_classes).
    transition_multiple = int(kwargs.get("transition_class_multiple", 3))
    step_multiple = int(kwargs.get("step_class_multiple", 1))
    transition_budget = min(transition_multiple * num_classes, max_budget)

    from training.probe import train_probe

    features = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features = F.normalize(features, p=2, dim=1)

    phase1 = _typiclust_phase(features, transition_budget, k_nn=k_nn, chunk_size=chunk_size)

    if transition_budget >= max_budget:
        del features
        clear_memory()
        return phase1[:max_budget]

    remaining_budget = max_budget - transition_budget
    phase1_set = set(phase1)
    unlabeled_indices = [i for i in range(len(image_embeddings)) if i not in phase1_set]

    # Paper Sec. 4.1: the classifier is retrained "following each sampling
    # step", so Margin re-fits on the FULL accumulated labeled set at every
    # acquisition rather than scoring once and taking a single cut.
    step_budget = max(1, step_multiple * num_classes)
    phase2: List[int] = []

    while len(phase2) < remaining_budget:
        current_need = min(step_budget, remaining_budget - len(phase2))
        labeled_so_far = phase1 + phase2

        probe = train_probe(
            image_embeddings[labeled_so_far],
            oracle_labels[labeled_so_far],
            num_classes, probe_epochs, probe_lr, device,
        )
        probs = probe.predict_proba(image_embeddings[unlabeled_indices], device)
        s_probs = np.sort(probs, axis=1)
        margin = s_probs[:, -1] - s_probs[:, -2]
        order = np.argsort(margin)
        chosen = [unlabeled_indices[i] for i in order[:current_need]]
        del probe

        phase2.extend(chosen)
        chosen_set = set(chosen)
        unlabeled_indices = [i for i in unlabeled_indices if i not in chosen_set]

    del features
    clear_memory()

    return phase1 + phase2
