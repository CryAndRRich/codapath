"""
TCM: TypiClust → Margin transition.

Simple two-phase heuristic for pre-trained backbone settings:
  Phase 1 (diversity): TypiClust-style selection for `transition_budget` samples.
  Phase 2 (uncertainty): sort remaining unlabeled by Margin (ascending) after fitting
                         a LinearProbe on phase-1 selections.

The returned list is ordered [phase1..., phase2...] and is therefore **sliceable**:
  budget ≤ transition_budget  → first `budget` phase-1 elements
  budget >  transition_budget → all phase-1 + first (budget − transition_budget) phase-2 elements

Transition heuristic (from paper): total diversity budget ≈ 20 × num_classes,
so transition_budget = min(2 × num_classes, max_budget).  Configurable via
'transition_budget' kwarg.

Reference: Doucet et al., arXiv:2403.03728 (2024)
"""
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans

from set_up import clear_memory
from . import register_sampler


def _typiclust_phase(features: torch.Tensor,
                     num_select: int,
                     k_nn: int = 20,
                     chunk_size: int = 5000) -> List[int]:
    """
    Greedy typicality-based selection for `num_select` samples.
    Mirrors the TypiClust sampler: K-means(num_select) → most typical per cluster,
    clusters visited in decreasing size order (gives a natural priority ordering).
    """
    num_samples = features.shape[0]

    # Typicality = 1 / mean_kNN_dist  (chunked for memory)
    typicality = torch.zeros(num_samples, device=features.device)
    for cs in range(0, num_samples, chunk_size):
        ce = min(cs + chunk_size, num_samples)
        chunk = features[cs:ce]
        sim = torch.matmul(chunk, features.T)
        dist = 1.0 - sim
        k_val = min(k_nn + 1, num_samples)
        topk_dist, _ = torch.topk(dist, k=k_val, dim=1, largest=False)
        typicality[cs:ce] = 1.0 / (topk_dist[:, 1:].mean(dim=1) + 1e-5)
        del sim, dist, topk_dist, chunk
        clear_memory()

    typ_np = typicality.cpu().numpy()
    feats_np = features.cpu().numpy()

    if num_select <= 50:
        km = KMeans(n_clusters=num_select, n_init="auto", random_state=42)
    else:
        km = MiniBatchKMeans(n_clusters=num_select, batch_size=5000,
                             n_init="auto", random_state=42)

    cluster_ids = km.fit_predict(feats_np)
    cluster_sizes = np.bincount(cluster_ids, minlength=num_select)
    sorted_clusters = np.argsort(cluster_sizes)[::-1]
    valid_clusters = [c for c in sorted_clusters if cluster_sizes[c] > 0]

    selected: List[int] = []
    selected_set: set = set()
    i = 0
    with tqdm(total=num_select, desc="TCM Phase-1 (TypiClust)", leave=False) as pbar:
        while len(selected) < num_select:
            cluster = valid_clusters[i % len(valid_clusters)]
            in_cluster = np.where(cluster_ids == cluster)[0]
            available = [idx for idx in in_cluster if idx not in selected_set]
            if available:
                best = available[int(np.argmax(typ_np[available]))]
                selected.append(best)
                selected_set.add(best)
                pbar.update(1)
            i += 1
            if i > len(valid_clusters) * num_select:
                break

    # Fallback if clustering didn't fill quota
    if len(selected) < num_select:
        remaining = list(set(range(num_samples)) - selected_set)
        missing = num_select - len(selected)
        selected.extend(np.random.choice(remaining, missing, replace=False).tolist())

    return selected


@register_sampler("tcm")
def tcm_sampling(**kwargs) -> List[int]:
    """
    TCM: TypiClust → Margin transition — sliceable.

    Returns a single ordered list [phase1..., phase2...] that can be sliced
    at any cumulative budget.
    """
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 50)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]
    k_nn = kwargs.get("k_nn", 20)
    chunk_size = kwargs.get("chunk_size", 5000)
    # transition_budget: how many samples to collect via TypiClust before switching
    transition_budget = kwargs.get("transition_budget", min(2 * num_classes, max_budget))
    transition_budget = min(transition_budget, max_budget)

    from trainer import train_linear

    features = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features = F.normalize(features, p=2, dim=1)

    # ── Phase 1: TypiClust-style diversity selection ───────────────────────────
    phase1 = _typiclust_phase(features, transition_budget, k_nn=k_nn, chunk_size=chunk_size)

    if transition_budget >= max_budget:
        del features
        clear_memory()
        return phase1[:max_budget]

    # ── Phase 2: Margin uncertainty on remaining unlabeled ────────────────────
    remaining_budget = max_budget - transition_budget
    phase1_set = set(phase1)
    unlabeled_indices = [i for i in range(len(image_embeddings)) if i not in phase1_set]

    probe = train_linear(
        image_embeddings[phase1],
        oracle_labels[phase1],
        num_classes, probe_epochs, probe_lr, device,
    )
    probs = probe.predict_proba(image_embeddings[unlabeled_indices], device)
    s_probs = np.sort(probs, axis=1)
    margin = s_probs[:, -1] - s_probs[:, -2]
    # Sort ascending: smallest margin = most uncertain
    order = np.argsort(margin)
    phase2 = [unlabeled_indices[i] for i in order[:remaining_budget]]
    del probe, features
    clear_memory()

    return phase1 + phase2
